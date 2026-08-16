"""15개 대표 검증 러너 — 개선→평가 루프용 (GPU 2장 병렬).

산출 구조 (육안 검증이 쉽도록 폴더로 분리):
  review/<라운드>/brainco/<태스크>/<카메라>/{A_visible.mp4, B_amodal.mp4, AB_compare.png, stats.json}
  review/<라운드>/he/<카테고리>/{A_visible.mp4, B_amodal.mp4, AB_compare.png, stats.json}

방식 A = 보이는 부분만 (5단계 원문)
방식 B = 가려짐 보정 (크기 급감 시 이력 크기 유지 + 위치 추정)
1~4단계는 공통이라 한 번 처리로 두 결과를 동시에 낸다.

사용법:
  python run_review.py bc <라운드> <샤드idx> <샤드수>
  python run_review.py he <라운드> <샤드idx> <샤드수>
"""
import os, sys, json, glob, time, shutil, subprocess, traceback
import numpy as np
import cv2
import torch

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
from geometry import Intrinsics
from track3d import EpisodeTracker
from profiles import profile_for, describe
from spec import BRAINCO, BRAINCO_CAMS, HE_REP

DATA = "/data2/humanoid_dataset_isangmin"
HE_ROOT = f"{DATA}/humanoid-everyday"
DEV = "cuda:0"
REVIEW = f"{ROOT}/review"
LOGD = f"{ROOT}/v2/logs"
os.makedirs(LOGD, exist_ok=True)

# 검증 단계는 품질 우선 — 10fps로 촘촘히 본다 (원본 30fps)
STRIDE = 3
BC_MAX_FRAMES = 220          # 카메라 4대 x 8태스크라 상한을 둔다
HE_MAX_FRAMES = 200
# 에피소드를 상수로 고정하면 "다른 에피소드에서도 되는가"를 원리적으로 측정할 수 없다.
# 튜닝은 EP_TUNE에서만 하고, EP_HOLDOUT은 파라미터 결정에 쓰지 않은 채 점수만 매긴다.
BC_EP = int(os.environ.get("BC_EP", "5"))
HE_EP_OFFSET = int(os.environ.get("HE_EP_OFFSET", "0"))   # 카테고리 대표 ep에서 몇 번째 뒤


def log(msg, tag="review"):
    line = f"[{time.strftime('%F %T')}][{tag}] {msg}"
    print(line, flush=True)
    with open(f"{LOGD}/{tag}.log", "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ 로더
def frames_brainco(task, ep, cam, stride=STRIDE, cap_n=BC_MAX_FRAMES):
    import pandas as pd
    root = f"{DATA}/G1_Brainco_{task}_Dataset"
    epm = pd.read_parquet(sorted(glob.glob(f"{root}/meta/episodes/chunk-000/*.parquet"))[0])
    rows = epm[epm.episode_index == ep]
    if len(rows) == 0:
        return [], None
    r = rows.iloc[0]
    key = f"videos/observation.images.{cam}"
    if f"{key}/file_index" not in r:
        return [], None
    fi = int(r[f"{key}/file_index"])
    t0, t1 = float(r[f"{key}/from_timestamp"]), float(r[f"{key}/to_timestamp"])
    mp4 = f"{root}/videos/observation.images.{cam}/chunk-000/file-{fi:03d}.mp4"
    tmp = f"/tmp/rv_{task}_{cam}_{ep}_{os.getpid()}"
    os.makedirs(tmp, exist_ok=True); os.system(f"rm -f {tmp}/*.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0), "-t", str(t1 - t0),
                    "-i", mp4, "-vf", f"fps={30.0/stride}", f"{tmp}/f%05d.png"], check=True)
    files = sorted(glob.glob(f"{tmp}/f*.png"))
    if cap_n and len(files) > cap_n:
        files = [files[i] for i in np.linspace(0, len(files) - 1, cap_n).astype(int)]
    return files, tmp


def frames_he(ep, stride=STRIDE, cap_n=HE_MAX_FRAMES):
    import pyarrow.parquet as pq
    ch = ep // 1000
    tbl = pq.read_table(f"{HE_ROOT}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet",
                        columns=["observation.depth.egocentric"])
    n = tbl.num_rows
    want = list(range(0, n, stride))
    if cap_n and len(want) > cap_n:
        want = [want[i] for i in np.linspace(0, len(want) - 1, cap_n).astype(int)]
    wset = set(want)
    darr = tbl["observation.depth.egocentric"]
    cap = cv2.VideoCapture(f"{HE_ROOT}/videos/chunk-{ch:03d}/egocentric/episode_{ep:06d}.mp4")
    out, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in wset and i < n:
            out.append((img, np.array(darr[i].as_py(), dtype=np.float32)))
        i += 1
    cap.release()
    return out


# ------------------------------------------------------------------ 처리
def run_one(outdir, seq, prompt, targets, det, seg, dep, is_he, task_name=""):
    """한 (에피소드 x 카메라) 처리. A/B 영상 2편 + 비교 이미지 + 통계 저장."""
    os.makedirs(outdir, exist_ok=True)
    prof = profile_for(task_name)
    trk = EpisodeTracker(targets, det, seg, prompt, fps=30.0 / STRIDE, profile=prof)
    H, W = seq[0][0].shape[:2]
    fps = 30.0 / STRIDE
    vA = cv2.VideoWriter(f"{outdir}/A_visible.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    vB = cv2.VideoWriter(f"{outdir}/B_amodal.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    recs, snapA, snapB, snap_occ = [], None, None, None
    for i, (img, gt_depth) in enumerate(seq):
        if is_he:
            depth = gt_depth
            if depth.ndim == 1:
                depth = depth.reshape(depth.size // W, W)
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            K = Intrinsics.from_fov(W, H, 70.0)
            dscale, adx = 1e-3, -20
        else:
            depth, K_pred = dep(img)
            K = (Intrinsics(float(K_pred[0, 0]), float(K_pred[1, 1]),
                            float(K_pred[0, 2]), float(K_pred[1, 2]))
                 if K_pred is not None else Intrinsics.from_fov(W, H, 70.0))
            dscale, adx = 1.0, 0
        rs = trk.step(img, depth, K, dscale, align_dx=adx)
        visA = trk.draw(img.copy(), rs, K, mode="A")
        visB = trk.draw(img.copy(), rs, K, mode="B")
        vA.write(visA); vB.write(visB)
        recs.append(dict(f=i, r=[{k: v for k, v in x.items() if k != "_b"} for x in rs]))
        if i == len(seq) // 2:
            snapA, snapB = visA.copy(), visB.copy()
        if snap_occ is None and any(x.get("occluded") for x in rs):
            snap_occ = np.hstack([visA, visB])          # 가림 순간의 A/B 대비
        for x in rs:
            x.pop("_b", None)
    vA.release(); vB.release()
    if snapA is not None:
        cv2.imwrite(f"{outdir}/AB_compare.png", np.hstack([snapA, snapB]))
    if snap_occ is not None:
        cv2.imwrite(f"{outdir}/AB_occluded.png", snap_occ)
    st = trk.stats(len(seq))
    json.dump(recs, open(f"{outdir}/frames.json", "w"), ensure_ascii=False, default=str)
    json.dump(dict(prompt=trk.prompt, n_frames=len(seq), fps=fps, stride=STRIDE, stats=st),
              open(f"{outdir}/stats.json", "w"), ensure_ascii=False, indent=1, default=str)
    return st


def main():
    kind = sys.argv[1]
    rnd = sys.argv[2] if len(sys.argv) > 2 else "r1"
    shard = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    nshard = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    tag = f"{kind}{shard}"
    from models_wrap import Detector, Segmenter, DepthEstimator
    t0 = time.time()
    det, seg = Detector(DEV), Segmenter(DEV)
    dep = DepthEstimator(DEV) if kind == "bc" else None
    log(f"모델 로드 {time.time()-t0:.0f}s", tag)
    torch.cuda.reset_peak_memory_stats()

    jobs = []
    if kind == "bc":
        for task, spec in BRAINCO.items():
            for cam in BRAINCO_CAMS:
                jobs.append((task, cam, spec))
    else:
        for cat, spec in HE_REP.items():
            jobs.append((cat, "egocentric", spec))
    jobs = [j for i, j in enumerate(jobs) if i % nshard == shard]
    log(f"{kind} 담당 {len(jobs)}건", tag)

    t_all = time.time(); nfr = 0
    for name, cam, spec in jobs:
        epn = BC_EP if kind == "bc" else (spec["ep"] + HE_EP_OFFSET)
        base = f"{REVIEW}/{rnd}/{'brainco' if kind=='bc' else 'he'}/{name}_ep{epn}"
        outdir = f"{base}/{cam}" if kind == "bc" else base
        if os.path.exists(f"{outdir}/stats.json"):
            log(f"  건너뜀(완료) {name}/{cam}", tag); continue
        try:
            t = time.time()
            if kind == "bc":
                files, tmp = frames_brainco(name, BC_EP, cam)
                if not files:
                    log(f"  프레임 없음 {name}/{cam}", tag); continue
                seq = [(cv2.imread(f), None) for f in files]
            else:
                seq = frames_he(spec["ep"] + HE_EP_OFFSET)
                tmp = None
                if not seq:
                    log(f"  프레임 없음 {name}", tag); continue
            st = run_one(outdir, seq, spec["prompt"], spec["targets"], det, seg, dep,
                         kind == "he", task_name=name)
            log(f"    프로파일: {describe(name)}", tag)
            nfr += len(seq)
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
            brief = {k: f"{v['accepted']}/{len(seq)} ({v['coverage']*100:.0f}%)"
                     + (f" 가림{v['occluded_frames']}" if v['occluded_frames'] else "")
                     for k, v in st.items()}
            log(f"  {name}/{cam} {len(seq)}f {time.time()-t:.0f}s {brief}", tag)
        except Exception as e:
            log(f"  실패 {name}/{cam}: {str(e)[:150]}", tag)
            with open(f"{LOGD}/fail_{tag}.log", "a") as f:
                f.write(f"{name}/{cam}\n{traceback.format_exc()}\n")
    el = time.time() - t_all
    log(f"완료 — {nfr}프레임 {el:.0f}s ({el/max(nfr,1):.2f}s/f), "
        f"peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB", tag)


if __name__ == "__main__":
    main()
