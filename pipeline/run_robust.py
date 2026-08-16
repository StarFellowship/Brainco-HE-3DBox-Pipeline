"""robustness 검증 러너 — 모든 태스크를 태스크당 n개 에피소드로 실행.

목적: 태스크당 에피소드 1개로 확인한 파이프라인이 **다른 에피소드에서도 같은 품질**을
내는지 본다. 커버리지가 아니라 **에피소드 간 편차**가 관심사다.

설계:
  - 작업 단위 = (태스크, 에피소드, 카메라). 결과 폴더가 있으면 건너뛰므로 중단 시 이어서 진행.
  - 에피소드는 seed 고정 무작위 샘플링 — 같은 표본이 재현된다.
  - 태스크별 프로파일(profiles.py)이 자동 적용된다.

사용법:
  python run_robust.py <라운드> [샤드idx] [샤드수]
  환경변수: BC_PER_TASK, HE_PER_TASK, FPS, BC_CAP, HE_CAP, CAMS(쉼표구분)
"""
import os, sys, json, glob, time, random, shutil, subprocess, traceback
import numpy as np
import cv2
import torch

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
from geometry import Intrinsics
from track3d import EpisodeTracker
from spec import BRAINCO, HE_REP
from profiles import profile_for, describe

DATA = "/data2/humanoid_dataset_isangmin"
HE_ROOT = f"{DATA}/humanoid-everyday"
DEV = "cuda:0"
OUTROOT = f"{ROOT}/robust"
LOGD = f"{ROOT}/v2/logs"
os.makedirs(LOGD, exist_ok=True)

BC_PER_TASK = int(os.environ.get("BC_PER_TASK", "4"))
HE_PER_TASK = int(os.environ.get("HE_PER_TASK", "2"))
FPS = float(os.environ.get("FPS", "6"))
BC_CAP = int(os.environ.get("BC_CAP", "200"))
HE_CAP = int(os.environ.get("HE_CAP", "90"))
CAMS = os.environ.get("CAMS", "cam_left_high,cam_right_high").split(",")
SEED = int(os.environ.get("SEED", "20260805"))
STRIDE = max(1, int(round(30.0 / FPS)))


def log(msg, tag="robust"):
    line = f"[{time.strftime('%F %T')}][{tag}] {msg}"
    print(line, flush=True)
    with open(f"{LOGD}/{tag}.log", "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ 작업 목록
def build_jobs():
    """(데이터셋, 태스크, 에피소드, 카메라, 프롬프트, targets) 목록. seed 고정."""
    import pandas as pd
    rng = random.Random(SEED)
    jobs = []

    for task, sp in BRAINCO.items():
        root = f"{DATA}/G1_Brainco_{task}_Dataset"
        f = sorted(glob.glob(f"{root}/meta/episodes/chunk-000/*.parquet"))
        if not f:
            continue
        eps = sorted(int(x) for x in pd.read_parquet(f[0])["episode_index"])
        # 대표 에피소드(5)는 항상 포함해 이전 라운드와 비교 가능하게 한다
        pick = ([5] if 5 in eps else []) + rng.sample([e for e in eps if e != 5],
                                                      min(BC_PER_TASK - 1, len(eps) - 1))
        for ep in sorted(set(pick)):
            for cam in CAMS:
                jobs.append(dict(ds="bc", task=task, ep=ep, cam=cam,
                                 prompt=sp["prompt"], targets=sp["targets"]))

    # HE: 카테고리 대표 태스크의 다른 에피소드들
    import collections
    tasks_meta = [json.loads(l) for l in open(f"{HE_ROOT}/meta/tasks.jsonl")]
    ti2cat = {t["task_index"]: t["category"] for t in tasks_meta}
    eps_meta = [json.loads(l) for l in open(f"{HE_ROOT}/meta/episodes.jsonl")]
    by_ti = collections.defaultdict(list)
    for e in eps_meta:
        if e.get("robot_type") != "g1":
            continue
        ep = e["episode_index"]; ch = ep // 1000
        if os.path.exists(f"{HE_ROOT}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet"):
            by_ti[e["tasks"][0]].append(e)

    for cat, sp in HE_REP.items():
        # 이 카테고리에 속한 모든 g1 태스크에서 균등 샘플링 (카테고리 대표만이 아니라
        # 카테고리 전체를 훑어야 '모든 종류'를 덮는다)
        tis = [ti for ti, c in ti2cat.items() if c == cat and ti in by_ti]
        if not tis:
            continue
        picked = []
        for ti in sorted(tis):
            cand = [e for e in by_ti[ti] if 200 <= e["length"] <= 800]
            if cand:
                picked.append(rng.choice(cand)["episode_index"])
        # 대표 에피소드는 항상 포함
        base = [sp["ep"]] if sp["ep"] not in picked else []
        rng.shuffle(picked)
        sel = base + picked[: max(0, HE_PER_TASK * max(len(tis), 1) - len(base))]
        for ep in sorted(set(sel)):
            jobs.append(dict(ds="he", task=cat, ep=ep, cam="egocentric",
                             prompt=sp["prompt"], targets=sp["targets"]))
    return jobs


# ------------------------------------------------------------------ 로더
def frames_bc(task, ep, cam):
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
    tmp = f"/tmp/rb_{task}_{cam}_{ep}_{os.getpid()}"
    os.makedirs(tmp, exist_ok=True); os.system(f"rm -f {tmp}/*.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0), "-t", str(t1 - t0),
                    "-i", mp4, "-vf", f"fps={FPS}", f"{tmp}/f%05d.png"], check=True)
    files = sorted(glob.glob(f"{tmp}/f*.png"))
    if BC_CAP and len(files) > BC_CAP:
        files = [files[i] for i in np.linspace(0, len(files) - 1, BC_CAP).astype(int)]
    return files, tmp


def frames_he(ep):
    import pyarrow.parquet as pq
    ch = ep // 1000
    tbl = pq.read_table(f"{HE_ROOT}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet",
                        columns=["observation.depth.egocentric"])
    n = tbl.num_rows
    want = list(range(0, n, STRIDE))
    if HE_CAP and len(want) > HE_CAP:
        want = [want[i] for i in np.linspace(0, len(want) - 1, HE_CAP).astype(int)]
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
def run_unit(outdir, seq, job, det, seg, dep):
    os.makedirs(outdir, exist_ok=True)
    prof = profile_for(job["task"])
    trk = EpisodeTracker(job["targets"], det, seg, job["prompt"], fps=FPS, profile=prof)
    is_he = job["ds"] == "he"
    H, W = seq[0][0].shape[:2]
    vA = cv2.VideoWriter(f"{outdir}/A_visible.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    vB = cv2.VideoWriter(f"{outdir}/B_amodal.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    recs, snap = [], None
    for i, (img, gt) in enumerate(seq):
        if is_he:
            depth = gt
            if depth.ndim == 1:
                depth = depth.reshape(depth.size // W, W)
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            K = Intrinsics.from_fov(W, H, 70.0); dscale, adx = 1e-3, -20
        else:
            depth, Kp = dep(img)
            K = (Intrinsics(float(Kp[0, 0]), float(Kp[1, 1]), float(Kp[0, 2]), float(Kp[1, 2]))
                 if Kp is not None else Intrinsics.from_fov(W, H, 70.0))
            dscale, adx = 1.0, 0
        rs = trk.step(img, depth, K, dscale, align_dx=adx)
        a = trk.draw(img.copy(), rs, K, mode="A")
        b = trk.draw(img.copy(), rs, K, mode="B")
        vA.write(a); vB.write(b)
        recs.append(dict(f=i, r=[{k: v for k, v in x.items() if k != "_b"} for x in rs]))
        if i == len(seq) // 2:
            snap = np.hstack([a, b])
    vA.release(); vB.release()
    if snap is not None:
        cv2.imwrite(f"{outdir}/AB_compare.png", snap)
    st = trk.stats(len(seq))
    json.dump(recs, open(f"{outdir}/frames.json", "w"), ensure_ascii=False, default=str)
    json.dump(dict(task=job["task"], ep=job["ep"], cam=job["cam"], prompt=trk.prompt,
                   n_frames=len(seq), fps=FPS, profile=describe(job["task"]), stats=st),
              open(f"{outdir}/stats.json", "w"), ensure_ascii=False, indent=1, default=str)
    return st


def main():
    rnd = sys.argv[1] if len(sys.argv) > 1 else "rb1"
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nshard = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    tag = f"rb{shard}"
    jobs = build_jobs()
    jobs = [j for i, j in enumerate(jobs) if i % nshard == shard]
    nb = sum(1 for j in jobs if j["ds"] == "bc")
    log(f"작업 {len(jobs)}건 (Brainco {nb} / HE {len(jobs)-nb}), "
        f"fps={FPS} BC{BC_PER_TASK}ep HE{HE_PER_TASK}ep cams={len(CAMS)}", tag)

    from models_wrap import Detector, Segmenter, DepthEstimator
    t0 = time.time()
    det, seg = Detector(DEV), Segmenter(DEV)
    dep = DepthEstimator(DEV)
    log(f"모델 로드 {time.time()-t0:.0f}s", tag)
    torch.cuda.reset_peak_memory_stats()

    t_all = time.time(); ok = fail = skip = nfr = 0
    for n, j in enumerate(jobs, 1):
        sub = "brainco" if j["ds"] == "bc" else "he"
        outdir = f"{OUTROOT}/{rnd}/{sub}/{j['task']}/ep{j['ep']:05d}_{j['cam']}"
        if os.path.exists(f"{outdir}/stats.json"):
            skip += 1; continue
        tmp = None
        try:
            t = time.time()
            if j["ds"] == "bc":
                files, tmp = frames_bc(j["task"], j["ep"], j["cam"])
                if not files:
                    log(f"  프레임 없음 {j['task']}/ep{j['ep']}/{j['cam']}", tag); continue
                seq = [(cv2.imread(f), None) for f in files]
            else:
                seq = frames_he(j["ep"])
                if not seq:
                    log(f"  프레임 없음 HE/{j['task']}/ep{j['ep']}", tag); continue
            st = run_unit(outdir, seq, j, det, seg, dep)
            nfr += len(seq); ok += 1
            brief = {k: f"{v['coverage']*100:.0f}%" for k, v in st.items() if v.get("is_target")}
            el = time.time() - t_all
            eta = (el / max(ok, 1)) * (len(jobs) - skip - ok) / 3600
            log(f"[{n}/{len(jobs)}] {j['task']}/ep{j['ep']}/{j['cam']} "
                f"{len(seq)}f {time.time()-t:.0f}s {brief} | ETA {eta:.1f}h", tag)
        except Exception as e:
            fail += 1
            log(f"  실패 {j['task']}/ep{j['ep']}/{j['cam']}: {str(e)[:120]}", tag)
            with open(f"{LOGD}/fail_{tag}.log", "a") as f:
                f.write(f"{j}\n{traceback.format_exc()}\n")
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
    el = time.time() - t_all
    log(f"완료 — 성공 {ok} 건너뜀 {skip} 실패 {fail}, {nfr}프레임 {el/3600:.1f}h, "
        f"peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB", tag)


if __name__ == "__main__":
    main()
