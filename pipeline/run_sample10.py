"""task당 10에피소드 샘플 러너 — 조작 대상 + 로봇 부위(+참조·대형 배경) 통합 산출.

run_review.py 를 복사·확장했다 (기존 파일 무수정 원칙 — 로더는 run_review 에서 임포트).

## 구조

- 조작 대상·기존 참조물: 검증된 track3d.EpisodeTracker 를 **그대로** 쓴다 (게이트 불변).
- 로봇 부위(robot arm/gripper)·추가 참조물·대형 배경: multiobj.ExtraTracker 의
  **별도 2차 검출 패스** — 카테고리별 게이트(robot_part 0.90/1.5m/1.5m,
  background 0.50/2.5m)가 조작 대상 게이트와 코드 경로부터 분리된다.
- 마스크 산출: EpisodeTracker 는 내부 마스크를 반환하지 않으므로, 승인된 대상 박스와
  2차 패스 후보 박스를 **한 번의 SAM 호출로 일괄** 재분할해 RLE 를 얻는다.
  (프레임당 비용: det 2회 + SAM 2회 — 샘플 러너라 감당 가능한 수준)

## 산출 스키마 (frames.json)

  {"frame": N, "objects": [{"label", "category": target|robot_part|reference|background,
    "side": left|right|none, "src": det|track, "box2d": [x0,y0,x1,y1],
    "mask_rle": {"size":[H,W],"counts":...}(COCO 압축 RLE),
    "box3d": {"center":[x,y,z],"size":[w,h,d]} 또는 null(게이트 미통과), "score": float,
    "reason": 게이트 미통과 사유(통과 시 "")}]}

  src: "det"=GroundingDINO 가 이 프레임에서 찾음(저임계 재검출 포함),
       "track"=SAM 2.1 이 직전 mask/박스를 이어 추적 (기존 코드의 "prop" 명칭 대체).
  box3d 는 방식 A(관측 기반) 값이다. 기존 트래커의 원시 레코드(방식 B 포함)는
  frames_raw.json 에 그대로 보존한다.

## 에피소드 샘플링 (재현 가능)

- 각 task 의 에피소드 목록을 정렬 후 random.Random(42).sample 로 10개 고정.
  10개 미만이면 전부. task 마다 rng 를 새로 만들어 task 추가·삭제가 표본을 흔들지 않는다.
- Brainco: 머리 2캠(cam_left_high/right_high) x 10ep 전량 + 손목 2캠은 표본의 앞 3ep 만.
- HE: 카테고리 대표 태스크(run_review 의 HE_REP 방식)와 같은 task 의 g1 에피소드에서 10개.

## 사용법 (환경변수)

  SPLIT=bc1|bc2|he   실행 범위 (3 GPU 분산, 인터리브 분할로 부하 균형):
                     bc1=BRAINCO 인덱스 0,2,4,6 (GraspOreo/PickApple/PickDoll/PickTissues)
                     bc2=BRAINCO 인덱스 1,3,5,7 (GraspRubiksCube/PickCharger/PickDrink/PickToothpaste)
                     he =HE 7카테고리 전부
  RND=s10            산출 폴더명 -> ~/task3/sample10/<RND>/<MODE>/...
  MODE=task|all      task=조작대상(spec.py)+로봇 부위 / all=+참조·대형 배경(spec_allobj.py)
  SMOKE=1            스모크 테스트 — 앞 2유닛 x 60프레임만
  STRIDE, BC_CAP, HE_CAP  프레임 밀도·상한 (기본 3 / 220 / 200 — run_review 와 동일)

  예: SPLIT=he RND=s10 MODE=all python run_sample10.py
"""
import os, sys, json, time, random, shutil, traceback
import numpy as np
import cv2
import torch

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
from geometry import Intrinsics
from track3d import EpisodeTracker
from profiles import profile_for, describe
from spec import BRAINCO, HE_REP
from spec_allobj import extra_spec
from multiobj import ExtraTracker, mask_to_rle, draw_objects
import run_review as RV                 # 로더 재사용 (frames_brainco / frames_he)

DATA = "/data2/humanoid_dataset_isangmin"
HE_ROOT = f"{DATA}/humanoid-everyday"
DEV = "cuda:0"
OUTROOT = f"{ROOT}/sample10"
LOGD = f"{ROOT}/v2/logs"
os.makedirs(LOGD, exist_ok=True)

SPLIT = os.environ.get("SPLIT", "bc1")          # bc1 | bc2 | he
RND = os.environ.get("RND", "s10")              # 산출 폴더명
MODE = os.environ.get("MODE", "all")            # task | all
SMOKE = os.environ.get("SMOKE", "0") == "1"     # 앞 2유닛 x 60프레임
STRIDE = int(os.environ.get("STRIDE", "3"))     # 10fps (원본 30fps)
BC_CAP = int(os.environ.get("BC_CAP", "220"))
HE_CAP = int(os.environ.get("HE_CAP", "200"))

SEED = 42                                       # 에피소드 샘플링 seed (고정)
N_EP = 10                                       # task 당 에피소드 수
WRIST_EP = 3                                    # 손목캠은 표본의 앞 3개만
HEAD_CAMS = ["cam_left_high", "cam_right_high"]
WRIST_CAMS = ["cam_left_wrist", "cam_right_wrist"]

if SMOKE:
    BC_CAP = min(BC_CAP, 60)
    HE_CAP = min(HE_CAP, 60)


def log(msg, tag=None):
    tag = tag or f"s10_{SPLIT}"
    line = f"[{time.strftime('%F %T')}][{tag}] {msg}"
    print(line, flush=True)
    with open(f"{LOGD}/{tag}.log", "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ 에피소드 샘플링
def sample_bc_eps(task):
    """Brainco task 의 에피소드 목록을 정렬 -> seed 42 로 10개 고정 샘플."""
    import glob
    import pandas as pd
    root = f"{DATA}/G1_Brainco_{task}_Dataset"
    files = sorted(glob.glob(f"{root}/meta/episodes/chunk-000/*.parquet"))
    if not files:
        return []
    eps = sorted(int(x) for x in pd.read_parquet(files[0])["episode_index"].unique())
    if len(eps) <= N_EP:
        return eps
    return sorted(random.Random(SEED).sample(eps, N_EP))


def sample_he_eps(cat):
    """HE 카테고리 대표 태스크(HE_REP)와 같은 task 의 g1 에피소드에서 10개 고정 샘플.
    대표 ep 의 task_index 를 episodes.jsonl 에서 역추적한다 (run_robust 와 같은 메타 활용)."""
    rep = int(HE_REP[cat]["ep"])
    try:
        eps_meta = [json.loads(l) for l in open(f"{HE_ROOT}/meta/episodes.jsonl")]
    except Exception as e:
        log(f"  HE 메타 읽기 실패({cat}): {str(e)[:80]} -> 대표 ep 만 사용")
        return [rep]
    rep_row = next((e for e in eps_meta if e["episode_index"] == rep), None)
    if rep_row is None:
        return [rep]
    ti = rep_row["tasks"][0]
    eps = []
    for e in eps_meta:
        if e.get("robot_type") != "g1" or e["tasks"][0] != ti:
            continue
        ep = int(e["episode_index"])
        ch = ep // 1000
        if os.path.exists(f"{HE_ROOT}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet"):
            eps.append(ep)
    eps = sorted(set(eps))
    if not eps:
        return [rep]
    if len(eps) <= N_EP:
        return eps
    return sorted(random.Random(SEED).sample(eps, N_EP))


def build_jobs(split):
    """(kind, task, ep, cam) 목록. 결과 폴더가 있으면 실행 시 건너뛴다(재개 가능)."""
    jobs = []
    if split in ("bc1", "bc2"):
        names = list(BRAINCO.keys())
        pick = names[0::2] if split == "bc1" else names[1::2]
        for task in pick:
            eps = sample_bc_eps(task)
            if not eps:
                log(f"  에피소드 메타 없음: {task}")
                continue
            for cam in HEAD_CAMS:
                for ep in eps:
                    jobs.append(("bc", task, ep, cam))
            for cam in WRIST_CAMS:
                for ep in eps[:WRIST_EP]:
                    jobs.append(("bc", task, ep, cam))
    elif split == "he":
        for cat in HE_REP:
            for ep in sample_he_eps(cat):
                jobs.append(("he", cat, ep, "egocentric"))
    else:
        raise SystemExit(f"SPLIT={split} 는 bc1|bc2|he 중 하나여야 한다")
    return jobs


# ------------------------------------------------------------------ 유닛 처리
def convert_main(rs, is_target_of):
    """EpisodeTracker 프레임 결과 -> 새 스키마 객체 목록 (마스크는 러너가 별도 부여).

    - category: is_target -> "target", 아니면 "reference" (spec.py 의 기존 참조물)
    - src: det/redet -> "det" (둘 다 이 프레임의 GroundingDINO 검출), prop -> "track"
    - box3d: 승인(accepted)된 방식 A 값. 기각이면 None + reason (스키마 요구)
    - 후보 자체가 없던 레코드(src="none", box2d 없음)는 목록에 넣지 않는다
    """
    objs = []
    for r in rs:
        if r.get("box2d") is None or "box2d" not in r:
            continue
        acc = bool(r.get("accepted"))
        box3d = (dict(center=[float(v) for v in r["center"]],
                      size=[float(v) for v in r["size"]]) if acc else None)
        objs.append(dict(label=r["label"],
                         category="target" if is_target_of.get(r["label"]) else "reference",
                         side="none",
                         src="det" if r.get("src") in ("det", "redet") else "track",
                         box2d=[float(v) for v in r["box2d"]],
                         score=float(r.get("det_score") or 0.0),
                         mask=None, box3d=box3d,
                         reason="" if acc else str(r.get("reason", ""))))
    return objs


def run_unit(outdir, seq, kind, task, det, seg, dep):
    """한 (task x ep x cam) 유닛. frames.json(새 스키마) + ALLOBJ.mp4 + stats.json."""
    os.makedirs(outdir, exist_ok=True)
    is_he = kind == "he"
    fps = 30.0 / STRIDE
    prof = profile_for(task)
    sp = BRAINCO.get(task) or HE_REP[task]
    trk = EpisodeTracker(sp["targets"], det, seg, sp["prompt"], fps=fps, profile=prof)
    ex_prompt, ex_entries = extra_spec(task, MODE)
    ext = ExtraTracker(ex_entries, fps=fps)
    is_target_of = {}
    for v in sp["targets"].values():
        lb, tgt = (v[0], bool(v[1])) if isinstance(v, (list, tuple)) else (v, False)
        is_target_of[lb] = is_target_of.get(lb, False) or tgt

    H, W = seq[0][0].shape[:2]
    vw = cv2.VideoWriter(f"{outdir}/ALLOBJ.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (W, H))
    frames_out, raw_out, snap = [], [], None
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

        # ① 조작 대상·기존 참조물 — 검증된 트래커 그대로
        rs = trk.step(img, depth, K, dscale, align_dx=adx)
        main_objs = convert_main(rs, is_target_of)

        # ② 2차 검출 패스 — 로봇 부위(+참조·대형 배경)
        b2, p2, s2 = det(img, ex_prompt)
        cands = ext.propose(b2, p2, s2, (W, H))

        # ③ SAM 일괄 호출: 대상 박스(마스크 회수용) + 2차 후보 박스
        all_boxes = [o["box2d"] for o in main_objs] + [c["box2d"] for c in cands]
        if all_boxes:
            masks = seg(img, np.array(all_boxes, dtype=np.float32))
        else:
            masks = np.empty((0, H, W), dtype=bool)
        for o, m in zip(main_objs, masks[:len(main_objs)]):
            o["mask"] = np.asarray(m, dtype=bool)
        ext_recs = ext.finalize(cands, masks[len(main_objs):], depth, K,
                                dscale, adx, (W, H))

        objects = main_objs + ext_recs
        vis = draw_objects(img.copy(), objects, K)
        vw.write(vis)
        if i == len(seq) // 2:
            snap = vis.copy()

        out_objs = []
        for o in objects:
            m = o.pop("mask", None)
            rle = (mask_to_rle(m) if (m is not None and m.any())
                   else mask_to_rle(np.zeros((H, W), dtype=bool)))
            rec = dict(label=o["label"], category=o["category"], side=o["side"],
                       src=o["src"], box2d=[round(float(v), 1) for v in o["box2d"]],
                       mask_rle=rle, box3d=o["box3d"],
                       score=round(float(o["score"]), 3), reason=o.get("reason", ""))
            out_objs.append(rec)
        frames_out.append(dict(frame=i, objects=out_objs))
        raw_out.append(dict(f=i, r=[{k: v for k, v in x.items() if k != "_b"} for x in rs]))
    vw.release()
    if snap is not None:
        cv2.imwrite(f"{outdir}/SNAPSHOT.png", snap)

    # ---- 통계: (category, label, side)별 프레임 수
    counts = {}
    for fr in frames_out:
        for o in fr["objects"]:
            key = f"{o['category']}/{o['label']}" + \
                  (f"/{o['side']}" if o["side"] != "none" else "")
            c = counts.setdefault(key, dict(frames=0, det=0, track=0, box3d=0))
            c["frames"] += 1
            c[o["src"]] += 1
            if o["box3d"] is not None:
                c["box3d"] += 1
    st = trk.stats(len(seq))
    json.dump(frames_out, open(f"{outdir}/frames.json", "w"),
              ensure_ascii=False, default=str)
    json.dump(raw_out, open(f"{outdir}/frames_raw.json", "w"),
              ensure_ascii=False, default=str)
    json.dump(dict(task=task, mode=MODE, split=SPLIT, n_frames=len(seq), fps=fps,
                   stride=STRIDE, prompt_main=trk.prompt, prompt_extra=ex_prompt,
                   profile=describe(task), objects=counts, tracker_stats=st),
              open(f"{outdir}/stats.json", "w"), ensure_ascii=False, indent=1, default=str)
    return counts


def main():
    jobs = build_jobs(SPLIT)
    if SMOKE:
        jobs = jobs[:2]
        log(f"SMOKE 모드 — 유닛 2개 x {max(BC_CAP, HE_CAP)}프레임 상한")
    log(f"SPLIT={SPLIT} MODE={MODE} RND={RND} — 작업 {len(jobs)}건 "
        f"(seed={SEED}, {N_EP}ep/task, 손목 {WRIST_EP}ep)")

    base = f"{OUTROOT}/{RND}/{MODE}"
    os.makedirs(base, exist_ok=True)
    json.dump(dict(split=SPLIT, mode=MODE, seed=SEED, n_ep=N_EP, wrist_ep=WRIST_EP,
                   stride=STRIDE, bc_cap=BC_CAP, he_cap=HE_CAP, smoke=SMOKE,
                   jobs=[dict(kind=k, task=t, ep=e, cam=c) for k, t, e, c in jobs]),
              open(f"{base}/manifest_{SPLIT}.json", "w"), ensure_ascii=False, indent=1)

    from models_wrap import Detector, Segmenter, DepthEstimator
    t0 = time.time()
    det, seg = Detector(DEV), Segmenter(DEV)
    dep = DepthEstimator(DEV) if any(k == "bc" for k, _, _, _ in jobs) else None
    log(f"모델 로드 {time.time()-t0:.0f}s")
    torch.cuda.reset_peak_memory_stats()

    t_all = time.time()
    ok = fail = skip = nfr = 0
    for n, (kind, task, ep, cam) in enumerate(jobs, 1):
        sub = "brainco" if kind == "bc" else "he"
        outdir = f"{base}/{sub}/{task}/ep{ep:05d}_{cam}"
        if os.path.exists(f"{outdir}/stats.json"):
            skip += 1
            continue
        tmp = None
        try:
            t = time.time()
            if kind == "bc":
                files, tmp = RV.frames_brainco(task, ep, cam, stride=STRIDE, cap_n=BC_CAP)
                if not files:
                    log(f"  프레임 없음 {task}/ep{ep}/{cam}")
                    continue
                seq = [(cv2.imread(f), None) for f in files]
            else:
                seq = RV.frames_he(ep, stride=STRIDE, cap_n=HE_CAP)
                if not seq:
                    log(f"  프레임 없음 {task}/ep{ep}")
                    continue
            counts = run_unit(outdir, seq, kind, task, det, seg, dep)
            ok += 1
            nfr += len(seq)
            el = time.time() - t_all
            eta = (el / max(ok, 1)) * (len(jobs) - skip - ok) / 3600
            brief = {k: f"3D {v['box3d']}/{v['frames']}f" for k, v in counts.items()}
            log(f"[{n}/{len(jobs)}] {task}/ep{ep}/{cam} {len(seq)}f "
                f"{time.time()-t:.0f}s {brief} | ETA {eta:.1f}h")
        except Exception as e:
            fail += 1
            log(f"  실패 {task}/ep{ep}/{cam}: {str(e)[:150]}")
            with open(f"{LOGD}/fail_s10_{SPLIT}.log", "a") as f:
                f.write(f"{task}/ep{ep}/{cam}\n{traceback.format_exc()}\n")
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
    el = time.time() - t_all
    log(f"완료 — 성공 {ok} 건너뜀 {skip} 실패 {fail}, {nfr}프레임 {el/3600:.2f}h "
        f"({el/max(nfr,1):.2f}s/f), peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB")


if __name__ == "__main__":
    main()
