"""소형 GPU 실험 3종 묶음 — 오늘 GPU 창 안에서 끝내야 하는 검증만 담는다.

전부 **기존 검증된 스택**(GroundingDINO + SAM 2.1 + UniDepthV2)만 사용한다.
신규 모델 설치가 없어 실행 시간이 예측 가능하다.

  A. bigobj   — 큰 객체(table/desk) 검출·분할이 되는지, 어느 게이트가 막는지
  B. gripper  — 로봇 손·팔·그리퍼 검출률 정량 측정 (RobotSeg 필요성 판단 근거)
  C. h1       — HE의 H1 로봇 에피소드에 현 파이프라인이 그대로 동작하는지

사용법:
  python run_probe.py bigobj|gripper|h1|all <출력디렉토리>
"""
import os, sys, json, glob, time, shutil
import numpy as np
import cv2

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
from geometry import Intrinsics
from track3d import EpisodeTracker
from profiles import Profile
import run_review as RV

DATA = "/data2/humanoid_dataset_isangmin"
HE_ROOT = f"{DATA}/humanoid-everyday"
DEV = "cuda:0"
FPS = 6.0


def log(m):
    print(f"[{time.strftime('%T')}] {m}", flush=True)


def summarize(trk, n, extra=None):
    st = trk.stats(n)
    out = {}
    for lb, v in st.items():
        out[lb] = dict(coverage=v["coverage"], coverage_B=v.get("coverage_B"),
                       accepted=v["accepted"], size_median=v["size_median"],
                       is_target=v.get("is_target"))
    if extra:
        out["_extra"] = extra
    return out


# ---------------------------------------------------------------- A. 큰 객체
BIGOBJ_CASES = [
    ("Brainco/GraspOreo", "bc", ("GraspOreo", 5, "cam_left_high"),
     "table . desk . white table .", {"table": ("table", True), "desk": ("table", True)}),
    ("HE/Basic", "he", (3800,),
     "table . desk .", {"table": ("table", True), "desk": ("table", True)}),
]

# 큰 객체 전용 프로파일 — 조작 대상용 크기 상한을 풀어야 테이블이 통과한다.
# 이 값들은 이 실험 전용이며 기존 태스크 프로파일에는 영향이 없다.
BIG_PROFILE = Profile(max_phys_size=3.0, area_max=0.95, min_points=200)


def probe_bigobj(outdir, det, seg, dep):
    """큰 객체가 왜 안 잡히는지: 검출 자체가 없는가, 크기 게이트가 막는가."""
    res = {}
    for name, ds, args, prompt, targets in BIGOBJ_CASES:
        log(f"[bigobj] {name}")
        if ds == "bc":
            files, tmp = RV.frames_brainco(*args, stride=15, cap_n=20)
            seq = [(cv2.imread(f), None) for f in files]
        else:
            seq = RV.frames_he(args[0], stride=25, cap_n=20); tmp = None
        if not seq:
            continue
        H, W = seq[0][0].shape[:2]
        # 두 조건을 나란히: 기본 프로파일 vs 큰 객체 프로파일
        for tag, prof in [("default", Profile()), ("bigobj_profile", BIG_PROFILE)]:
            trk = EpisodeTracker(targets, det, seg, prompt, fps=FPS, profile=prof)
            reasons, raw_sizes, det_scores = {}, [], []
            for img, gt in seq:
                if ds == "he":
                    d = np.array(gt, dtype=np.float32)
                    if d.ndim == 1: d = d.reshape(d.size // W, W)
                    if d.shape != (H, W): d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
                    K = Intrinsics.from_fov(W, H, 70.0); ds_, adx = 1e-3, -20
                else:
                    d, Kp = dep(img)
                    K = (Intrinsics(float(Kp[0,0]), float(Kp[1,1]), float(Kp[0,2]), float(Kp[1,2]))
                         if Kp is not None else Intrinsics.from_fov(W, H, 70.0))
                    ds_, adx = 1.0, 0
                for r in trk.step(img, d, K, ds_, align_dx=adx):
                    if not r.get("accepted"):
                        reasons[r.get("reason", "?")] = reasons.get(r.get("reason", "?"), 0) + 1
                    else:
                        raw_sizes.append(r.get("raw_size"))
                    if r.get("det_score"):
                        det_scores.append(float(r["det_score"]))
            sz = (np.median(np.array([s for s in raw_sizes if s]), axis=0).tolist()
                  if raw_sizes else None)
            res[f"{name}/{tag}"] = dict(
                n_frames=len(seq), stats=summarize(trk, len(seq)),
                reject_reasons=reasons, size_median_raw=sz,
                det_score_median=float(np.median(det_scores)) if det_scores else None)
            log(f"   {tag}: {res[f'{name}/{tag}']['stats']}")
            log(f"   기각사유: {reasons}")
        if tmp: shutil.rmtree(tmp, ignore_errors=True)
    json.dump(res, open(f"{outdir}/bigobj.json", "w"), ensure_ascii=False, indent=1, default=str)
    return res


# ---------------------------------------------------------------- B. 그리퍼/손
GRIPPER_PROMPTS = [
    "robot hand . robot arm . gripper . human hand .",
    "robotic gripper . mechanical hand . robot manipulator .",
    "black gripper finger . white robot arm .",
]
GRIPPER_CASES = [
    ("Brainco/GraspOreo/head", "bc", ("GraspOreo", 5, "cam_left_high")),
    ("Brainco/GraspOreo/wrist", "bc", ("GraspOreo", 5, "cam_right_wrist")),
    ("HE/Basic", "he", (3800,)),
]


def probe_gripper(outdir, det, seg, dep):
    """현 스택(GroundingDINO)이 로봇 손·팔을 얼마나 잡는가 — RobotSeg 필요성의 정량 근거."""
    res = {}
    for name, ds, args in GRIPPER_CASES:
        log(f"[gripper] {name}")
        if ds == "bc":
            files, tmp = RV.frames_brainco(*args, stride=15, cap_n=20)
            seq = [cv2.imread(f) for f in files]
        else:
            seq = [im for im, _ in RV.frames_he(args[0], stride=25, cap_n=20)]; tmp = None
        if not seq:
            continue
        H, W = seq[0].shape[:2]
        for pi, prompt in enumerate(GRIPPER_PROMPTS):
            hits, scores, areas = {}, [], []
            for img in seq:
                boxes, phrases, sc = det(img, prompt)
                for b, p, s in zip(boxes, phrases, sc):
                    key = str(p)
                    hits[key] = hits.get(key, 0) + 1
                    scores.append(float(s))
                    areas.append(float((b[2]-b[0])*(b[3]-b[1])/(W*H)))
            res[f"{name}/p{pi}"] = dict(
                prompt=prompt, n_frames=len(seq),
                detections_per_frame=round(sum(hits.values())/max(len(seq),1), 2),
                phrase_counts=hits,
                score_median=float(np.median(scores)) if scores else None,
                area_ratio_median=float(np.median(areas)) if areas else None)
            log(f"   p{pi}: 프레임당 {res[f'{name}/p{pi}']['detections_per_frame']}건, {hits}")
        # 마스크 품질도 한 프레임 확인 (첫 프레임)
        boxes, phrases, sc = det(seq[0], GRIPPER_PROMPTS[0])
        if len(boxes):
            m = seg(seq[0], np.array(boxes, dtype=np.float32))
            vis = seq[0].copy()
            for i, (b, p) in enumerate(zip(boxes, phrases)):
                col = [(0,220,120),(0,165,255),(255,120,60),(200,80,220)][i % 4]
                vis[m[i]] = (0.5*vis[m[i]] + 0.5*np.array(col)).astype(np.uint8)
                cv2.rectangle(vis, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), col, 2)
                cv2.putText(vis, str(p), (int(b[0]), max(15,int(b[1])-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            cv2.imwrite(f"{outdir}/gripper_{name.replace('/','_')}.png", vis)
        if tmp: shutil.rmtree(tmp, ignore_errors=True)
    json.dump(res, open(f"{outdir}/gripper.json", "w"), ensure_ascii=False, indent=1, default=str)
    return res


# ---------------------------------------------------------------- C. H1 로봇
def pick_h1_episodes(k=3):
    """HE에서 robot_type=h1인 에피소드를 카테고리별로 고른다."""
    import collections
    tasks = {t["task_index"]: t for t in
             (json.loads(l) for l in open(f"{HE_ROOT}/meta/tasks.jsonl"))}
    eps = [json.loads(l) for l in open(f"{HE_ROOT}/meta/episodes.jsonl")]
    byti = collections.defaultdict(list)
    for e in eps:
        if e.get("robot_type") != "h1":
            continue
        ep = e["episode_index"]; ch = ep // 1000
        if os.path.exists(f"{HE_ROOT}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet"):
            byti[e["tasks"][0]].append(e)
    out = []
    for ti, lst in sorted(byti.items()):
        t = tasks.get(ti)
        if not t:
            continue
        cand = [e for e in lst if 200 <= e["length"] <= 700]
        if cand:
            out.append((t["category"], t["task"], cand[0]["episode_index"], t["description"]))
        if len(out) >= k:
            break
    return out


# H1 대표 6건 — description을 읽고 대상 물체를 직접 지정한다.
# 1차 시험에서 태스크명 자동 파싱이 동사를 프롬프트에 넣어("adjust . tilt .") 검증이 무의미했다.
# G1의 spec.py와 같은 방식(수작업 지정)으로 맞춰야 G1/H1 비교가 성립한다.
H1_CASES = [
    ("Articulated", 120, "click_a_button_on_the_mouse",
     "black computer mouse . desk .", {"mouse": ("mouse", True), "desk": ("desk", False)}),
    ("Basic", 2680, "pick_a_web_cam_and_place_it_in_a_container",
     "webcam . container box .", {"webcam": ("webcam", True), "container": ("container", False)}),
    ("Basic", 1960, "lift_a_kettle_from_base",
     "kettle . base .", {"kettle": ("kettle", True), "base": ("base", False)}),
    ("Precision", 7998, "place_a_soldering_gun_into_its_base_h1",
     "soldering gun . base .", {"gun": ("soldering gun", True), "base": ("base", False)}),
    ("Tool_use", 8398, "pick_the_hammer_and_hit_desk",
     "hammer . desk .", {"hammer": ("hammer", True), "desk": ("desk", False)}),
    ("deformable", 4718, "fold_a_jacket_on_the_desk",
     "black jacket . cloth on desk .", {"jacket": ("jacket", True), "cloth": ("jacket", True)}),
]


def probe_h1(outdir, det, seg, dep):
    """H1 에피소드에 현 파이프라인을 그대로 적용. 프롬프트는 description 기반 수작업 지정."""
    res = {}
    log(f"[h1] 대상 {len(H1_CASES)}건")
    for cat, ep, task, prompt, targets in H1_CASES:
        seq = RV.frames_he(ep, stride=20, cap_n=30)
        if not seq:
            log(f"   프레임 없음 ep{ep}"); continue
        H, W = seq[0][0].shape[:2]
        trk = EpisodeTracker(targets, det, seg, prompt, fps=FPS, profile=Profile())
        snap = snap_b = None
        for i, (img, gt) in enumerate(seq):
            d = np.array(gt, dtype=np.float32)
            if d.ndim == 1: d = d.reshape(d.size // W, W)
            if d.shape != (H, W): d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
            K = Intrinsics.from_fov(W, H, 70.0)
            rs = trk.step(img, d, K, 1e-3, align_dx=-20)
            if i == len(seq) // 2:
                snap = trk.draw(img.copy(), rs, K, mode="A")
                snap_b = trk.draw(img.copy(), rs, K, mode="B")
        if snap is not None:
            cv2.imwrite(f"{outdir}/h1_{cat}_ep{ep}.png", np.hstack([snap, snap_b]))
        st = summarize(trk, len(seq))
        # 로봇 손이 대상으로 오검출되는지: 승인 박스가 화면 하단에 몰리는지 확인
        res[f"{cat}/ep{ep}"] = dict(task=task, prompt=prompt, resolution=[W, H],
                                    n_frames=len(seq), stats=st)
        tg = {lb: (round(v["coverage"]*100), [round(x*100) for x in v["size_median"]]
                   if v["size_median"] else None)
              for lb, v in st.items() if isinstance(v, dict) and v.get("is_target")}
        log(f"   {cat}/ep{ep} '{prompt[:34]}' -> {tg}")
    json.dump(res, open(f"{outdir}/h1.json", "w"), ensure_ascii=False, indent=1, default=str)
    return res


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    outdir = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/probe"
    os.makedirs(outdir, exist_ok=True)
    from models_wrap import Detector, Segmenter, DepthEstimator
    t0 = time.time()
    det, seg, dep = Detector(DEV), Segmenter(DEV), DepthEstimator(DEV)
    log(f"모델 로드 {time.time()-t0:.0f}s")
    if what in ("bigobj", "all"):  probe_bigobj(outdir, det, seg, dep)
    if what in ("gripper", "all"): probe_gripper(outdir, det, seg, dep)
    if what in ("h1", "all"):      probe_h1(outdir, det, seg, dep)
    log("PROBE_DONE")


if __name__ == "__main__":
    main()
