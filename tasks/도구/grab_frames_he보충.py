"""HE 쪽 GD/BP 후보 보충 추출 (CPU) — 데이터셋별 3~5장을 채우기 위한 2차."""
import os, json, glob, statistics as st
import cv2

ROOT = os.path.expanduser("~/task3")
R6 = f"{ROOT}/review/r6"
OUT = f"{ROOT}/probe6cpu"

def load_frames(u): return json.load(open(f"{R6}/{u}/frames.json"))
def target_label(u):
    s = json.load(open(f"{R6}/{u}/stats.json"))
    for lb, v in s["stats"].items():
        if v.get("is_target"): return lb
def det_frames(d, lb):
    return [(i, x) for i, r in enumerate(d) for x in r["r"]
            if x["label"] == lb and x.get("accepted") and x.get("src") in ("det", "redet")]
def rec_at(u, lb, fi):
    for x in load_frames(u)[fi]["r"]:
        if x["label"] == lb: return x
def grab(video, idx):
    cap = cv2.VideoCapture(video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0: cap.release(); return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, n - 1)); ok, img = cap.read(); cap.release()
    return img if ok else None
def median_det_frame(u):
    lb = target_label(u); f = det_frames(load_frames(u), lb)
    if not f: return None
    med = st.median([x.get("mask_px", 0) for _, x in f])
    f.sort(key=lambda t: abs(t[1].get("mask_px", 0) - med)); return f[0][0]
def no_box_gap_frame(u):
    lb = target_label(u); d = load_frames(u)
    acc = set(i for i, r in enumerate(d) for x in r["r"] if x["label"] == lb and x.get("accepted"))
    dets = [i for i, _ in det_frames(d, lb)]
    if not dets: return None
    best, run = [], []
    for i in range(dets[0], dets[-1] + 1):
        if i not in acc: run.append(i)
        else:
            if len(run) > len(best): best = run
            run = []
    if len(run) > len(best): best = run
    return best[len(best) // 2] if len(best) >= 5 else None
def max_size_det_frame(u):
    lb = target_label(u); f = [(i, x) for i, x in det_frames(load_frames(u), lb) if x.get("size")]
    if not f: return None
    f.sort(key=lambda t: -max(t[1]["size"])); return f[0][0]

meta = {}
JOBS = [
    # GroundingDINO 보충 (HE)
    ("gd", "gd_best_bottle_he",  "he/Locomanip_ep7120",  "median", "BOX2D.mp4"),
    ("gd", "gd_best_towel_he",   "he/deformable_ep4838", "median", "BOX2D.mp4"),
    ("gd", "gd_best_duster_he",  "he/Tool_use_ep8198",   "median", "BOX2D.mp4"),
    ("gd", "gd_best_rose_he",    "he/HRI_ep5598",        "median", "BOX2D.mp4"),
    ("gd", "gd_worst_basic_gap", "he/Basic_ep3800",      "gap",    "BOX2D.mp4"),
    ("gd", "gd_worst_towel_gap", "he/deformable_ep4838", "gap",    "BOX2D.mp4"),
    ("gd", "gd_worst_duster_gap","he/Tool_use_ep8198",   "gap",    "BOX2D.mp4"),
    ("gd", "gd_worst_loco_gap",  "he/Locomanip_ep7120",  "gap",    "BOX2D.mp4"),
    # Back-projection 보충 (HE + brainco 워스트 1)
    ("bp", "bp_best_rose_he",    "he/HRI_ep5598",        "median", "A_visible.mp4"),
    ("bp", "bp_best_duster_he",  "he/Tool_use_ep8198",   "median", "A_visible.mp4"),
    ("bp", "bp_worst_towel_max", "he/deformable_ep4838",  82,      "A_visible.mp4"),
    ("bp", "bp_worst_duster_he", "he/Tool_use_ep8198",   "maxsize","A_visible.mp4"),
    ("bp", "bp_worst_rose_hri",  "he/HRI_ep5598",        "maxsize","A_visible.mp4"),
    ("bp", "bp_worst_oreo_wrist","brainco/GraspOreo_ep5/cam_right_wrist", "maxsize", "A_visible.mp4"),
]
for sub, key, unit, fi, vid in JOBS:
    try:
        if fi == "median": fi = median_det_frame(unit)
        elif fi == "gap": fi = no_box_gap_frame(unit)
        elif fi == "maxsize": fi = max_size_det_frame(unit)
        if fi is None:
            print(f"  !! {key}: 프레임 선정 실패(공백 없음 등)"); continue
        img = grab(f"{R6}/{unit}/{vid}", fi)
        if img is None:
            print(f"  !! {key}: 읽기 실패"); continue
        cv2.imwrite(f"{OUT}/{sub}/{key}_f{fi}.png", img)
        lb = target_label(unit); r = rec_at(unit, lb, fi)
        meta[key] = dict(unit=unit, frame=fi, label=lb,
                         rec=None if not r else dict(src=r.get("src"),
                             size_cm=[round(v*100,1) for v in r.get("size",[])],
                             det_score=round(float(r.get("det_score",0)),3)))
        print(f"  {sub} {key} f{fi}")
    except Exception as e:
        print(f"  !! {key}: {e}")
json.dump(meta, open(f"{OUT}/meta2.json", "w"), ensure_ascii=False, indent=1)
print("완료")
