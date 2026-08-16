"""SAM 픽셀마스크 / UniDepth 깊이 — 베스트·워스트 '후보' 대량 렌더 (GPU).

지난 실패: 수치로 뽑은 사례를 육안 검증 없이 폴더에 넣어 라벨과 안 맞는 샘플이 섞였다.
이번 절차: 여기서는 후보를 '넓게' 렌더만 하고, 채택은 로컬에서 전 장 육안 검증 후 한다.

출력: ~/task3/probe6/sam/*.png + meta.json
      ~/task3/probe6/depth/*.png + meta.json
"""
import os, sys, json, shutil, statistics as st
import numpy as np
import cv2

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
from geometry import Intrinsics
import run_review as RV

R6 = f"{ROOT}/review/r6"
OUT = f"{ROOT}/probe6"
DEV = "cuda:0"


def log(m):
    print(m, flush=True)


def load_frames(unit):
    return json.load(open(f"{R6}/{unit}/frames.json"))


def target_label(unit):
    s = json.load(open(f"{R6}/{unit}/stats.json"))
    for lb, v in s["stats"].items():
        if v.get("is_target"):
            return lb
    return None


def recs_of(d, label):
    return [(i, x) for i, r in enumerate(d) for x in r["r"] if x["label"] == label]


def det_frames(d, label):
    return [(i, x) for i, x in recs_of(d, label)
            if x.get("accepted") and x.get("src") in ("det", "redet")]


def rec_at(unit, label, fi):
    d = load_frames(unit)
    for x in d[fi]["r"]:
        if x["label"] == label:
            return x
    return None


def get_seq(unit):
    parts = unit.rstrip("/").split("/")
    if unit.startswith("brainco/"):
        task, ep = parts[1].split("_ep")
        files, tmp = RV.frames_brainco(task, int(ep), parts[2])
        return [(cv2.imread(f), None) for f in files], tmp
    ep = int(parts[1].split("_ep")[1])
    return RV.frames_he(ep), None


def draw_mask_panel(img, mask, box, label):
    vis = img.copy()
    ov = vis.copy()
    ov[mask] = (0, 235, 255)
    vis = cv2.addWeighted(ov, 0.45, vis, 0.55, 0)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 140, 255), 2)
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
    px = int(mask.sum())
    ar = px / (img.shape[0] * img.shape[1]) * 100
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vis, f"{label}  mask {px:,}px ({ar:.1f}% of frame)  components={len(cnts)}",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis, px, ar, len(cnts)


# ---------------------------------------------------------------- SAM 후보
# 워스트 후보 — 수치 근거가 있는 실패 의심 프레임 (채택은 육안으로)
SAM_WORST = [
    ("bw_apple_hand",   "brainco/PickApple_ep5/cam_left_wrist",       156),
    ("bw_oreo_hand",    "brainco/GraspOreo_ep5/cam_right_wrist",      178),
    ("bw_tpaste_hand",  "brainco/PickToothpaste_ep5/cam_right_wrist", 107),
    ("bw_doll_frag",    "brainco/PickDoll_ep5/cam_right_wrist",       115),
    ("bw_doll_prop",    "brainco/PickDoll_ep5/cam_left_wrist",        163),
    ("bw_drink_45cm",   "brainco/PickDrink_ep5/cam_right_wrist",      161),
    ("hw_towel_frag",   "he/deformable_ep4838",                        92),
    ("hw_towel_frag2",  "he/deformable_ep4838",                        89),
    ("hw_duster_speck", "he/Tool_use_ep8198",                          32),
    ("hw_rose_vase",    "he/Precision_ep7918",                        156),
    ("hw_laptop_over",  "he/Articulated_ep280",                        36),
    ("hw_toy_part",     "he/Basic_ep3800",                             88),
]
# 베스트 후보 — det 프레임 중 fill 상위 (동적 선정, 유닛당 2)
SAM_BEST_UNITS = [
    ("bb_cube",   "brainco/GraspRubiksCube_ep5/cam_right_wrist"),
    ("bb_doll",   "brainco/PickDoll_ep5/cam_right_wrist"),
    ("bb_drink",  "brainco/PickDrink_ep5/cam_left_wrist"),
    ("bb_oreo",   "brainco/GraspOreo_ep5/cam_right_wrist"),
    ("bb_apple",  "brainco/PickApple_ep5/cam_left_wrist"),
    ("bb_tpaste", "brainco/PickToothpaste_ep5/cam_right_wrist"),
    ("hb_towel",  "he/deformable_ep4838"),
    ("hb_laptop", "he/Articulated_ep280"),
    ("hb_toy",    "he/Basic_ep3800"),
    ("hb_bottle", "he/Locomanip_ep7120"),
    ("hb_rose",   "he/Precision_ep7918"),
]


def ghost_frame(unit, min_px=2000):
    """마지막 실검출 이후 전파(prop)로 큰 마스크가 남은 첫 프레임 (유령 컷)."""
    lb = target_label(unit)
    d = load_frames(unit)
    dets = det_frames(d, lb)
    le = dets[-1][0] if dets else -1
    for i, x in recs_of(d, lb):
        if i > le + 3 and x.get("accepted") and x.get("src") == "prop" \
                and x.get("mask_px", 0) >= min_px:
            return i
    return None


def best_frame_ids(unit, n=2):
    lb = target_label(unit)
    d = load_frames(unit)
    f = [(i, x) for i, x in det_frames(d, lb) if x.get("fill") and x.get("mask_px")]
    if not f:
        return []
    med = st.median([x["mask_px"] for _, x in f])
    ok = [(i, x) for i, x in f if 0.4 * med <= x["mask_px"] <= 2.5 * med]
    ok.sort(key=lambda t: -t[1]["fill"])
    out = []
    for i, x in ok:
        if any(abs(i - j) < 15 for j in out):
            continue
        out.append(i)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- 깊이 후보
DEPTH_UNITS = [
    ("dp_oreo_head",  "brainco/GraspOreo_ep5/cam_left_high"),
    ("dp_apple_head", "brainco/PickApple_ep5/cam_right_high"),
    ("dp_doll_head",  "brainco/PickDoll_ep5/cam_right_high"),
    ("dp_cube_wrist", "brainco/GraspRubiksCube_ep5/cam_right_wrist"),
]


def depth_pairs(unit):
    """연속(det) 프레임 쌍 중 물체 중앙깊이 변화가 최소/최대인 쌍."""
    lb = target_label(unit)
    d = load_frames(unit)
    f = det_frames(d, lb)
    pairs = []
    for (i, xi), (j, xj) in zip(f, f[1:]):
        if j == i + 1 and xi.get("dmed") and xj.get("dmed"):
            r = abs(xj["dmed"] - xi["dmed"]) / max(xi["dmed"], 1e-6)
            pairs.append((r, i, j, xi["dmed"], xj["dmed"]))
    if not pairs:
        return None
    pairs.sort()
    return pairs[0], pairs[-1]


def depth_cm(depth, lo=0.2, hi=3.0):
    dd = np.clip(depth, lo, hi)
    u = ((dd - lo) / (hi - lo) * 255).astype(np.uint8)
    return cv2.applyColorMap(u, cv2.COLORMAP_TURBO)


def depth_row(img, depth, box, txt):
    a, c = img.copy(), depth_cm(depth)
    if box:
        x1, y1, x2, y2 = [int(v) for v in box]
        for v in (a, c):
            cv2.rectangle(v, (x1, y1), (x2, y2), (255, 255, 255), 2)
    row = np.hstack([a, c])
    cv2.rectangle(row, (0, 0), (row.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(row, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return row


# ---------------------------------------------------------------- 메인
def main():
    from models_wrap import Segmenter, DepthEstimator
    os.makedirs(f"{OUT}/sam", exist_ok=True)
    os.makedirs(f"{OUT}/depth", exist_ok=True)
    seg = Segmenter(DEV)
    dep = DepthEstimator(DEV)
    log("모델 로드 완료")

    worst = list(SAM_WORST)
    gi = ghost_frame("brainco/PickTissues_ep5/cam_left_wrist")
    if gi is not None:
        worst.append(("bw_tissue_ghost", "brainco/PickTissues_ep5/cam_left_wrist", gi))

    tasks = {}          # unit -> [(key, fi, bucket)]
    for key, unit, fi in worst:
        tasks.setdefault(unit, []).append((key, fi, "worst"))
    for key, unit in SAM_BEST_UNITS:
        for rank, fi in enumerate(best_frame_ids(unit)):
            tasks.setdefault(unit, []).append((f"{key}{rank+1}", fi, "best"))

    depth_by_unit = {u: k for k, u in DEPTH_UNITS}

    meta_sam, meta_dep = {}, {}
    for unit, items in tasks.items():
        lb = target_label(unit)
        log(f"--- {unit} / {lb} ({len(items)}건)")
        try:
            seq, tmp = get_seq(unit)
        except Exception as e:
            log(f"  !! 시퀀스 로드 실패: {e}")
            continue
        for key, fi, bucket in items:
            try:
                if fi >= len(seq):
                    log(f"  !! {key}: 프레임 {fi} 없음")
                    continue
                r = rec_at(unit, lb, fi)
                if not r or "box2d" not in r:
                    log(f"  !! {key}: box2d 기록 없음")
                    continue
                img = seq[fi][0]
                m = seg(img, np.array([r["box2d"]], dtype=np.float32))
                if len(m) == 0:
                    log(f"  !! {key}: 마스크 없음")
                    continue
                vis, px, ar, nc = draw_mask_panel(img, m[0], r["box2d"], lb)
                cv2.imwrite(f"{OUT}/sam/{key}_f{fi}.png", vis)
                meta_sam[key] = dict(unit=unit, label=lb, frame=fi, bucket=bucket,
                                     mask_px=px, area_pct=round(ar, 1), components=nc,
                                     src=r.get("src"), fill=r.get("fill"),
                                     size_cm=[round(v * 100, 1) for v in r.get("size", [])],
                                     det_score=round(float(r.get("det_score", 0)), 3))
                log(f"  {key} f{fi}: {px}px {ar:.1f}% comp={nc} src={r.get('src')}")
            except Exception as e:
                log(f"  !! {key}: {e}")

        # 깊이 후보 (이 유닛이 대상이면 같은 seq 재사용)
        if unit in depth_by_unit:
            dkey = depth_by_unit[unit]
            try:
                pr = depth_pairs(unit)
                if pr:
                    for tag, (ratio, i, j, di, dj) in (("stable", pr[0]), ("jump", pr[1])):
                        rows = []
                        for fi2, dm in ((i, di), (j, dj)):
                            img = seq[fi2][0]
                            dpt, _ = dep(img)
                            r2 = rec_at(unit, lb, fi2)
                            rows.append(depth_row(img, dpt, r2.get("box2d") if r2 else None,
                                                  f"frame {fi2}   {lb} median depth {dm:.2f} m"))
                        out = np.vstack(rows)
                        cv2.imwrite(f"{OUT}/depth/{dkey}_{tag}_f{i}-{j}.png", out)
                        meta_dep[f"{dkey}_{tag}"] = dict(unit=unit, frames=[i, j],
                                                         dmed=[round(di, 3), round(dj, 3)],
                                                         change_pct=round(ratio * 100, 1))
                        log(f"  depth {dkey} {tag}: f{i}->{j} {di:.2f}->{dj:.2f}m ({ratio*100:.1f}%)")
            except Exception as e:
                log(f"  !! depth {dkey}: {e}")
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    # 깊이 전용 유닛 (SAM 작업이 없어 위 루프에서 안 돈 것)
    for dkey, unit in DEPTH_UNITS:
        if any(f"{dkey}_" in k for k in meta_dep):
            continue
        lb = target_label(unit)
        try:
            seq, tmp = get_seq(unit)
            pr = depth_pairs(unit)
            if pr:
                for tag, (ratio, i, j, di, dj) in (("stable", pr[0]), ("jump", pr[1])):
                    rows = []
                    for fi2, dm in ((i, di), (j, dj)):
                        img = seq[fi2][0]
                        dpt, _ = dep(img)
                        r2 = rec_at(unit, lb, fi2)
                        rows.append(depth_row(img, dpt, r2.get("box2d") if r2 else None,
                                              f"frame {fi2}   {lb} median depth {dm:.2f} m"))
                    cv2.imwrite(f"{OUT}/depth/{dkey}_{tag}_f{i}-{j}.png", np.vstack(rows))
                    meta_dep[f"{dkey}_{tag}"] = dict(unit=unit, frames=[i, j],
                                                     dmed=[round(di, 3), round(dj, 3)],
                                                     change_pct=round(ratio * 100, 1))
                    log(f"  depth {dkey} {tag}: f{i}->{j} {di:.2f}->{dj:.2f}m ({ratio*100:.1f}%)")
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
        except Exception as e:
            log(f"  !! depth-only {dkey}: {e}")

    json.dump(meta_sam, open(f"{OUT}/sam/meta.json", "w"), ensure_ascii=False, indent=1)
    json.dump(meta_dep, open(f"{OUT}/depth/meta.json", "w"), ensure_ascii=False, indent=1)
    log(f"완료 -> {OUT} (sam {len(meta_sam)}장, depth {len(meta_dep)}장)")


if __name__ == "__main__":
    main()
