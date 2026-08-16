"""CPU 전용 정지컷 대량 추출 — GPU 없이 만들 수 있는 후보 전부.

  gd/    GroundingDINO 베스트·워스트 후보 (BOX2D.mp4에서 — 2D box만 그려진 영상)
  bp/    Back-projection 베스트·워스트 후보 (A_visible.mp4에서 — 3D box 영상)
  ghost/ 유령 개선 전후 컷 (r6 vs r6post, 같은 프레임 번호)
  g1/    G1 결과 컷 (H1 비교용 — AB_compare + A 중간 프레임)

채택은 로컬에서 전 장 육안 검증 후 한다.
"""
import os, json, glob, shutil, statistics as st
import cv2

ROOT = os.path.expanduser("~/task3")
R6 = f"{ROOT}/review/r6"
R6P = f"{ROOT}/review/r6post"
OUT = f"{ROOT}/probe6cpu"


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


def det_frames(d, label):
    return [(i, x) for i, r in enumerate(d) for x in r["r"]
            if x["label"] == label and x.get("accepted")
            and x.get("src") in ("det", "redet")]


def rec_at(unit, label, fi):
    d = load_frames(unit)
    for x in d[fi]["r"]:
        if x["label"] == label:
            return x
    return None


def grab(video, idx):
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, n - 1))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def median_det_frame(unit):
    lb = target_label(unit)
    f = det_frames(load_frames(unit), lb)
    if not f:
        return None
    med = st.median([x.get("mask_px", 0) for _, x in f])
    f.sort(key=lambda t: abs(t[1].get("mask_px", 0) - med))
    return f[0][0]


def min_size_det_frame(unit):
    lb = target_label(unit)
    f = [(i, x) for i, x in det_frames(load_frames(unit), lb) if x.get("size")]
    if not f:
        return None
    f.sort(key=lambda t: max(t[1]["size"]))
    return f[0][0]


def no_box_gap_frame(unit):
    """실검출 사이, 어떤 승인 기록도 없는(=박스가 안 그려진) 최장 공백의 중앙 프레임."""
    lb = target_label(unit)
    d = load_frames(unit)
    acc = set(i for i, r in enumerate(d) for x in r["r"]
              if x["label"] == lb and x.get("accepted"))
    dets = [i for i, _ in det_frames(d, lb)]
    if not dets:
        return None
    best, run = [], []
    for i in range(dets[0], dets[-1] + 1):
        if i not in acc:
            run.append(i)
        else:
            if len(run) > len(best):
                best = run
            run = []
    if len(run) > len(best):
        best = run
    return best[len(best) // 2] if best else None


def ghost_frame(unit, min_px=1500):
    lb = target_label(unit)
    d = load_frames(unit)
    dets = det_frames(d, lb)
    le = dets[-1][0] if dets else -1
    for i, r in enumerate(d):
        for x in r["r"]:
            if x["label"] == lb and i > le + 3 and x.get("accepted") \
                    and x.get("src") == "prop" and x.get("mask_px", 0) >= min_px:
                return i
    return None


def main():
    for sub in ("gd", "bp", "ghost", "g1"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    meta = {"gd": {}, "bp": {}, "ghost": {}, "g1": {}}

    # ---------------- GroundingDINO (BOX2D.mp4) ----------------
    GD = [
        ("gd_best_doll_head",   "brainco/PickDoll_ep5/cam_right_high",       102),
        ("gd_best_cube_head",   "brainco/GraspRubiksCube_ep5/cam_right_high", 198),
        ("gd_best_drink_head",  "brainco/PickDrink_ep5/cam_left_high",        "median"),
        ("gd_best_oreo_head",   "brainco/GraspOreo_ep5/cam_left_high",        "median"),
        ("gd_best_toy",         "he/Basic_ep3800",                             44),
        ("gd_best_laptop",      "he/Articulated_ep280",                         7),
        ("gd_worst_charger_gap",  "brainco/PickCharger_ep5/cam_left_high",   "gap"),
        ("gd_worst_charger_gap2", "brainco/PickCharger_ep5/cam_right_high",  "gap"),
        ("gd_worst_tpaste_gap",   "brainco/PickToothpaste_ep5/cam_left_high", "gap"),
        ("gd_worst_tissue_ghosthand", "brainco/PickTissues_ep5/cam_left_wrist", "ghost"),
        ("gd_worst_apple_hand",   "brainco/PickApple_ep5/cam_left_wrist",     156),
        ("gd_worst_rose_vase",    "he/Precision_ep7918",                      156),
    ]
    for key, unit, fi in GD:
        try:
            if fi == "median":
                fi = median_det_frame(unit)
            elif fi == "gap":
                fi = no_box_gap_frame(unit)
            elif fi == "ghost":
                fi = ghost_frame(unit)
            if fi is None:
                log(f"  !! {key}: 프레임 선정 실패")
                continue
            img = grab(f"{R6}/{unit}/BOX2D.mp4", fi)
            if img is None:
                log(f"  !! {key}: 프레임 읽기 실패")
                continue
            cv2.imwrite(f"{OUT}/gd/{key}_f{fi}.png", img)
            lb = target_label(unit)
            r = rec_at(unit, lb, fi)
            meta["gd"][key] = dict(unit=unit, frame=fi, label=lb,
                                   rec=None if not r else dict(
                                       src=r.get("src"), accepted=r.get("accepted"),
                                       det_score=round(float(r.get("det_score", 0)), 3),
                                       size_cm=[round(v * 100, 1) for v in r.get("size", [])]))
            log(f"  gd {key} f{fi}")
        except Exception as e:
            log(f"  !! {key}: {e}")

    # ---------------- Back-projection (A_visible.mp4) ----------------
    BP = [
        ("bp_best_doll_head",   "brainco/PickDoll_ep5/cam_right_high",        102),
        ("bp_best_cube_head",   "brainco/GraspRubiksCube_ep5/cam_right_high", 198),
        ("bp_best_apple_head",  "brainco/PickApple_ep5/cam_right_high",        26),
        ("bp_best_oreo_head",   "brainco/GraspOreo_ep5/cam_left_high",        "median"),
        ("bp_best_drink_head",  "brainco/PickDrink_ep5/cam_right_high",       "median"),
        ("bp_best_bottle_he",   "he/Locomanip_ep7120",                          81),
        ("bp_best_toy_he",      "he/Basic_ep3800",                              44),
        ("bp_best_laptop_he",   "he/Articulated_ep280",                          7),
        ("bp_worst_cube_wrist",  "brainco/GraspRubiksCube_ep5/cam_right_wrist", 187),
        ("bp_worst_apple_wrist", "brainco/PickApple_ep5/cam_left_wrist",         91),
        ("bp_worst_drink_wrist", "brainco/PickDrink_ep5/cam_right_wrist",       161),
        ("bp_worst_doll_wrist",  "brainco/PickDoll_ep5/cam_right_wrist",        131),
        ("bp_worst_tissue_wrist", "brainco/PickTissues_ep5/cam_right_wrist", "minsize"),
        ("bp_worst_towel_he",    "he/deformable_ep4838",                         83),
        ("bp_worst_rose_he",     "he/Precision_ep7918",                         156),
    ]
    for key, unit, fi in BP:
        try:
            if fi == "median":
                fi = median_det_frame(unit)
            elif fi == "minsize":
                fi = min_size_det_frame(unit)
            if fi is None:
                log(f"  !! {key}: 프레임 선정 실패")
                continue
            img = grab(f"{R6}/{unit}/A_visible.mp4", fi)
            if img is None:
                log(f"  !! {key}: 프레임 읽기 실패")
                continue
            cv2.imwrite(f"{OUT}/bp/{key}_f{fi}.png", img)
            lb = target_label(unit)
            r = rec_at(unit, lb, fi)
            meta["bp"][key] = dict(unit=unit, frame=fi, label=lb,
                                   rec=None if not r else dict(
                                       src=r.get("src"),
                                       size_cm=[round(v * 100, 1) for v in r.get("size", [])],
                                       n_points=r.get("n_points")))
            log(f"  bp {key} f{fi}")
        except Exception as e:
            log(f"  !! {key}: {e}")

    # ---------------- 유령 전후 (r6 vs r6post) ----------------
    units = sorted(glob.glob(f"{R6P}/brainco/*/cam_*") + glob.glob(f"{R6P}/he/*_ep*"))
    for up in units:
        rel = up.replace(R6P + "/", "")
        u6 = f"{R6}/{rel}"
        name = rel.replace("/", "_")
        try:
            tr = json.load(open(f"{up}/trim_report.json"))
        except Exception:
            continue
        for lb, v in tr.items():
            if v.get("trimmed_A", 0) < 3:
                continue
            fe = v["last_evidence"]
            slug = lb.replace(" ", "")
            for off in (3, 10):
                b = grab(f"{u6}/A_visible.mp4", fe + off)
                a = grab(f"{up}/A_visible.mp4", fe + off)
                if b is None or a is None:
                    continue
                cv2.imwrite(f"{OUT}/ghost/{name}_{slug}_f{fe+off}_before.png", b)
                cv2.imwrite(f"{OUT}/ghost/{name}_{slug}_f{fe+off}_after.png", a)
            bB = grab(f"{u6}/B_amodal.mp4", fe + 13)
            aB = grab(f"{up}/B_amodal.mp4", fe + 13)
            if bB is not None and aB is not None:
                cv2.imwrite(f"{OUT}/ghost/{name}_{slug}_f{fe+13}_B_before.png", bB)
                cv2.imwrite(f"{OUT}/ghost/{name}_{slug}_f{fe+13}_B_after.png", aB)
            meta["ghost"][f"{name}_{slug}"] = dict(last_evidence=fe,
                                                   trimmed_A=v["trimmed_A"],
                                                   trimmed_B=v.get("trimmed_B", 0))
            log(f"  ghost {name}/{lb} le={fe} trimA={v['trimmed_A']}")

    # ---------------- G1 비교 컷 ----------------
    for ud in sorted(glob.glob(f"{R6}/he/*_ep*")):
        cat = os.path.basename(ud)
        img = grab(f"{ud}/A_visible.mp4", 110)
        if img is not None:
            cv2.imwrite(f"{OUT}/g1/g1_{cat}_A_f110.png", img)
        if os.path.exists(f"{ud}/AB_compare.png"):
            shutil.copy(f"{ud}/AB_compare.png", f"{OUT}/g1/g1_{cat}_ABcompare.png")
        meta["g1"][cat] = target_label(f"he/{cat}")
        log(f"  g1 {cat}")

    json.dump(meta, open(f"{OUT}/meta.json", "w"), ensure_ascii=False, indent=1)
    n = sum(len(glob.glob(f"{OUT}/{s}/*.png")) for s in ("gd", "bp", "ghost", "g1"))
    log(f"완료 -> {OUT} (총 {n}장)")


if __name__ == "__main__":
    main()
