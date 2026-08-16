"""객체 쌍 상관관계 산출 — CPU 후처리 (GPU 루프 밖, 산출된 json 만 읽는다).

run_sample10.py 가 저장한 frames.json(객체 리스트 스키마)에서 프레임별 객체 쌍의
  - 3D 중심 거리(m)                       — 두 box3d center 의 유클리드 거리
  - 수직 관계(위/아래)                    — 카메라 좌표 y축 기준 (y 는 아래로 +,
                                            따라서 center y 가 **작은** 쪽이 '위')
  - 접촉 추정                             — 두 3D box(AABB) 표면 간 최소 거리 < 5cm
  - 2D box IoU
를 계산해 relations.json 과 사람이 읽을 요약 텍스트(relations.txt)를 만든다.

객체 이름: side 가 있으면 "left robot gripper" 식으로 붙는다. box3d 가 null 인
객체는 2D IoU 만 계산된다(3D 항목은 null).

의존성: numpy + 표준 라이브러리뿐 — torch/cv2 불필요, GPU 없는 로컬에서도 실행·테스트 가능.

사용법:
  python relations.py <유닛폴더 또는 frames.json 경로> [...]     # 유닛별 처리
  python relations.py --tree <라운드폴더>                        # 하위 전 유닛 일괄
"""
import os, sys, json, glob
import numpy as np

CONTACT_THR = 0.05      # m — AABB 표면 간 최소 거리가 이보다 작으면 '접촉'으로 추정
LEVEL_BAND = 0.02       # m — 중심 y 차가 이 이내면 위/아래를 가르지 않는다("level")


# ------------------------------------------------------------------ 기본 계산
def obj_name(o):
    side = o.get("side") or "none"
    return o["label"] if side == "none" else f"{side} {o['label']}"


def iou2d(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def surface_gap3d(ca, sa, cb, sb):
    """두 axis-aligned 3D box 표면 간 최소 거리(m). 겹치면 0.
    축별로 gap_i = max(0, |dc_i| - (sa_i+sb_i)/2), 최소 거리 = ||gap||."""
    ca, sa = np.asarray(ca, float), np.asarray(sa, float)
    cb, sb = np.asarray(cb, float), np.asarray(sb, float)
    gaps = np.maximum(np.abs(cb - ca) - (sa + sb) / 2.0, 0.0)
    return float(np.linalg.norm(gaps))


def vertical_rel(ca, cb):
    """카메라 좌표 y축(아래로 +) 기준 위/아래. 반환 'a_above'|'b_above'|'level'."""
    dy = float(cb[1]) - float(ca[1])    # b 가 더 아래면 dy > 0 -> a 가 위
    if dy > LEVEL_BAND:
        return "a_above"
    if dy < -LEVEL_BAND:
        return "b_above"
    return "level"


# ------------------------------------------------------------------ 프레임/유닛 처리
def frame_pairs(objects):
    """한 프레임의 객체 쌍 관계 목록. 이름이 겹치면 #2 를 붙여 구분한다."""
    named, seen = [], {}
    for o in objects:
        nm = obj_name(o)
        seen[nm] = seen.get(nm, 0) + 1
        if seen[nm] > 1:
            nm = f"{nm}#{seen[nm]}"
        named.append((nm, o))
    pairs = []
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            na, a = named[i]
            nb, b = named[j]
            rec = dict(a=na, b=nb, iou2d=round(iou2d(a["box2d"], b["box2d"]), 4),
                       dist=None, gap=None, vertical=None, contact=None)
            ba, bb = a.get("box3d"), b.get("box3d")
            if ba is not None and bb is not None:
                ca, sa = ba["center"], ba["size"]
                cb, sb = bb["center"], bb["size"]
                rec["dist"] = round(float(np.linalg.norm(
                    np.asarray(ca, float) - np.asarray(cb, float))), 4)
                gap = surface_gap3d(ca, sa, cb, sb)
                rec["gap"] = round(gap, 4)
                rec["contact"] = bool(gap < CONTACT_THR)
                rec["vertical"] = vertical_rel(ca, cb)
            pairs.append(rec)
    return pairs


def relations_for_frames(frames):
    """frames.json 내용 -> relations dict (프레임별 + 쌍별 집계)."""
    out_frames, agg = [], {}
    for fr in frames:
        if "objects" not in fr:
            raise ValueError(
                "frames.json 스키마가 다르다 — run_sample10 산출(frame/objects)만 지원한다"
                " (run_review 의 f/r 스키마는 대상이 아님)")
        pairs = frame_pairs(fr["objects"])
        out_frames.append(dict(frame=fr["frame"], pairs=pairs))
        for p in pairs:
            key = f"{p['a']} | {p['b']}"
            g = agg.setdefault(key, dict(a=p["a"], b=p["b"], n_frames=0, n_3d=0,
                                         n_contact=0, dists=[], ious=[],
                                         vertical={"a_above": 0, "b_above": 0, "level": 0}))
            g["n_frames"] += 1
            g["ious"].append(p["iou2d"])
            if p["dist"] is not None:
                g["n_3d"] += 1
                g["dists"].append(p["dist"])
                g["vertical"][p["vertical"]] += 1
                if p["contact"]:
                    g["n_contact"] += 1
    summary = {}
    for key, g in agg.items():
        d = np.array(g["dists"], float) if g["dists"] else None
        vert = max(g["vertical"], key=g["vertical"].get) if g["n_3d"] else None
        summary[key] = dict(
            a=g["a"], b=g["b"], n_frames=g["n_frames"], n_3d=g["n_3d"],
            n_contact=g["n_contact"],
            dist_mean=None if d is None else round(float(d.mean()), 4),
            dist_min=None if d is None else round(float(d.min()), 4),
            iou2d_mean=round(float(np.mean(g["ious"])), 4) if g["ious"] else 0.0,
            vertical_mode=vert, vertical_counts=g["vertical"])
    return dict(contact_thr=CONTACT_THR, level_band=LEVEL_BAND,
                n_frames=len(out_frames), frames=out_frames, pairs_summary=summary)


def summarize(rel):
    """사람이 읽을 요약 텍스트. 예: 'left robot gripper–bottle 평균 0.12m, 접촉 37/180f'."""
    lines = [f"객체 쌍 관계 요약 — {rel['n_frames']}프레임, "
             f"접촉 임계 {rel['contact_thr']*100:.0f}cm"]
    items = sorted(rel["pairs_summary"].values(),
                   key=lambda g: (-g["n_3d"], g["a"], g["b"]))
    for g in items:
        if g["n_3d"] == 0:
            lines.append(f"  {g['a']}–{g['b']}: 3D 동시 산출 없음 "
                         f"(2D 동시 등장 {g['n_frames']}f, IoU 평균 {g['iou2d_mean']:.2f})")
            continue
        vert_txt = {"a_above": f"{g['a']}가 위", "b_above": f"{g['b']}가 위",
                    "level": "높이 비슷"}.get(g["vertical_mode"], "")
        contact = (f", 접촉 {g['n_contact']}/{g['n_3d']}f" if g["n_contact"] else "")
        lines.append(f"  {g['a']}–{g['b']}: 평균 {g['dist_mean']:.2f}m "
                     f"(최소 {g['dist_min']:.2f}m){contact}, {vert_txt}, "
                     f"2D IoU 평균 {g['iou2d_mean']:.2f}")
    return "\n".join(lines)


def process_unit(path):
    """유닛 폴더(또는 frames.json 경로) -> relations.json + relations.txt 저장."""
    fj = path if path.endswith(".json") else os.path.join(path, "frames.json")
    unit = os.path.dirname(os.path.abspath(fj))
    frames = json.load(open(fj))
    rel = relations_for_frames(frames)
    json.dump(rel, open(os.path.join(unit, "relations.json"), "w"),
              ensure_ascii=False)
    txt = summarize(rel)
    with open(os.path.join(unit, "relations.txt"), "w") as f:
        f.write(txt + "\n")
    return rel, txt


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    if args[0] == "--tree":
        root = args[1]
        targets = sorted(glob.glob(os.path.join(root, "**", "frames.json"),
                                   recursive=True))
    else:
        targets = args
    print(f"[relations] 대상 {len(targets)}건")
    for t in targets:
        try:
            rel, txt = process_unit(t)
            unit = t if not t.endswith(".json") else os.path.dirname(t)
            print(f"-- {unit}: 쌍 {len(rel['pairs_summary'])}개")
            print(txt)
        except Exception as e:
            print(f"  실패 {t}: {e}")


if __name__ == "__main__":
    main()
