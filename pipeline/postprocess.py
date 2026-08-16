"""후처리 (5단계 이후) — 2D box 산출물 생성 + ghost 꼬리 트리밍.

## ① 2D Bounding Box 출력 (task: 3D 파이프라인 앞단의 2D box를 산출물로 노출)

2D box는 파이프라인 1단계(GroundingDINO)에서 이미 프레임별로 계산되어
frames.json의 `box2d` 필드에 저장돼 있다. 여기서는 그것을 두 가지 표준 산출물로 만든다.
  - boxes2d.json : 프레임별 2D box 목록 (label, xyxy, score, src, is_target)
  - BOX2D.mp4    : 원본 프레임 위 2D box만 그린 오버레이 영상

## ② Ghost 꼬리 트리밍 (task: 화면에 없는 물체에 박스가 남는 현상 해결)

r6 전수 분석 결과, ghost의 실체는 **"마지막 실검출(det/redet) 이후 에피소드 끝까지
이어지는 꼬리 전파(prop)"**였다(예: PickTissues 손목 81프레임, PickApple 손목 50프레임).
반면 에피소드 **중간**의 긴 전파 런(GraspOreo 머리 62프레임)은 이후 실검출로 회복되는
정상 추적이다 — 이걸 건드리면 잘 되던 결과가 무너진다(r3에서 실제로 겪은 소탐대실).

그래서 **런타임 게이트가 아니라 오프라인 후처리**로 자른다:
  - 방식 A: 마지막 실검출 이후의 전파 승인 프레임은 그리지 않는다.
  - 방식 B: 마지막 실검출 이후 TAIL_B_SEC(1.0초)까지만 추정을 유지하고 이후 중단한다.
  - 에피소드 중간 구간은 **원천적으로 건드릴 수 없다**(마지막 실검출 이전이므로).
  - 원본 결과는 보존하고 별도 폴더(r6post)에 재렌더한다 → 전/후 비교 가능.

사용법 (서버):
  python postprocess.py box2d <r6디렉토리>          # 전 유닛 boxes2d.json + BOX2D.mp4
  python postprocess.py ghosttrim <r6디렉토리>      # 꼬리 트리밍 재렌더 (영향 유닛만)
  python postprocess.py json_only <r6디렉토리>      # boxes2d.json만 (GPU/프레임 불필요)
"""
import os, sys, json, glob, time
import numpy as np

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")

FPS = 10.0            # r6 검증 실행 기준 (run_review STRIDE=3)
TAIL_B_SEC = 1.0      # 방식 B가 마지막 실검출 이후 추정을 유지할 최대 시간
CONFIRM_N = 8         # 이 횟수만큼 실검출돼야 그 라벨을 확립된 것으로 본다


# ------------------------------------------------------------------ 공통
def load_unit(unit_dir):
    recs = json.load(open(f"{unit_dir}/frames.json"))
    stats = json.load(open(f"{unit_dir}/stats.json"))
    return recs, stats


def last_evidence(recs, label):
    """이 라벨의 마지막 실검출(det/redet 승인) 프레임 번호. 없으면 -1."""
    last = -1
    for i, r in enumerate(recs):
        for x in r["r"]:
            if x["label"] == label and x.get("accepted") and x.get("src") in ("det", "redet"):
                last = i
    return last


def established(recs, label, n=CONFIRM_N):
    """에피소드 전체에서 실검출이 n회 이상 누적된 라벨인가.

    한 번도 제대로 못 본 라벨의 전파 박스는 근거가 없다. 육안 검증에서 이런 라벨
    (PickApple 좌손목 plate, PickDrink 좌손목 plate)의 표본 6장이 모두 유령으로
    확인됐다. 반대로 확립된 라벨의 초반 전파는 정상 박스가 많아(PickCharger 충전기,
    PickToothpaste 치약) 일괄로 자르면 멀쩡한 결과를 잃는다.
    """
    seen = 0
    for r in recs:
        for x in r["r"]:
            if x["label"] == label and x.get("accepted") and x.get("src") in ("det", "redet"):
                seen += 1
                if seen >= n:
                    return True
    return False


# ------------------------------------------------------------------ ① 2D box
def export_boxes2d(unit_dir):
    recs, stats = load_unit(unit_dir)
    is_tgt = {lb: bool(v.get("is_target")) for lb, v in stats["stats"].items()}
    frames = []
    n_boxes = 0
    for r in recs:
        boxes = []
        for x in r["r"]:
            if not x.get("accepted") or "box2d" not in x:
                continue
            b = x["box2d"]
            boxes.append(dict(label=x["label"],
                              x1=round(float(b[0]), 1), y1=round(float(b[1]), 1),
                              x2=round(float(b[2]), 1), y2=round(float(b[3]), 1),
                              score=float(x.get("det_score", 0.0)), src=x.get("src", ""),
                              is_target=is_tgt.get(x["label"], False)))
            n_boxes += 1
        frames.append(dict(frame=r["f"], boxes=boxes))
    out = dict(unit=os.path.relpath(unit_dir), fps=FPS, n_frames=len(recs),
               n_boxes=n_boxes, frames=frames)
    json.dump(out, open(f"{unit_dir}/boxes2d.json", "w"), ensure_ascii=False)
    return n_boxes


def render_box2d_video(unit_dir, seq):
    """원본 프레임 위에 2D box만 그린 영상. seq = [(idx, BGR이미지)]"""
    import cv2
    d = json.load(open(f"{unit_dir}/boxes2d.json"))
    bymap = {f["frame"]: f["boxes"] for f in d["frames"]}
    H, W = seq[0][1].shape[:2]
    vw = cv2.VideoWriter(f"{unit_dir}/BOX2D.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for idx, img in seq:
        vis = img.copy()
        for b in bymap.get(idx, []):
            col = (0, 220, 120) if b["is_target"] else (200, 160, 60)
            cv2.rectangle(vis, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), col, 2)
            cv2.putText(vis, f"{b['label']} {b['score']:.2f}",
                        (int(b["x1"]), max(15, int(b["y1"]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        vw.write(vis)
    vw.release()


# ------------------------------------------------------------------ ② ghost 트리밍
def trim_plan(unit_dir):
    """유닛의 트리밍 계획. 반환 {label: (last_evi, trimA수, trimB수)} — 대상 없으면 {}"""
    recs, stats = load_unit(unit_dir)
    plan = {}
    b_cap = int(TAIL_B_SEC * FPS)
    for lb in stats["stats"]:
        # 미확립 라벨은 기준점을 -1로 둬서 그 라벨의 전파 전 구간이 대상이 되게 한다.
        # 실검출 프레임은 그대로 남으므로 관측 사실 자체는 보존된다.
        le = last_evidence(recs, lb) if established(recs, lb) else -1
        trimA = trimB = 0
        for i, r in enumerate(recs):
            for x in r["r"]:
                if x["label"] != lb:
                    continue
                if i > le and x.get("accepted") and x.get("src") == "prop":
                    trimA += 1
                if i > le + b_cap and (x.get("accepted_amodal") or x.get("coasting")):
                    trimB += 1
        if trimA >= 3 or trimB >= 3:          # 잡음 수준(1~2프레임)은 손대지 않는다
            plan[lb] = (le, trimA, trimB)
    return plan


def apply_trim(unit_dir, out_dir, seq, K_draw):
    """트리밍을 적용해 A/B 영상을 재렌더. 원본은 보존한다."""
    import cv2
    from geometry import Box3D, draw_box3d
    recs, stats = load_unit(unit_dir)
    plan = trim_plan(unit_dir)
    if not plan:
        return None
    os.makedirs(out_dir, exist_ok=True)
    b_cap = int(TAIL_B_SEC * FPS)
    H, W = seq[0][1].shape[:2]
    vA = cv2.VideoWriter(f"{out_dir}/A_visible.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    vB = cv2.VideoWriter(f"{out_dir}/B_amodal.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    pal = {"det": (0, 220, 120), "redet": (0, 200, 255), "prop": (255, 160, 40)}
    for idx, img in seq:
        r = recs[idx]
        visA, visB = img.copy(), img.copy()
        for x in r["r"]:
            lb = x["label"]
            le, _, _ = plan.get(lb, (10**9, 0, 0))
            # 방식 A — 꼬리 전파 제거
            if x.get("accepted") and not (idx > le and x.get("src") == "prop"):
                c, s = np.array(x["center"]), np.array(x["size"])
                b = Box3D(center=c, size=s, min_xyz=c - s / 2, max_xyz=c + s / 2, n_points=0)
                draw_box3d(visA, b, K_draw, color=pal.get(x.get("src"), (0, 220, 120)), thickness=2,
                           label=f"{lb} {s[0]*100:.0f}x{s[1]*100:.0f}x{s[2]*100:.0f}cm")
            # 방식 B — 꼬리 추정 시간 제한
            if (x.get("accepted_amodal") or x.get("accepted")) and "center_amodal" in x \
                    and idx <= le + b_cap:
                c, s = np.array(x["center_amodal"]), np.array(x["size_amodal"])
                est = x.get("coasting") or x.get("src") == "prop"
                col = (60, 120, 255) if est else pal.get(x.get("src"), (0, 220, 120))
                b = Box3D(center=c, size=s, min_xyz=c - s / 2, max_xyz=c + s / 2, n_points=0)
                draw_box3d(visB, b, K_draw, color=col, thickness=2,
                           label=f"{lb} {s[0]*100:.0f}x{s[1]*100:.0f}x{s[2]*100:.0f}cm"
                                 + (" (추정)" if est else ""))
        vA.write(visA); vB.write(visB)
    vA.release(); vB.release()
    json.dump({lb: dict(last_evidence=le, trimmed_A=a, trimmed_B=bb)
               for lb, (le, a, bb) in plan.items()},
              open(f"{out_dir}/trim_report.json", "w"), ensure_ascii=False, indent=1)
    return plan


# ------------------------------------------------------------------ 프레임 로더 (서버 전용)
def load_seq(unit_dir):
    """유닛 폴더 이름에서 (태스크, ep, 카메라)를 복원해 원본 프레임을 다시 뽑는다.
    run_review와 같은 파라미터(STRIDE=3, cap)라 프레임 정합이 보장된다."""
    import cv2
    import run_review as RV
    parts = unit_dir.rstrip("/").split("/")
    if "/brainco/" in unit_dir:
        task = parts[-2].split("_ep")[0]
        ep = int(parts[-2].split("_ep")[1])
        cam = parts[-1]
        files, tmp = RV.frames_brainco(task, ep, cam)
        return [(i, cv2.imread(f)) for i, f in enumerate(files)], tmp
    else:
        ep = int(parts[-1].split("_ep")[1])
        seq = RV.frames_he(ep)
        return [(i, img) for i, (img, _) in enumerate(seq)], None


def main():
    import shutil
    cmd, root = sys.argv[1], sys.argv[2]
    units = sorted(glob.glob(f"{root}/brainco/*/cam_*") + glob.glob(f"{root}/he/*_ep*"))
    units = [u for u in units if os.path.exists(f"{u}/frames.json")]
    print(f"[postprocess:{cmd}] 유닛 {len(units)}개")

    if cmd == "json_only":
        for u in units:
            n = export_boxes2d(u)
            print(f"  {os.path.relpath(u, root)}: 2D box {n}개")
        return

    from geometry import Intrinsics
    for u in units:
        t0 = time.time()
        export_boxes2d(u)
        plan = trim_plan(u) if cmd == "ghosttrim" else {}
        if cmd == "ghosttrim" and not plan:
            continue                      # ghost 없는 유닛은 손대지 않는다
        seq, tmp = load_seq(u)
        if not seq or seq[0][1] is None:
            print(f"  프레임 로드 실패: {u}"); continue
        H, W = seq[0][1].shape[:2]
        K = Intrinsics.from_fov(W, H, 70.0)
        if cmd == "box2d":
            render_box2d_video(u, seq)
            print(f"  BOX2D {os.path.relpath(u, root)} ({time.time()-t0:.0f}s)")
        elif cmd == "ghosttrim":
            out = u.replace("/r6/", "/r6post/")
            p = apply_trim(u, out, seq, K)
            print(f"  TRIM  {os.path.relpath(u, root)} -> {p} ({time.time()-t0:.0f}s)")
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
