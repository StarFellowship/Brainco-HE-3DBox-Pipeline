"""같은 trajectory에서 파이프라인 5단계를 전부 렌더 — 어느 단계에서 깨졌는지 추적용.

피드백: "파이프라인 중간 결과들은 모두 같은 Trajectory에서 봐야 3D BBOX만이 잘못되었는지
판단이 가능하다. 단계별로 다른 예시를 쓸 것이면 모든 예시에 대한 모든 단계를 달라."

그래서 한 (에피소드·카메라·프레임)에 대해 6개 패널을 낸다:
  p0 원본 RGB
  p1 ① GroundingDINO 2D 박스   (BOX2D.mp4 = 파이프라인이 실제로 그린 것)
  p2 ② SAM 2.1 픽셀 마스크      (저장된 box2d를 SAM에 넣어 동일 마스크 재현)
  p3 ③ Depth                    (Brainco=UniDepthV2 추정 / HE=실측)
  p4 ④ Back-projection 점군      (마스크 픽셀을 3D로 쏜 뒤 위에서 본 모습)
  p5 ⑤ 3D 박스                  (A_visible.mp4 = 파이프라인이 실제로 그린 것)

p1·p5는 파이프라인 산출 영상에서 그대로 뽑으므로 "보고용으로 다시 그린 그림"이 아니다.
한글 라벨·판정 배지는 로컬 합성 단계에서 붙인다(cv2가 한글을 못 그림).

출력: ~/task3/traj/<키>_p{0..5}_*.png + meta.json
"""
import os, sys, json, shutil
import numpy as np
import cv2

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
from geometry import Intrinsics
import run_review as RV

R6 = f"{ROOT}/review/r6"
OUT = f"{ROOT}/traj"
DEV = "cuda:0"

# depth 컬러맵 공통 범위 — 사례 간 색을 비교할 수 있게 고정한다
DMIN, DMAX = 0.15, 2.0

# (키, 유닛, 라벨, 프레임, 직전프레임도_렌더할지)
CASES = [
    ("T1_oreo_head",    "brainco/GraspOreo_ep5/cam_left_high",           "oreo",          6, False),
    ("T2_laptop_he",    "he/Articulated_ep280",                          "laptop",        7, False),
    ("T3_charger_head", "brainco/PickCharger_ep5/cam_left_high",         "charger",     109, False),
    ("T4_duster_he",    "he/Tool_use_ep8198",                            "duster",       32, False),
    ("T5_cube_wrist",   "brainco/GraspRubiksCube_ep5/cam_right_wrist",   "rubiks cube", 196, True),
    ("T6_apple_head",   "brainco/PickApple_ep5/cam_left_high",           "apple",        49, False),
]


def log(m):
    print(m, flush=True)


def get_seq(unit):
    parts = unit.rstrip("/").split("/")
    if unit.startswith("brainco/"):
        task, ep = parts[1].split("_ep")
        files, tmp = RV.frames_brainco(task, int(ep), parts[2])
        return [(cv2.imread(f), None) for f in files], tmp, False
    return RV.frames_he(int(parts[1].split("_ep")[1])), None, True


def grab(video, idx):
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0:
        cap.release(); return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, n - 1))
    ok, img = cap.read(); cap.release()
    return img if ok else None


def rec_at(unit, label, fi):
    d = json.load(open(f"{R6}/{unit}/frames.json"))
    for x in d[fi]["r"]:
        if x["label"] == label:
            return x
    return None


def mask_panel(img, mask, box, accepted):
    vis = img.copy()
    ov = vis.copy()
    ov[mask] = (0, 235, 255)
    vis = cv2.addWeighted(ov, 0.45, vis, 0.55, 0)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 140, 255), 2)
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        col = (255, 255, 255) if accepted else (80, 80, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 1)
    return vis, int(mask.sum()), len(cnts)


def depth_panel(depth, mask=None):
    dd = np.clip(depth, DMIN, DMAX)
    u = ((dd - DMIN) / (DMAX - DMIN) * 255).astype(np.uint8)
    cm = cv2.applyColorMap(u, cv2.COLORMAP_TURBO)
    if mask is not None:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cm, cnts, -1, (255, 255, 255), 2)
    return cm


def backproject(mask, depth, K, dscale, adx):
    H, W = depth.shape
    m = np.roll(mask, adx, axis=1) if adx else mask
    ys, xs = np.nonzero(m[:H, :W])
    if len(xs) == 0:
        return np.empty((0, 3))
    z = depth[ys, xs].astype(np.float64) * dscale
    ok = np.isfinite(z) & (z > 0.05) & (z < 8.0)
    xs, ys, z = xs[ok], ys[ok], z[ok]
    return np.stack([(xs - K.cx) / K.fx * z, (ys - K.cy) / K.fy * z, z], 1)


def bev_panel(pts, box=None, wpx=640, hpx=480):
    """점군을 위에서 본 그림. 가로=좌우 폭, 세로=카메라 방향 깊이.

    초록 점 = 역투영된 원본 점군(필터 전).
    주황 사각 = 파이프라인이 최종 채택한 3D 박스의 발자국(중심·크기 그대로).
    둘을 같은 축에 겹쳐 그려야 '점군은 이런데 박스는 이렇게 잡혔다'가 보인다.
    box = (center_x, center_z, size_x, size_z) 단위 m.
    """
    pan = np.full((hpx, wpx, 3), 24, np.uint8)
    if len(pts) < 5:
        return pan, None
    lo, hi = np.percentile(pts, [1, 99], axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    q = pts[keep] if keep.sum() > 5 else pts
    # 표시 중심·배율은 점군과 박스를 모두 담도록 잡는다
    xs_all, zs_all = list(q[:, 0]), list(q[:, 2])
    if box:
        bcx, bcz, bsx, bsz = box
        xs_all += [bcx - bsx / 2, bcx + bsx / 2]
        zs_all += [bcz - bsz / 2, bcz + bsz / 2]
    cx, cz = (min(xs_all) + max(xs_all)) / 2, (min(zs_all) + max(zs_all)) / 2
    span = max(max(xs_all) - min(xs_all), max(zs_all) - min(zs_all), 0.04) * 1.35
    s = min(wpx, hpx) / span
    g = 0.05 * s                                   # 5cm 격자
    if g > 8:
        k = 0
        while k * g < max(wpx, hpx):
            for sign in (1, -1):
                cvx = int(wpx / 2 + sign * k * g)
                cvy = int(hpx / 2 + sign * k * g)
                cv2.line(pan, (cvx, 0), (cvx, hpx), (46, 46, 46), 1)
                cv2.line(pan, (0, cvy), (wpx, cvy), (46, 46, 46), 1)
            k += 1
    u = ((q[:, 0] - cx) * s + wpx / 2).astype(int)
    v = ((q[:, 2] - cz) * s + hpx / 2).astype(int)
    ok = (u >= 0) & (u < wpx) & (v >= 0) & (v < hpx)
    for a, b in zip(u[ok], v[ok]):
        cv2.circle(pan, (a, b), 1, (120, 235, 120), -1)
    ex, ez = float(np.ptp(q[:, 0])), float(np.ptp(q[:, 2]))
    if box:
        bcx, bcz, bsx, bsz = box
        x1 = int((bcx - bsx / 2 - cx) * s + wpx / 2); x2 = int((bcx + bsx / 2 - cx) * s + wpx / 2)
        z1 = int((bcz - bsz / 2 - cz) * s + hpx / 2); z2 = int((bcz + bsz / 2 - cz) * s + hpx / 2)
        cv2.rectangle(pan, (x1, z1), (x2, z2), (90, 200, 255), 2)
    # 5cm 눈금 안내
    cv2.line(pan, (20, hpx - 24), (20 + int(g), hpx - 24), (210, 210, 210), 2)
    cv2.putText(pan, "5cm", (20, hpx - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (210, 210, 210), 1, cv2.LINE_AA)
    cv2.arrowedLine(pan, (wpx - 34, hpx - 92), (wpx - 34, hpx - 40),
                    (210, 210, 210), 2, tipLength=0.3)
    cv2.putText(pan, "camera", (wpx - 92, hpx - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (210, 210, 210), 1, cv2.LINE_AA)
    return pan, (ex, ez)


def main():
    from models_wrap import Segmenter, DepthEstimator
    os.makedirs(OUT, exist_ok=True)
    seg, dep = Segmenter(DEV), DepthEstimator(DEV)
    log("모델 로드 완료")
    meta = {}

    for key, unit, lb, fi, want_prev in CASES:
        log(f"--- {key}  {unit} / {lb} f{fi}")
        try:
            seq, tmp, is_he = get_seq(unit)
            img = seq[fi][0]
            H, W = img.shape[:2]
            r = rec_at(unit, lb, fi)
            m = dict(unit=unit, label=lb, frame=fi, is_he=is_he,
                     accepted=bool(r and r.get("accepted")), src=r.get("src") if r else None,
                     det_score=round(float(r.get("det_score", 0)), 3) if r else None,
                     dmed=round(float(r.get("dmed", 0)), 3) if r else None,
                     size_cm=[round(v * 100, 1) for v in r.get("size", [])] if r and r.get("size") else None,
                     n_points_pipe=r.get("n_points") if r else None,
                     mask_px_pipe=r.get("mask_px") if r else None)

            # p0 원본
            cv2.imwrite(f"{OUT}/{key}_p0_orig.png", img)
            # p1 2D 박스 (파이프라인 산출 영상)
            b1 = grab(f"{R6}/{unit}/BOX2D.mp4", fi)
            if b1 is not None:
                cv2.imwrite(f"{OUT}/{key}_p1_box2d.png", b1)
            # p5 3D 박스 (파이프라인 산출 영상)
            b5 = grab(f"{R6}/{unit}/A_visible.mp4", fi)
            if b5 is not None:
                cv2.imwrite(f"{OUT}/{key}_p5_box3d.png", b5)

            # depth
            if is_he:
                d = seq[fi][1]
                if d.ndim == 1:
                    d = d.reshape(d.size // W, W)
                if d.shape != (H, W):
                    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
                depth = d.astype(np.float64) * 1e-3
                K = Intrinsics.from_fov(W, H, 70.0); adx = -20
                m["depth_src"] = "HE 실측"
            else:
                dm, Kp = dep(img)
                depth = dm.astype(np.float64)
                K = (Intrinsics(float(Kp[0, 0]), float(Kp[1, 1]), float(Kp[0, 2]), float(Kp[1, 2]))
                     if Kp is not None else Intrinsics.from_fov(W, H, 70.0))
                adx = 0
                m["depth_src"] = "UniDepthV2 추정"

            # p2 마스크 — 저장된 box2d를 SAM에 그대로 투입
            mask = None
            if r and r.get("box2d"):
                mk = seg(img, np.array([r["box2d"]], dtype=np.float32))
                if len(mk):
                    mask = mk[0]
                    vis, px, nc = mask_panel(img, mask, r["box2d"], m["accepted"])
                    cv2.imwrite(f"{OUT}/{key}_p2_mask.png", vis)
                    m.update(mask_px=px, components=nc)
            if mask is None:
                cv2.imwrite(f"{OUT}/{key}_p2_mask.png", np.full((H, W, 3), 40, np.uint8))
                m.update(mask_px=0, components=0)

            # p3 depth
            cv2.imwrite(f"{OUT}/{key}_p3_depth.png", depth_panel(depth, mask))
            if mask is not None:
                mm = np.roll(mask, adx, axis=1) if adx else mask
                sel = depth[mm[:H, :W]]
                sel = sel[np.isfinite(sel) & (sel > 0.05)]
                if len(sel):
                    m["depth_median_m"] = round(float(np.median(sel)), 3)

            # 직전 프레임 depth (시간적 흔들림을 보여야 하는 사례)
            if want_prev and fi > 0:
                pimg = seq[fi - 1][0]
                if is_he:
                    pd = seq[fi - 1][1]
                    if pd.ndim == 1:
                        pd = pd.reshape(pd.size // W, W)
                    if pd.shape != (H, W):
                        pd = cv2.resize(pd, (W, H), interpolation=cv2.INTER_NEAREST)
                    pdepth = pd.astype(np.float64) * 1e-3
                else:
                    pdm, _ = dep(pimg)
                    pdepth = pdm.astype(np.float64)
                cv2.imwrite(f"{OUT}/{key}_p3prev_depth.png", depth_panel(pdepth))
                pr = rec_at(unit, lb, fi - 1)
                m["prev_frame"] = fi - 1
                m["prev_dmed"] = round(float(pr.get("dmed", 0)), 3) if pr else None

            # p4 점군 BEV — 원본 점군 위에 파이프라인 최종 박스 발자국을 겹쳐 그린다
            if mask is not None:
                pts = backproject(mask, depth, K, 1.0, adx)
                np.save(f"{OUT}/{key}_points.npy", pts.astype(np.float32))
                bx = None
                if r and r.get("center") and r.get("size"):
                    c, sz = r["center"], r["size"]
                    bx = (float(c[0]), float(c[2]), float(sz[0]), float(sz[2]))
                bev, ext = bev_panel(pts, bx)
                cv2.imwrite(f"{OUT}/{key}_p4_bev.png", bev)
                m["n_points"] = int(len(pts))
                m["bev_wd_cm"] = [round(ext[0] * 100, 1), round(ext[1] * 100, 1)] if ext else None
                m["box_footprint_cm"] = [round(bx[2] * 100, 1), round(bx[3] * 100, 1)] if bx else None
            else:
                cv2.imwrite(f"{OUT}/{key}_p4_bev.png", np.full((480, 640, 3), 24, np.uint8))
                m["n_points"] = 0

            meta[key] = m
            log(f"  accepted={m['accepted']} src={m['src']} mask={m.get('mask_px')} "
                f"comp={m.get('components')} pts={m.get('n_points')} "
                f"size={m.get('size_cm')} depth={m.get('depth_median_m')} ({m['depth_src']})")
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
        except Exception as e:
            log(f"  !! {key}: {e}")

    json.dump(meta, open(f"{OUT}/meta.json", "w"), ensure_ascii=False, indent=1)
    log(f"완료 -> {OUT}")


if __name__ == "__main__":
    main()
