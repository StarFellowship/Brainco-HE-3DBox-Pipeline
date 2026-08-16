"""큰 객체 확장 실험 (GPU) — table 외 대형 객체 포함, 모듈별 산출물 전부 출력.

지금까지의 큰객체 실험은 table/desk 프롬프트만 시험하고 게이트 차단 지점만 기록했다.
이번에는 장면마다 여러 대형 객체(basket, laptop, vase, shelf, chair, door ...)를 넣고,
객체마다 (1) 2D box  (2) 픽셀 분할 마스크  (3) 3D box  이미지를 각각 저장한다.

게이트는 대형객체 프로파일 수준으로 완화(면적·절대크기 상한 해제, 점군 필터만 유지).
출력: ~/task3/probe6/bigobj/<장면>_2dbox_all.png, <장면>_<n>_<객체>_{mask,3dbox}.png, meta.json
"""
import os, sys, json, shutil
import numpy as np
import cv2

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
from geometry import Intrinsics, Box3D, draw_box3d
import run_review as RV

OUT = f"{ROOT}/probe6/bigobj"
DEV = "cuda:0"

# (키, 종류, 소스, 프레임, 프롬프트) — 장면마다 화면에 보이는 대형 객체를 넣는다
SCENES = [
    ("bc_oreo_head",  "bc", ("GraspOreo", 5, "cam_left_high"), 10,
     "white table . plate . chair . desk ."),
    ("bc_doll_head",  "bc", ("PickDoll", 5, "cam_right_high"), 10,
     "white table . plate . chair ."),
    ("he_basic",      "he", 3800, 5,
     "table . basket . chair ."),
    ("he_articulated", "he", 280, 7,
     "desk . laptop . chair . floor ."),
    ("he_hri",        "he", 5598, 60,
     "table . vase . chair ."),
    ("he_locomanip",  "he", 7120, 40,
     "table . shelf . cabinet . door ."),
]

COLORS = [(0, 220, 120), (60, 160, 255), (0, 200, 255),
          (200, 120, 255), (120, 220, 0), (255, 120, 120)]


def log(m):
    print(m, flush=True)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def backproject(mask, depth, K, dscale, adx):
    H, W = depth.shape
    m = np.roll(mask, adx, axis=1) if adx else mask
    ys, xs = np.nonzero(m[:H, :W])
    if len(xs) == 0:
        return np.empty((0, 3))
    z = depth[ys, xs].astype(np.float64) * dscale
    ok = np.isfinite(z) & (z > 0.05) & (z < 8.0)
    xs, ys, z = xs[ok], ys[ok], z[ok]
    X = (xs - K.cx) / K.fx * z
    Y = (ys - K.cy) / K.fy * z
    return np.stack([X, Y, z], 1)


def robust_filter(pts):
    """백분위(2,98) + MAD 3.0 — 파이프라인 점군 필터와 같은 계열."""
    lo, hi = np.percentile(pts, [2, 98], axis=0)
    q = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
    if len(q) < 20:
        return q
    med = np.median(q, 0)
    mad = np.median(np.abs(q - med), 0) * 1.4826 + 1e-6
    return q[np.all(np.abs(q - med) <= 3.0 * mad, axis=1)]


def main():
    from models_wrap import Detector, Segmenter, DepthEstimator
    os.makedirs(OUT, exist_ok=True)
    det, seg = Detector(DEV), Segmenter(DEV)
    dep = DepthEstimator(DEV)
    log("모델 로드 완료")
    meta = {}
    for key, kind, src, fi, prompt in SCENES:
        tmp = None
        try:
            if kind == "bc":
                task, ep, cam = src
                files, tmp = RV.frames_brainco(task, ep, cam)
                img = cv2.imread(files[min(fi, len(files) - 1)])
                depth, Kp = dep(img)
                H, W = img.shape[:2]
                K = (Intrinsics(float(Kp[0, 0]), float(Kp[1, 1]),
                                float(Kp[0, 2]), float(Kp[1, 2]))
                     if Kp is not None else Intrinsics.from_fov(W, H, 70.0))
                dscale, adx = 1.0, 0
            else:
                seq = RV.frames_he(src)
                img, gd = seq[min(fi, len(seq) - 1)]
                H, W = img.shape[:2]
                d = gd
                if d.ndim == 1:
                    d = d.reshape(d.size // W, W)
                if d.shape != (H, W):
                    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
                depth = d
                K = Intrinsics.from_fov(W, H, 70.0)
                dscale, adx = 1e-3, -20

            boxes, phrases, scores = det(img, prompt)
            kept = []
            for oi in (np.argsort(-scores) if len(scores) else []):
                b, ph, sc = boxes[oi], phrases[oi], float(scores[oi])
                if sc < 0.30 or not ph:
                    continue
                if len([k for k in kept if k[1] == ph]) >= 2:
                    continue
                if any(iou(b, k[0]) > 0.5 for k in kept):
                    continue
                kept.append((b, ph, sc))

            vis2 = img.copy()
            objs = []
            for n, (b, ph, sc) in enumerate(kept):
                col = COLORS[n % len(COLORS)]
                x1, y1, x2, y2 = [int(v) for v in b]
                cv2.rectangle(vis2, (x1, y1), (x2, y2), col, 2)
                cv2.putText(vis2, f"{ph} {sc:.2f}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
                m = seg(img, np.array([b], dtype=np.float32))
                if len(m) == 0:
                    continue
                mask = m[0]
                slug = ph.replace(" ", "_")
                mv = img.copy()
                ov = mv.copy()
                ov[mask] = col
                mv = cv2.addWeighted(ov, 0.45, mv, 0.55, 0)
                cv2.rectangle(mv, (0, 0), (W, 26), (0, 0, 0), -1)
                cv2.putText(mv, f"{ph}  mask {int(mask.sum()):,}px "
                                f"({mask.sum()/(H*W)*100:.1f}% of frame)",
                            (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(f"{OUT}/{key}_{n}_{slug}_mask.png", mv)

                pts = backproject(mask, depth, K, dscale, adx)
                q = robust_filter(pts) if len(pts) >= 50 else pts
                rec = dict(phrase=ph, score=round(sc, 3), mask_px=int(mask.sum()),
                           area_pct=round(float(mask.sum()) / (H * W) * 100, 1),
                           n_points=int(len(q)))
                if len(q) >= 200:
                    mn, mx = q.min(0), q.max(0)
                    size = mx - mn
                    b3 = Box3D(center=(mn + mx) / 2, size=size,
                               min_xyz=mn, max_xyz=mx, n_points=len(q))
                    v3 = img.copy()
                    draw_box3d(v3, b3, K, color=col, thickness=2,
                               label=f"{ph} {size[0]*100:.0f}x{size[1]*100:.0f}"
                                     f"x{size[2]*100:.0f}cm")
                    cv2.imwrite(f"{OUT}/{key}_{n}_{slug}_3dbox.png", v3)
                    rec["size_cm"] = [round(float(v) * 100, 1) for v in size]
                    rec["absmax_cm"] = round(float(size.max()) * 100, 1)
                objs.append(rec)
                log(f"  {key} [{n}] {ph} {sc:.2f} -> {rec.get('size_cm')}")
            cv2.imwrite(f"{OUT}/{key}_2dbox_all.png", vis2)
            meta[key] = dict(prompt=prompt, frame=fi, objects=objs)
        except Exception as e:
            log(f"  !! {key}: {e}")
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
    json.dump(meta, open(f"{OUT}/meta.json", "w"), ensure_ascii=False, indent=1)
    log(f"완료 -> {OUT}")


if __name__ == "__main__":
    main()
