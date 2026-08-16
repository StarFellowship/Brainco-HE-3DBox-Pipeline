"""
Task 3 파이프라인 4~5단계: Back-projection + 3D Bounding Box
GPU 불필요 (CPU only). GroundingDINO/SAM/UniDepth 결과가 없어도
실측 depth + 임의 마스크로 단독 검증 가능.

RoboBrain(RefSpatial) 파이프라인 기준:
  4단계 Back-projection : 마스크 + depth + intrinsics -> 객체별 point cloud
  5단계 3D Bounding Box : point cloud -> axis-aligned 3D box
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2


# ---------------------------------------------------------------- intrinsics
@dataclass
class Intrinsics:
    """카메라 내부 파라미터. fx,fy=초점거리(px), cx,cy=주점(px)"""
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = 60.0) -> "Intrinsics":
        """intrinsics가 없을 때 수평 FOV 가정으로 근사.
        WildCamera 추정치가 들어오기 전 임시 대체용 (STEP 0.5 검증에 사용)."""
        fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0)

    def as_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)


# ------------------------------------------------------- 4단계 back-projection
def backproject(
    depth: np.ndarray,              # (H,W) 거리값. 단위는 depth_scale로 m 변환
    K: Intrinsics,
    mask: Optional[np.ndarray] = None,   # (H,W) bool. None이면 전체 화면
    depth_scale: float = 1e-3,      # mm -> m (HE/AgiBot 실측 depth 기준)
    valid_range: Tuple[float, float] = (0.05, 10.0),   # m. 이 밖은 버림
) -> np.ndarray:
    """마스크 영역의 픽셀을 카메라 좌표계 3D 점군으로 복원한다.

    핀홀 모델:  X = (u - cx) * Z / fx,   Y = (v - cy) * Z / fy,   Z = depth
    반환: (N,3) float64,  카메라 좌표계 (x=우, y=하, z=전방), 단위 m
    """
    H, W = depth.shape
    z = depth.astype(np.float64) * depth_scale

    valid = np.isfinite(z) & (z > valid_range[0]) & (z < valid_range[1])
    if mask is not None:
        valid &= mask.astype(bool)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)

    v, u = np.nonzero(valid)
    zz = z[v, u]
    x = (u - K.cx) * zz / K.fx
    y = (v - K.cy) * zz / K.fy
    return np.stack([x, y, zz], axis=1)


# ------------------------------------------------------------ 점군 정제
def filter_points(
    pts: np.ndarray,
    percentile: Tuple[float, float] = (5.0, 95.0),   # depth 이상치 클리핑
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    use_statistical: bool = True,
    foreground_mad: Optional[float] = 3.0,   # None이면 비활성. depth 주 모드만 취함
) -> np.ndarray:
    """RoboBrain 리포트에 세부가 없어 표준 관행으로 구현한 정제 단계.
    ⓪ foreground 분리 : depth 중앙값 ± k·MAD 밖(=배경 혼입) 제거   ← 실측으로 필요성 확인
    ① depth percentile 클리핑 → 경계 튐 제거
    ② statistical outlier removal → 산발 노이즈 제거 (scipy, open3d 불필요)

    ⓪이 필요한 이유(STEP 0.5 실측): 마스크가 정확해도 RGB-depth 정렬 오차·
    경계 픽셀 때문에 배경 거리값이 섞이며, 이때 z 범위가 실제 물체의 5배까지 부풀었다.
    """
    if len(pts) == 0:
        return pts

    if foreground_mad is not None and len(pts) > 50:
        z = pts[:, 2]
        med = np.median(z)
        mad = np.median(np.abs(z - med)) * 1.4826      # 표준편차 환산
        if mad > 1e-6:
            pts = pts[np.abs(z - med) <= foreground_mad * mad]
        if len(pts) == 0:
            return pts

    lo, hi = np.percentile(pts[:, 2], percentile)
    pts = pts[(pts[:, 2] >= lo) & (pts[:, 2] <= hi)]
    if len(pts) < nb_neighbors + 1 or not use_statistical:
        return pts

    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=nb_neighbors + 1)      # 자기 자신 포함
    mean_d = d[:, 1:].mean(axis=1)
    thr = mean_d.mean() + std_ratio * mean_d.std()
    return pts[mean_d <= thr]


def same_depth_components(mask: np.ndarray, depth: np.ndarray, dscale: float,
                          box2d, dz: float = 0.03) -> np.ndarray:
    """마스크의 연결 성분 중 **같은 물체로 볼 수 있는 것들의 합집합**을 남긴다.

    '가장 큰 덩어리만' 남기면 얇게 이어진 부속물이 잘려나간다 — 라운드2에서
    토끼 귀와 꽃 줄기가 3D 박스에서 빠진 원인이다. 반면 충전기 케이블처럼
    깊이가 다른 부속물은 여전히 배제해야 한다.
    두 요구를 만족하는 기준이 **깊이 근접성**이다: 최대 성분과 depth 중앙값이
    dz 이내이고 검출 박스 안에 있는 성분만 합친다.
    """
    m = np.ascontiguousarray(mask.astype(np.uint8))
    if m.sum() < 30:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    k0 = int(np.argmax(areas)) + 1
    d0 = depth[lab == k0]
    d0 = d0[np.isfinite(d0) & (d0 > 0)]
    if d0.size < 10:
        return lab == k0
    z0 = float(np.median(d0)) * dscale
    x1, y1, x2, y2 = box2d
    out = (lab == k0)
    a0 = float(areas[k0 - 1])
    for i in range(1, n):
        if i == k0 or areas[i - 1] < 20:
            continue
        # 부속물(귀·줄기)은 본체보다 작다. 이 제한이 없으면 같은 깊이에 있는
        # 로봇 손이 통째로 합쳐진다(스모크 실측: charger 6x6x2 -> 7x25x24cm).
        if areas[i - 1] > 0.5 * a0:
            continue
        cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2
        cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        di = depth[lab == i]
        di = di[np.isfinite(di) & (di > 0)]
        if di.size < 10:
            continue
        if abs(float(np.median(di)) * dscale - z0) <= dz:
            out |= (lab == i)
    return out

def mirror_extend_z(box3d, pts, min_pts: int = 30):
    """단일 시점 두께 결손을 대칭 가정으로 복원한다 (방식 B 전용).

    카메라는 물체 앞면만 보므로 AABB의 깊이 방향 두께가 구조적으로 과소 추정된다
    (실측: 두꺼운 물체에서 z/min(x,y)가 0.25~0.47로 붕괴).
    점군의 앞면 깊이를 기준으로 뒤쪽을 대칭 확장하되, 관측된 두께의 2.5배를
    넘지 않게 제한한다.
    """
    s = np.array(box3d.size, dtype=float)
    if pts is None or len(pts) < min_pts:
        return s
    z = pts[:, 2]
    z_front = float(np.percentile(z, 5))
    z_med = float(np.median(z))
    depth_obs = float(s[2])
    est = 2.0 * (z_med - z_front)
    if est <= depth_obs:
        return s
    out = s.copy()
    # 상한을 관측 두께의 1.8배로 둔다. 물체가 기울어져 있으면 z_med-z_front가
    # 두께가 아니라 기울기를 재므로 무제한 확장은 위험하다.
    out[2] = min(est, depth_obs * 1.8 if depth_obs > 1e-4 else est)
    return out


def largest_component(mask: np.ndarray, min_ratio: float = 0.15) -> np.ndarray:
    """마스크에서 가장 큰 연결 덩어리만 남긴다.

    필요성(V2 1차 실측): 충전기는 본체+케이블, 컵은 컵+로봇팔 그림자가 하나의 마스크로
    묶여 나왔고, 이들이 전부 한 점군으로 합쳐져 박스가 실제의 2~3배로 부풀었다.
    (charger 8x22x21cm, red cup 13x17x30cm)
    가장 큰 덩어리가 전체의 min_ratio 미만이면 물체가 쪼개진 상황이라 보고 원본을 유지한다.
    """
    import cv2
    m = mask.astype(np.uint8)
    if m.sum() < 30:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:                                   # 배경 + 덩어리 1개
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    if areas[k - 1] / max(m.sum(), 1) < min_ratio:
        return mask
    return lab == k


def cluster_depth(pts: np.ndarray, gap: float = 0.06) -> np.ndarray:
    """점군을 z(깊이)로 정렬해 gap보다 큰 간격에서 끊고, 점이 가장 많은 구간만 취한다.

    MAD 필터는 대칭 범위를 쓰므로 '물체 + 그로부터 떨어진 팔' 처럼 한쪽으로만 치우친
    혼입을 놓친다. 이 함수는 깊이 방향으로 실제 분리된 덩어리를 골라낸다.
    """
    if len(pts) < 50:
        return pts
    z = np.sort(pts[:, 2])
    idx = np.argsort(pts[:, 2])
    breaks = np.where(np.diff(z) > gap)[0]
    if len(breaks) == 0:
        return pts
    seg_start = np.concatenate([[0], breaks + 1])
    seg_end = np.concatenate([breaks + 1, [len(z)]])
    sizes = seg_end - seg_start
    b = int(np.argmax(sizes))
    keep = idx[seg_start[b]:seg_end[b]]
    return pts[keep]


def align_mask_to_depth(mask: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    """RGB 기준 마스크를 depth 좌표계로 평행이동 보정.
    HE 데이터 실측 결과 depth가 RGB 대비 약 dx=-20px 어긋나 있어 필요.

    zero-fill 시프트다. np.roll을 쓰면 잘린 열이 **반대편 가장자리로 감겨 들어와**
    화면 끝 물체가 반대편 depth를 읽는다 — 라운드1에서 HE Locomanip의 병 박스가
    4.2x1.3x4.3cm 쓰레기로 나와 이력을 오염시킨 원인이었다(재투영 오차 du=+606px).
    """
    if dx == 0 and dy == 0:
        return mask
    out = np.zeros_like(mask)
    H, W = mask.shape
    xs_dst = slice(max(0, dx), W + min(0, dx))
    xs_src = slice(max(0, -dx), W + min(0, -dx))
    ys_dst = slice(max(0, dy), H + min(0, dy))
    ys_src = slice(max(0, -dy), H + min(0, -dy))
    out[ys_dst, xs_dst] = mask[ys_src, xs_src]
    return out


def check_reprojection(box3d, box2d, K: Intrinsics, align_dx: int = 0):
    """3D 박스가 그것을 만든 2D 증거와 정합하는지 검사한다.

    핀홀 모델상 반드시 성립해야 하는 자기정합 조건이라 물체 종류·태스크·카메라에
    의존하지 않는다. 반환 (du, dv, rx, ry, inside):
      du,dv : 3D 중심을 재투영한 점이 2D 박스 중심에서 벗어난 픽셀 거리
      rx,ry : 3D 박스 폭·높이 / 2D 박스가 그 깊이에서 가리키는 기대 폭·높이
      inside: 재투영점이 2D 박스를 1.25배 팽창시킨 영역 안에 있는가

    주의: rx/ry는 fx가 소거되므로 '살아남은 점군이 2D 박스를 얼마나 채우는가'를 재는
    양이다. 세로(ry)는 물체가 박스 상하단까지 꽉 차는 경우가 드물어 가로(rx)보다
    계통적으로 작게 나온다(실측 rx 0.90~0.97 vs ry 0.61~0.78). 임계를 축별로 달리 둔다.
    """
    x1, y1, x2, y2 = box2d
    x1 -= align_dx; x2 -= align_dx          # 마스크를 옮겼으므로 2D 증거도 같은 계로
    cx3, cy3, z = float(box3d.center[0]), float(box3d.center[1]), float(box3d.center[2])
    if z <= 1e-6:
        return 0.0, 0.0, 0.0, 0.0, False
    u = cx3 * K.fx / z + K.cx
    v = cy3 * K.fy / z + K.cy
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    ecx, ecy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    du, dv = u - ecx, v - ecy
    inside = (abs(du) <= 0.625 * bw * 1.0) and (abs(dv) <= 0.625 * bh * 1.0)
    w_exp = bw * z / K.fx
    h_exp = bh * z / K.fy
    rx = float(box3d.size[0]) / max(w_exp, 1e-9)
    ry = float(box3d.size[1]) / max(h_exp, 1e-9)
    return float(du), float(dv), float(rx), float(ry), bool(inside)


# ------------------------------------------------------------- 5단계 3D box
@dataclass
class Box3D:
    """axis-aligned 3D bounding box (카메라 좌표계, 단위 m)"""
    center: np.ndarray   # (3,)
    size: np.ndarray     # (3,) = (width_x, height_y, depth_z)
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    n_points: int

    def corners(self) -> np.ndarray:
        """(8,3) 코너 좌표. 2D 재투영 시각화에 사용"""
        x0, y0, z0 = self.min_xyz
        x1, y1, z1 = self.max_xyz
        return np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                         [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])

    def summary(self) -> str:
        c, s = self.center, self.size
        return (f"center=({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})m  "
                f"size=({s[0]:.3f} x {s[1]:.3f} x {s[2]:.3f})m  pts={self.n_points}")


def fit_box3d(pts: np.ndarray, pct: float = 1.0) -> Optional[Box3D]:
    """정제된 점군에서 axis-aligned 3D box 산출 (리포트 명시 방식).

    pct<1이면 축별 극값 대신 [pct, 100-pct] 백분위를 쓴다. 극값은 점 1개로 결정되어
    프레임 간 잡음이 크고(실측: 프레임간 z 변동 p90 1.62배), 그 잡음이 크기 게이트에
    그대로 실려 정상 프레임을 기각시킨다. 백분위는 박스 자체를 안정화한다.
    """
    if len(pts) < 10:
        return None
    if pct and pct > 0 and len(pts) >= 50:
        mn = np.percentile(pts, pct, axis=0)
        mx = np.percentile(pts, 100.0 - pct, axis=0)
    else:
        mn, mx = pts.min(axis=0), pts.max(axis=0)
    return Box3D(center=(mn + mx) / 2.0, size=mx - mn,
                 min_xyz=mn, max_xyz=mx, n_points=len(pts))


# --------------------------------------------------------- 3D -> 2D 재투영
def project_points(pts: np.ndarray, K: Intrinsics) -> np.ndarray:
    """3D 점을 이미지 평면으로 투영. 반환 (N,2) 픽셀 좌표."""
    z = np.clip(pts[:, 2], 1e-6, None)
    u = pts[:, 0] * K.fx / z + K.cx
    v = pts[:, 1] * K.fy / z + K.cy
    return np.stack([u, v], axis=1)


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),      # 앞면
             (4, 5), (5, 6), (6, 7), (7, 4),      # 뒷면
             (0, 4), (1, 5), (2, 6), (3, 7)]      # 연결


def draw_box3d(img, box: Box3D, K: Intrinsics, color=(0, 165, 255), thickness=2, label=None):
    """3D 박스를 원본 이미지 위에 재투영해 그린다 (OpenCV BGR in-place)."""
    import cv2
    pts2d = project_points(box.corners(), K).astype(int)
    for a, b in BOX_EDGES:
        cv2.line(img, tuple(pts2d[a]), tuple(pts2d[b]), color, thickness, cv2.LINE_AA)
    if label:
        x, y = pts2d[:, 0].min(), pts2d[:, 1].min()
        cv2.putText(img, label, (int(x), max(18, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img
