"""추적 + 검증 게이트 + 3D box 2가지 방식 (V4).

## 라운드1에서 드러난 공통 뿌리

증상 5가지(head 물체별 편차 / wrist 유령 / wrist 대형물체 침묵 / 옮기는 중 소실 /
A·B 무차이)는 모두 같은 원인의 다른 단면이었다.

1. **판정 기준이 스케일 가변량이었다.** 화면 면적비 0.12는 카메라-물체 거리에 의존한다.
   겉보기 면적은 (물체크기·f)²/z²에 비례하므로 head(z≈0.8m)와 wrist(z≈0.2m)에서 같은
   물체가 100배 차이 난다. 축별 크기 게이트(2.5배/4배)도 회전만으로 √3배 변한다.
2. **이력이 자기 자신을 오염시켰다.** 게이트는 자기 이력 중앙값과만 비교하는데 승인분을
   무조건 이력에 넣으니, 가려져 축소된 박스가 중앙값을 끌어내리고 물체가 다시 보일 때
   정상 박스를 기각했다(실측: 이력 부피가 1020→9cm³로 붕괴한 트랙 존재).
3. **판정이 이진값이라 복구 경로가 없었다.** 기각되면 amodal이 호출조차 안 돼
   전체 프레임의 38%가 A·B 모두 빈 화면이었다.

## V4의 대응

- 크기 판정을 **깊이로 정규화한 물리 크기**(미터)와 **회전불변 대각**으로 옮겼다.
  카메라가 바뀌어도, 물체가 회전해도 같은 기준이 된다.
- **재투영 자기정합**(3D 박스는 그것을 만든 2D 증거와 맞아야 한다)을 추가했다.
- **그린다**와 **이력에 넣는다**를 분리했다. 이력 갱신은 관측 품질이 좋을 때만.
- 트랙을 **confirmed/coasting** 2상태로 두고, coasting 구간은 방식 B가 채운다.
  방식 A의 렌더 조건은 건드리지 않아 보수적 결과가 그대로 남는다.
- 가려짐 판정을 '부피가 줄었다'(잡음)에서 **가림의 물리적 증거**(손 마스크 겹침,
  물체보다 앞선 픽셀 비율)로 교체했다.
"""
import numpy as np
import cv2

from geometry import (Intrinsics, backproject, filter_points, fit_box3d, draw_box3d,
                      align_mask_to_depth, largest_component, cluster_depth,
                      check_reprojection, same_depth_components, mirror_extend_z)
from profiles import Profile, profile_for

# --- 전역 안전판 (라벨·태스크·카메라 무관) -----------------------------------
# 물리 크기 상한은 Profile.max_phys_size(profiles.py, 기본 0.75)에서 읽는다      # m. 후보 2D 박스가 그 깊이에서 가리키는 물리 폭 상한.
                          # 면적비와 달리 거리에 불변이라 head/wrist에 같은 값을 쓴다.
                          # 0.5m로 두면 펼친 수건(실측 0.65~0.71m)이 잘린다. 로봇 팔
                          # 오검출은 이 값이 아니라 아래 DISTRACTOR 겹침 배제가 막는다.
AREA_SAFETY = 0.60        # 화면 점유율 상한. 정체성 판별이 아니라 '화면을 통째로 잡았다'는
                          # 축퇴 상태만 걸러내는 최후 안전판.
MAX_SIZE_GLOBAL = 0.80    # m. 3D 박스 최대 변 (전역 안전판). head 라운드1 기각 0건.
DIST_IOU_MAX = 0.75       # 손/팔 후보와 이만큼 겹치는 target 후보는 배제.
                          # 손에 쥔 물체는 부분적으로만 겹치므로(IoU<0.5) 살아남고,
                          # '팔 전체를 물체로 오인'한 후보만 걸린다. 거리·면적에 무관.
DIAG_UP = 2.0             # 대각이 이력 중앙값의 2배 초과면 기각 (회전불변)
DIAG_LO = 0.5             # 하한은 기각이 아니라 '이력 미갱신'으로만 쓴다 (아래 참조)
RX_BAND = (0.40, 1.80)    # 재투영 가로 스케일 허용 범위
RY_BAND = (0.30, 2.20)    # 세로는 물체가 박스 상하단까지 안 차는 경우가 많아 더 넓게

# 모든 태스크·데이터셋에 공통으로 붙이는 방해물 어휘.
# 로봇 손/사람 손은 Unitree G1 데이터에 항상 존재하는 물리적 상수라 태스크 지식이 아니다.
# 이 트랙이 있어야 (1) 검출기가 손을 target으로 흡수하지 않고 (2) 가림 판정에 손 마스크를 쓴다.
DISTRACTORS = ["robot hand", "robot arm", "gripper", "human hand"]


def iou2d(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def inflate(box, ratio, W, H):
    x1, y1, x2, y2 = box
    w, h = (x2 - x1) * ratio / 2, (y2 - y1) * ratio / 2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [max(0, cx - w), max(0, cy - h), min(W - 1, cx + w), min(H - 1, cy + h)]


def clip_mask_to_box(mask, box, ratio=1.15):
    H, W = mask.shape
    x1, y1, x2, y2 = [int(v) for v in inflate(box, ratio, W, H)]
    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def phys_width(box2d, z, K):
    """2D 박스가 깊이 z에서 가리키는 물리 폭·높이(m). 거리에 불변한 크기 척도."""
    x1, y1, x2, y2 = box2d
    return (x2 - x1) * z / max(K.fx, 1e-6), (y2 - y1) * z / max(K.fy, 1e-6)


class Track:
    """라벨 1개의 시간축 상태.

    핵심 설계: '그린다'(accepted)와 '이력에 넣는다'(update_history)를 분리한다.
    게이트가 자기 이력과 비교하는 구조에서 이력을 무비판적으로 갱신하면
    가려진 박스가 기준을 끌어내려 정상 박스를 기각하는 양의 피드백이 생긴다.
    """
    HIST = 30

    def __init__(self, label, is_target=False, fps=10.0, prof=None):
        self.prof = prof
        self.label = label
        self.is_target = is_target
        self.fps = fps
        self.sizes = []          # 자격을 통과한 관측만 쌓인다
        self.areas = []          # mask_px * depth^2 — 거리 보정된 겉보기 크기
        self.center = None
        self.prev_center = None
        self.box2d = None
        self.depth_med = None
        self.ema_c = self.ema_s = None          # 방식 A 렌더용
        self.ema_c_b = self.ema_s_b = None      # 방식 B 렌더용 (A와 같은 필터)
        self.miss = 0
        self.n_acc = self.n_rej = 0
        self.n_occluded = self.n_coast = 0
        # confirmed: 고임계 det이 짧은 창 안에 2회(또는 det 1회 + 정합 redet 3회) 나와야 True.
        # 이 조건 전에는 저임계 재검출/전파를 쓰지 않는다 -> '물체가 보인 적 없는 손'의 유령 차단.
        self.confirmed = False
        self._det_hits = []      # (frame_idx, center) 최근 고임계 det
        self._redet_hits = []
        self.frames_since_det = 0
        self.det_gaps = []       # det 간격 실측 — 하드컷을 상수로 두지 않기 위해
        self.anchor_box2d = None      # 전파 창의 기준 (프로파일 anchor_prop용)
        self.frames_since_evi = 0
        if prof is not None and getattr(prof, 'size_prior', None) and is_target:
            # 알려진 실제 치수를 이력 시드로 넣는다. 이력이 빈 초반에 크기 게이트가
            # 무력해져 엉뚱한 후보(로봇 팔 등)가 기준이 되는 것을 막는다.
            self.sizes = [np.array(prof.size_prior, dtype=float)] * int(prof.prior_n)

    # ---- 이력 조회
    def med_size(self):
        return np.median(np.array(self.sizes), axis=0) if self.sizes else None

    def med_diag(self):
        ms = self.med_size()
        return float(np.linalg.norm(ms)) if ms is not None else None

    def med_area(self):
        return float(np.median(self.areas)) if self.areas else None

    def det_budget(self):
        """det 없이 버틸 프레임 수. 상수가 아니라 이 트랙이 실측한 det 간격에서 유도한다.
        (라운드1에서 고정 3초를 걸었다면 정상 동작하던 head 9/16 에피소드가 잘렸다)"""
        if len(self.det_gaps) >= 3:
            return max(int(3.0 * float(np.median(self.det_gaps))), int(4.0 * self.fps))
        return int(8.0 * self.fps)

    # ---- 게이트
    def gate(self, b, mask, box2d, K, WH, src):
        """반환 (통과, 사유, 진단dict). 하드 기각은 물리적으로 불가능한 것만."""
        W, H = WH
        diag = {}
        du, dv, rx, ry, inside = check_reprojection(b, box2d, K, 0)
        diag.update(du=round(du, 1), dv=round(dv, 1), rx=round(rx, 3), ry=round(ry, 3))

        barea = max((box2d[2] - box2d[0]) * (box2d[3] - box2d[1]), 1.0)
        diag["area_ratio"] = round(barea / (W * H), 4)
        if barea > (self.prof.area_max if self.prof else AREA_SAFETY) * W * H:
            return False, f"area>{barea/(W*H):.2f}", diag

        if np.max(b.size) > MAX_SIZE_GLOBAL:
            return False, f"absmax>{MAX_SIZE_GLOBAL}", diag

        if not inside:                       # 3D 중심이 자기 2D 증거 밖 = 다른 것을 잡았다
            return False, "reproj", diag

        fill = mask.sum() / barea
        diag["fill"] = round(float(fill), 3)
        if fill > 0.995:                     # 마스크가 박스를 통째로 채움 = 배경 락온
            return False, f"fill={fill:.2f}", diag

        dm = self.med_diag()
        d = float(np.linalg.norm(b.size))
        diag["diag"] = round(d, 4)
        if dm is not None and len(self.sizes) >= 5:
            diag["d_over_dm"] = round(d / max(dm, 1e-9), 3)
            lo = getattr(self.prof, 'shrink_lo', None) if self.prof else None
            if lo and src == "prop" and d < lo * dm:
                return False, "prop_shrink", diag   # 전파 박스 붕괴 = 유령
            hi = getattr(self.prof, 'prop_grow_hi', 0.0) if self.prof else 0.0
            if hi and src == "prop" and d > hi * dm:
                # 전파 창이 인접한 손을 삼켰다. 기각하면 tr.box2d가 갱신되지 않아
                # 다음 프레임 창이 마지막 정상 박스에 머문다 = 복리 성장이 끊긴다.
                return False, "prop_grow", diag
            if d > DIAG_UP * dm:             # 회전불변 상한 (축별 게이트는 회전을 오차로 오인)
                return False, f"diag>{DIAG_UP}x", diag
            # 하한(d < dm/2)은 기각하지 않는다 — 라운드1 실측상 이 조건에 걸리는 프레임은
            # 100%가 가려짐 구간이었다. 기각하면 '옮기는 중 소실'을 더 악화시킨다.
            # 대신 이력 갱신 자격에서만 제외한다(update_ok 참조).
        return True, "", diag

    def update_ok(self, src, occ, diag, n_points, WH):
        cs = getattr(self.prof, 'cold_start_hist', 0) if self.prof else 0
        if cs and src == "det" and len(self.sizes) < cs:
            # 이력이 빈 초반에는 가림 판정이 과민해도 고임계 검출을 기준으로 받는다.
            # (이력이 굶으면 이후 크기 게이트와 방식 B가 함께 무너진다)
            return True
        """이 관측이 '기준'이 될 자격이 있는가. 렌더링과 무관하게 이력 오염만 막는다."""
        if src not in ("det", "redet"):
            return False
        if occ:
            return False
        rx, ry = diag.get("rx", 0), diag.get("ry", 0)
        if not (RX_BAND[0] <= rx <= RX_BAND[1] and RY_BAND[0] <= ry <= RY_BAND[1]):
            return False
        W, H = WH
        if n_points < max(30, 0.00015 * W * H):
            return False
        r = diag.get("d_over_dm")
        if r is not None and not (0.7 <= r <= 1.4):
            return False
        return True

    def accept(self, b, box2d, dmed, area_px, update_history, src, fidx):
        if update_history:
            self.sizes.append(np.array(b.size)); self.sizes = self.sizes[-self.HIST:]
            if area_px is not None:
                self.areas.append(area_px); self.areas = self.areas[-self.HIST:]
            if dmed is not None:
                self.depth_med = dmed
        self.prev_center = self.center
        self.center = np.array(b.center)
        self.box2d = list(box2d)
        a = 0.45
        self.ema_c = self.center if self.ema_c is None else (1 - a) * self.ema_c + a * self.center
        s = np.array(b.size)
        self.ema_s = s if self.ema_s is None else (1 - a) * self.ema_s + a * s
        self.miss = 0
        self.n_acc += 1
        if src == "det":
            if self.frames_since_det > 0:
                self.det_gaps.append(self.frames_since_det)
            self.frames_since_det = 0
        else:
            self.frames_since_det += 1

    def reject(self):
        self.miss += 1
        self.n_rej += 1
        self.frames_since_det += 1
        if self.miss > self.det_budget():
            # 완전 리셋하되 크기 기준 1개는 시드로 남긴다 — 전부 비우면 게이트가
            # 무방비 상태가 되어 틀린 물체가 새 기준으로 자리잡는다.
            seed = self.med_size()
            self.box2d = None; self.center = None; self.prev_center = None
            self.sizes = [seed] if seed is not None else []
            self.areas = []
            self.confirmed = False
            self._det_hits = []; self._redet_hits = []

    def note_det(self, fidx, center3d, src):
        """confirmed 판정용 관측 기록. det 2회 또는 det 1회 + 정합 redet 3회."""
        win = int(1.5 * self.fps)
        if src == "det":
            self._det_hits.append((fidx, np.array(center3d)))
            self._det_hits = [h for h in self._det_hits if fidx - h[0] <= win]
        elif src == "redet":
            self._redet_hits.append((fidx, np.array(center3d)))
            self._redet_hits = [h for h in self._redet_hits if fidx - h[0] <= win]
        if self.confirmed:
            return
        def agree(hits, k):
            if len(hits) < k:
                return False
            c = np.array([h[1] for h in hits[-k:]])
            return float(np.max(np.linalg.norm(c - c.mean(axis=0), axis=1))) <= 0.15
        if agree(self._det_hits, 2):
            self.confirmed = True
        elif len(self._det_hits) >= 1 and agree(self._redet_hits, 3):
            # HE Articulated처럼 에피소드 전체에서 고임계 det이 1회뿐인 경우를 위한 경로.
            self.confirmed = True

    def coast(self, mask_center_3d):
        """방식 B 전용 — 관측이 끊긴 구간을 이력 크기 + 위치 추정으로 채운다."""
        ms = self.med_size()
        if ms is None or self.center is None:
            return None
        if mask_center_3d is not None:
            c = np.array(mask_center_3d)
        elif self.prev_center is not None:
            c = self.center + (self.center - self.prev_center)   # 등속 외삽
        else:
            c = self.center.copy()
        self.n_coast += 1
        return c, ms.copy()

    def amodal(self, b, occ):
        """방식 B — 가려짐이 확인된 동안 크기를 이력값으로 되돌린다.
        판정(occ)은 밖에서 물리적 증거로 내리고, 여기서는 복원만 한다."""
        cen, cur = np.array(b.center), np.array(b.size)
        ms = self.med_size()
        if not occ or ms is None or len(self.sizes) < 5:
            return cen, cur
        size = ms.copy()
        vmin, vmax = cen - cur / 2, cen + cur / 2
        new_c = cen.copy()
        for i in range(3):                    # 보이는 조각이 박스 안에 들어오도록 최소 이동
            lo, hi = new_c[i] - size[i] / 2, new_c[i] + size[i] / 2
            if vmin[i] < lo:
                new_c[i] -= (lo - vmin[i])
            elif vmax[i] > hi:
                new_c[i] += (vmax[i] - hi)
        return new_c, size

    def ema_b(self, c, s):
        a = 0.45
        self.ema_c_b = c if self.ema_c_b is None else (1 - a) * self.ema_c_b + a * c
        self.ema_s_b = s if self.ema_s_b is None else (1 - a) * self.ema_s_b + a * s
        return self.ema_c_b, self.ema_s_b


class EpisodeTracker:
    """에피소드 1개 처리기.
    targets = {매칭키: 표시라벨} 또는 {매칭키: (표시라벨, is_target)}
    프롬프트에는 DISTRACTORS가 자동으로 붙고, 그 트랙은 가림 판정에 쓰인다.
    """

    def __init__(self, targets, det, seg, prompt, low_thr=0.20, fps=10.0, profile=None):
        # 태스크별 프로파일. r2를 뼈대로 하되 일부 태스크만 r3/r4 설정을 쓴다.
        self.prof = profile or Profile()
        # 여러 매칭키가 같은 표시라벨을 가리키면(rose/flower/bouquet) 한 트랙으로 묶는다.
        # 묶지 않으면 같은 물체에 박스가 겹쳐 그려지고 통계도 중복된다.
        self.targets, self.tracks, self.key_alias = {}, {}, {}
        for k, v in targets.items():
            label, is_tgt = (v[0], bool(v[1])) if isinstance(v, (list, tuple)) else (v, False)
            prim = next((kk for kk, ll in self.targets.items() if ll == label), None)
            if prim is None:
                self.targets[k] = label
                self.tracks[k] = Track(label, is_target=is_tgt, fps=fps, prof=self.prof)
                self.key_alias[k] = k
            else:
                self.key_alias[k] = prim
        # 방해물 트랙 (렌더링하지 않음. 가림 판정과 검출 분리용)
        self.distractor_keys = []
        for d in (DISTRACTORS if self.prof.use_distractors else []):
            k = d.split()[-1] if d != "robot arm" else "arm"
            if k in self.tracks or k in self.key_alias:
                continue
            self.key_alias[k] = k
            self.distractor_keys.append(k)
            self.tracks[k] = Track(d, is_target=False, fps=fps, prof=self.prof)
        base = prompt.rstrip().rstrip(".").strip()
        self.prompt = (" . ".join([base] + DISTRACTORS) + " ."
                       if self.prof.use_distractors else base + " .")
        self.det, self.seg = det, seg
        self.low_thr = low_thr
        self.fps = fps
        self.fidx = -1
        self._prev_dist = {}     # grip_follow: 직전 프레임의 방해물(손) 박스
        self._win = {}           # grip_follow: 검출 공백 동안 이동시킨 전파 창

    def _match(self, key, boxes, phrases, scores, WH, K, depth, dscale, dist_boxes=()):
        """후보 중 최적 1개. 배제 기준은 화면 면적비가 아니라 **깊이 정규화 물리 크기**다.
        면적비는 카메라 거리에 의존해 head/wrist에서 100배 차이 나지만, 물리 크기는
        거리에 불변이라 같은 상한을 모든 카메라에 쓸 수 있다."""
        tr = self.tracks[key]
        W, H = WH
        aliases = [k for k, prim in self.key_alias.items() if prim == key] or [key]
        best, bs, rej = None, -1, None
        for b, p, s in zip(boxes, phrases, scores):
            if not any(a in str(p) for a in aliases):
                continue
            barea = (b[2] - b[0]) * (b[3] - b[1])
            if barea > self.prof.area_max * W * H:
                rej = "area"; continue
            if tr.is_target and dist_boxes:
                if max((iou2d(b, db) for db in dist_boxes), default=0.0) > DIST_IOU_MAX:
                    rej = "overlaps_hand"; continue
            if tr.is_target:
                x1, y1, x2, y2 = [int(max(0, v)) for v in b]
                x2, y2 = min(W, x2), min(H, y2)
                if x2 > x1 and y2 > y1:
                    d = depth[y1:y2, x1:x2]
                    d = d[np.isfinite(d) & (d > 0)]
                    md = self.prof.max_depth
                    if md and d.size > 20:
                        # 근거리 20분위로 판정한다. 대상이 화면을 벗어났을 때 배경의
                        # 비슷한 색 물체로 트랙이 옮겨 붙는 것을 막는다.
                        if float(np.percentile(d, 20)) * dscale > md:
                            rej = "too_far"; continue
                    if d.size > 20:
                        z = float(np.median(d)) * dscale
                        pw, ph = phys_width(b, z, K)
                        if max(pw, ph) > self.prof.max_phys_size:
                            rej = f"phys>{max(pw,ph):.2f}m"; continue
            bonus = self.prof.iou_bonus * iou2d(b, tr.box2d) if tr.box2d is not None else 0.0
            if s + bonus > bs:
                bs, best = s + bonus, (list(map(float, b)), float(s))
        return best, rej

    def _grip_shift(self, key, ref_box, cur_dist):
        """잡고 있는 그리퍼의 2D 이동량을 구한다.

        검출이 끊긴 이동 구간에서 전파 창이 집는 지점에 얼어붙으면 로봇팔을 물체로
        잡게 된다. 매 프레임 검출되는 손 박스의 변위만큼 창을 함께 옮겨 이를 막는다.
        반환: (dx, dy) 또는 None(잡고 있다고 볼 근거가 없음)
        """
        thr = self.prof.grip_follow
        if not thr or ref_box is None or not cur_dist:
            return None
        bx1, by1, bx2, by2 = ref_box
        barea = max((bx2 - bx1) * (by2 - by1), 1.0)
        best_k, best_ov = None, 0.0
        for dk, dbox in cur_dist.items():
            ix1, iy1 = max(bx1, dbox[0]), max(by1, dbox[1])
            ix2, iy2 = min(bx2, dbox[2]), min(by2, dbox[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            ov = (ix2 - ix1) * (iy2 - iy1) / barea      # 창이 손에 얼마나 덮였는가
            if ov > best_ov:
                best_ov, best_k = ov, dk
        if best_k is None or best_ov < thr:
            return None
        prev = self._prev_dist.get(best_k)
        if prev is None:
            return None
        cur = cur_dist[best_k]
        return ((cur[0] + cur[2]) / 2 - (prev[0] + prev[2]) / 2,
                (cur[1] + cur[3]) / 2 - (prev[1] + prev[3]) / 2)

    def step(self, img, depth, K, dscale, align_dx=0):
        self.fidx += 1
        H, W = img.shape[:2]
        boxes, phrases, scores = self.det(img, self.prompt)
        results, pend, missing = [], {}, []
        # 손/팔을 먼저 매칭해 그 영역을 알고 나서 target 후보를 고른다
        dist_boxes = []
        for key in self.distractor_keys:
            m, rej = self._match(key, boxes, phrases, scores, (W, H), K, depth, dscale)
            if m is not None:
                pend[key] = (m[0], "det", m[1]); dist_boxes.append(m[0])
            else:
                missing.append((key, rej))
        for key in self.targets:
            m, rej = self._match(key, boxes, phrases, scores, (W, H), K, depth, dscale, dist_boxes)
            if m is not None:
                pend[key] = (m[0], "det", m[1])
            else:
                missing.append((key, rej))

        # 저임계 재검출 — confirmed 트랙에만 허용 (유령 차단)
        need_low = [k for k, _ in missing
                    if self.tracks[k].confirmed and self.tracks[k].box2d is not None]
        if need_low:
            old = self.det.box_thr
            self.det.box_thr = self.low_thr
            b2, p2, s2 = self.det(img, self.prompt)
            self.det.box_thr = old
            for key in need_low:
                tr = self.tracks[key]
                cand, bi, cs = None, 0.15, 0.0
                for b, p, s in zip(b2, p2, s2):
                    if key not in str(p):
                        continue
                    v = iou2d(b, tr.box2d)
                    if v > bi:
                        bi, cand, cs = v, list(map(float, b)), float(s)
                if cand is not None:
                    pend[key] = (cand, "redet", cs)
                    missing = [(k, r) for k, r in missing if k != key]

        # 방해물(손) 박스 — grip_follow가 켜진 태스크에서만 쓴다
        cur_dist = {k: pend[k][0] for k in self.distractor_keys if k in pend}

        # 마스크 전파 — confirmed 트랙에만
        for key, _ in list(missing):
            tr = self.tracks[key]
            if not (tr.confirmed and tr.box2d is not None):
                continue
            base = self._win.get(key) or tr.box2d
            if self.prof.grip_follow and tr.is_target:
                sh = self._grip_shift(key, base, cur_dist)
                if sh is not None:
                    base = [base[0] + sh[0], base[1] + sh[1],
                            base[2] + sh[0], base[3] + sh[1]]
                    self._win[key] = base       # 공백이 이어지는 동안 누적 이동
            pend[key] = (inflate(base, 1.12, W, H), "prop", 0.0)
            missing = [(k, r) for k, r in missing if k != key]

        if self.prof.exclusive_tracks and len(pend) > 1:
            # 두 target 후보가 사실상 같은 박스를 가리키면 검출 점수가 높은 쪽만 남긴다.
            # target이 시야를 벗어났을 때 옆의 다른 물체(꽃 -> 꽃병)로 옮겨 붙는 것을 막는다.
            ranked = sorted(pend.items(), key=lambda kv: -float(kv[1][2]))
            kept = []
            for k_, v_ in ranked:
                if any(iou2d(v_[0], pend[k2][0]) > 0.85 for k2 in kept):
                    pend.pop(k_, None)
                else:
                    kept.append(k_)

        masks_by_key = {}
        if pend:
            keys = list(pend.keys())
            bxs = np.array([pend[k][0] for k in keys], dtype=np.float32)
            masks = self.seg(img, bxs)
            for k, m in zip(keys, masks):
                masks_by_key[k] = m.astype(bool)

        # 방해물(손/팔) 마스크 합집합 — 가림 판정에 쓴다
        occ_mask = np.zeros((H, W), dtype=bool)
        for k in self.distractor_keys:
            if k in masks_by_key:
                occ_mask |= masks_by_key[k]

        for key in self.targets:
            tr = self.tracks[key]
            if key in pend and pend[key][1] in ("det", "redet"):
                self._win.pop(key, None)      # 실관측이 돌아오면 창 이동 누적을 리셋
            if key not in pend:
                rej = dict(missing).get(key)
                rec = dict(key=key, label=tr.label, src="none", accepted=False,
                           reason=rej or "no_candidate", frame=self.fidx,
                           track_state="confirmed" if tr.confirmed else "unconfirmed")
                cb = tr.coast(None) if tr.confirmed else None
                if cb is not None:
                    c, s = tr.ema_b(*cb)
                    rec.update(accepted_amodal=True, coasting=True,
                               center_amodal=list(map(float, c)), size_amodal=list(map(float, s)),
                               conf=round(float(np.exp(-tr.frames_since_det / 8.0)), 3))
                tr.reject()
                results.append(rec)
                continue

            box2d, src, score = pend[key]
            m = masks_by_key.get(key)
            mk = clip_mask_to_box(m, box2d, 1.15)
            mk = (same_depth_components(mk, depth, dscale, box2d, dz=0.02)
                  if self.prof.components == "same_depth" else largest_component(mk))
            if src == "prop" and mk.sum() > 50:
                ys, xs = np.where(mk)
                box2d = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            _k = int(self.prof.erode_kernel)
            mk_e = cv2.erode(mk.astype(np.uint8), np.ones((_k, _k), np.uint8), 1).astype(bool)
            mk_a = align_mask_to_depth(mk_e, dx=align_dx)
            mask_px = int(mk_a.sum())

            # --- 가림의 물리적 증거 (부피 감소가 아니라 직접 관측) ---
            occ_frac = 0.0
            if mask_px > 0:
                reg = clip_mask_to_box(np.ones((H, W), bool), box2d, 1.1)
                inter = float((occ_mask & reg).sum()) / max(float(reg.sum()), 1.0)
                occ_frac = inter
                if tr.depth_med is not None:
                    dz = depth[reg] * dscale
                    dz = dz[np.isfinite(dz) & (dz > 0)]
                    if dz.size > 20:
                        ms = tr.med_size()
                        # 마진을 상수(3cm)가 아니라 물체 자기 두께의 절반으로 — 두꺼운 물체가
                        # 자기 앞면 때문에 상시 '가려짐'으로 판정되는 것을 막는다.
                        marg = max(0.02, 0.5 * float(ms[2])) if ms is not None else 0.03
                        occ_frac = max(occ_frac, float((dz < tr.depth_med - marg).mean()))

            pts = filter_points(backproject(depth, K, mk_a, depth_scale=dscale,
                                            valid_range=(0.1, 5.0)),
                                percentile=self.prof.filter_pct,
                                foreground_mad=self.prof.filter_mad)
            pts = cluster_depth(pts, gap=0.06)
            b = fit_box3d(pts, pct=1.0) if len(pts) >= self.prof.min_points else None
            size_mir = (mirror_extend_z(b, pts) if (self.prof.mirror_z and b is not None)
                        else None)
            dmed = float(np.median(depth[mk_a])) * dscale if mask_px > 0 else None
            area_px = (mask_px * dmed * dmed) if (dmed and mask_px) else None

            if b is None:
                rec = dict(key=key, label=tr.label, src=src, accepted=False, reason="no_points",
                           frame=self.fidx, box2d=box2d, mask_px=mask_px, det_score=score,
                           occluder_frac=round(occ_frac, 3),
                           track_state="confirmed" if tr.confirmed else "unconfirmed")
                cb = tr.coast(None) if tr.confirmed else None
                if cb is not None:
                    c, s = tr.ema_b(*cb)
                    rec.update(accepted_amodal=True, coasting=True,
                               center_amodal=list(map(float, c)), size_amodal=list(map(float, s)),
                               conf=round(float(np.exp(-tr.frames_since_det / 8.0)), 3))
                tr.reject()
                results.append(rec)
                continue

            if src == "det":
                tr.note_det(self.fidx, b.center, "det")
            elif src == "redet":
                tr.note_det(self.fidx, b.center, "redet")

            ok, why, diag = tr.gate(b, mk_a, box2d, K, (W, H), src)
            # 미확정 트랙은 고임계 det만 승인 — '한 번도 제대로 본 적 없는' 유령을 막는다
            if ok and not tr.confirmed and src != "det":
                ok, why = False, "unconfirmed"

            occ = occ_frac >= 0.25
            if occ:
                tr.n_occluded += 1

            rec = dict(key=key, label=tr.label, src=src, frame=self.fidx, box2d=box2d,
                       det_score=round(float(score), 3), mask_px=mask_px,
                       n_points=int(b.n_points), raw_size=list(map(float, b.size)),
                       dmed=None if dmed is None else round(dmed, 4),
                       occluder_frac=round(occ_frac, 3), occluded=bool(occ),
                       track_state="confirmed" if tr.confirmed else "unconfirmed", **diag)

            if ok:
                upd = tr.update_ok(src, occ, diag, b.n_points, (W, H))
                cB, sB = tr.amodal(b, occ)
                if size_mir is not None:
                    # 단일 시점은 물체 뒷면을 못 보므로 두께가 구조적으로 결손된다.
                    # 방식 B에만 대칭 확장을 적용한다(방식 A는 관측값 그대로).
                    sB = np.maximum(np.asarray(sB, dtype=float), np.asarray(size_mir, dtype=float))
                tr.accept(b, box2d, dmed, area_px, upd, src, self.fidx)
                cBe, sBe = tr.ema_b(cB, sB)
                rec.update(accepted=True, reason="", updated_history=bool(upd),
                           center=list(map(float, tr.ema_c)), size=list(map(float, tr.ema_s)),
                           center_amodal=list(map(float, cBe)), size_amodal=list(map(float, sBe)),
                           accepted_amodal=True, coasting=False, conf=1.0)
            else:
                # 전파 창이 붕괴했거나(prop_shrink) 손을 삼킨(prop_grow) 경우 그 중심은
                # 물체가 아니라 손·유령이다. 방식 B가 그것을 이어받지 않게 한다.
                bad_win = why.startswith("prop_") or why in ("reproj",) or why.startswith("diag>")
                cb = tr.coast(None if bad_win else np.array(b.center)) if tr.confirmed else None
                tr.reject()
                rec.update(accepted=False, reason=why)
                if cb is not None:
                    c, s = tr.ema_b(*cb)
                    rec.update(accepted_amodal=True, coasting=True,
                               center_amodal=list(map(float, c)), size_amodal=list(map(float, s)),
                               conf=round(float(np.exp(-tr.frames_since_det / 8.0)), 3))
            results.append(rec)
        self._prev_dist = cur_dist
        return results

    def draw(self, vis, results, K, mode="A"):
        """mode='A' 관측된 것만 / 'B' coasting·가림보정 포함."""
        from geometry import Box3D
        pal = {"det": (0, 220, 120), "redet": (0, 200, 255), "prop": (255, 160, 40),
               "none": (255, 160, 40)}
        for r in results:
            if mode == "B":
                if not r.get("accepted_amodal"):
                    continue
                c, s = np.array(r["center_amodal"]), np.array(r["size_amodal"])
                conf = float(r.get("conf", 1.0))
                col = (60, 120, 255) if (r.get("coasting") or r.get("occluded")) else pal[r["src"]]
                th = 2 if conf >= 0.6 else 1
                tag = " (추정)" if r.get("coasting") else (" (가림보정)" if r.get("occluded") else "")
            else:
                if not r.get("accepted"):
                    continue
                c, s = np.array(r["center"]), np.array(r["size"])
                col, th, tag = pal[r["src"]], 2, ""
            b = Box3D(center=c, size=s, min_xyz=c - s / 2, max_xyz=c + s / 2,
                      n_points=int(r.get("n_points", 0)))
            draw_box3d(vis, b, K, color=col, thickness=th,
                       label=f"{r['label']} {s[0]*100:.0f}x{s[1]*100:.0f}x{s[2]*100:.0f}cm{tag}")
        return vis

    def stats(self, n_frames):
        out = {}
        for k, tr in self.tracks.items():
            if k in self.distractor_keys:
                continue
            ms = tr.med_size()
            out[tr.label] = dict(
                is_target=tr.is_target, confirmed=tr.confirmed,
                accepted=tr.n_acc, rejected=tr.n_rej,
                coverage=round(tr.n_acc / max(n_frames, 1), 3),
                coasting_frames=tr.n_coast,
                coverage_B=round((tr.n_acc + tr.n_coast) / max(n_frames, 1), 3),
                occluded_frames=tr.n_occluded,
                n_history=len(tr.sizes),
                size_median=None if ms is None else [round(float(x), 4) for x in ms])
        return out
