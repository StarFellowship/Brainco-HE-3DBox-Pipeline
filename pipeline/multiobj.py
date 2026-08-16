"""로봇 부위·추가 참조물·대형 배경의 2차 검출/추적 + COCO RLE + 카테고리 렌더.

## 왜 별도 모듈인가

기존 track3d.EpisodeTracker 의 게이트(물리 크기 0.75m, 화면 점유율 0.60, 3D 최대변
0.80m 등)는 **조작 대상용 값**이라 로봇 팔(길이 1m+, 손목캠에서 화면 대부분 차지)과
테이블(폭 1m+)에는 원리적으로 맞지 않는다. 기존 파일을 절대 수정하지 않는다는 원칙에
따라, 이 모듈은 EpisodeTracker 와 **완전히 분리된 2차 검출 패스**로
robot_part / reference / background 카테고리를 처리한다. 카테고리별 게이트 값은
아래 GATES 에만 존재하며, 조작 대상의 기존 게이트에는 어떤 영향도 없다.

## 게이트 값의 근거

- robot_part : 화면 점유율 ≤0.90, 물리 크기 ≤1.5m, 3D 최대변 ≤1.5m.
  G1 팔 전장 약 1m 미만 + 여유. 손목캠에서 팔이 화면을 거의 다 채우므로 점유율을
  0.90 까지 연다 (조작 대상의 0.60 을 그대로 쓰면 손목캠 팔은 전부 기각된다).
- background : 화면 점유율 ≤0.50, 3D 최대변 ≤2.5m — tasks/6_큰객체/README.md 의
  **실측 확정값**이다 (조합 스윕에서 통과 13건 중 12건 타당). 물리 폭 게이트는
  bigobj2 와 같이 비활성.
- reference  : 기존 파이프라인이 비대상(is_target=False) 트랙에 쓰던 수준
  (점유율 0.60, 3D 최대변 0.80m)을 그대로 옮겼다. 물리 폭 게이트는 기존에도
  target 전용이었으므로 비활성.

## 임포트 구조

torch 불필요. cv2/scipy 가 필요한 geometry 함수는 **함수 안에서 지연 임포트**한다 —
GPU 없는 로컬에서도 이 모듈의 순수 부분(RLE, 좌우 휴리스틱)을 임포트·테스트할 수 있다.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

try:                                    # 서버에는 pycocotools 가 있으면 그것을 쓴다
    from pycocotools import mask as _pycoco
except Exception:
    _pycoco = None


# ================================================================== COCO RLE
# pycocotools.mask.encode 와 동일한 압축 포맷( maskApi.c rleToString / rleFrString 이식).
# pycocotools 미설치 환경 폴백 + 로컬 라운드트립 테스트용. 인코딩은 열 우선(Fortran) 순회,
# 첫 런은 0(배경)의 개수다.

def _runs_fortran(mask):
    """(H,W) bool/uint8 -> 열 우선 런 길이 목록. 첫 런은 0 의 개수(0 일 수 있음)."""
    m = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    if m.size == 0:
        return [0]
    change = np.flatnonzero(m[1:] != m[:-1]) + 1
    idx = np.concatenate(([0], change, [m.size]))
    runs = np.diff(idx).astype(np.int64).tolist()
    if m[0] == 1:                       # 1 로 시작하면 '0 이 0개' 런을 앞에 붙인다
        runs.insert(0, 0)
    return runs


def _rle_string(cnts):
    """런 길이 -> COCO 압축 문자열 (pycocotools rleToString 과 동일한 가변길이 부호화)."""
    out = []
    for i, c in enumerate(cnts):
        x = int(c)
        if i > 2:                       # 3번째 이후는 두 칸 전 값과의 차분을 저장
            x -= int(cnts[i - 2])
        more = True
        while more:
            ch = x & 0x1F
            x >>= 5
            more = (x != -1) if (ch & 0x10) else (x != 0)
            if more:
                ch |= 0x20
            out.append(chr(ch + 48))
    return "".join(out)


def _rle_unstring(s):
    """COCO 압축 문자열 -> 런 길이 목록 (rleFrString 이식)."""
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            ch = ord(s[p]) - 48
            x |= (ch & 0x1F) << (5 * k)
            more = bool(ch & 0x20)
            p += 1
            k += 1
            if not more and (ch & 0x10):
                x |= (-1) << (5 * k)    # 부호 확장
        if len(cnts) > 2:
            x += cnts[len(cnts) - 2]
        cnts.append(int(x))
    return cnts


def mask_to_rle(mask):
    """(H,W) bool -> {"size":[H,W], "counts":str} (COCO 압축 RLE, JSON 저장 가능).
    640x480 기준 객체당 수 KB. pycocotools 가 있으면 그 encode 를, 없으면
    동일 포맷의 순수 파이썬 구현을 쓴다."""
    mask = np.asarray(mask)
    h, w = mask.shape
    if _pycoco is not None:
        enc = _pycoco.encode(np.asfortranarray(mask.astype(np.uint8)))
        counts = enc["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return {"size": [int(h), int(w)], "counts": counts}
    return {"size": [int(h), int(w)], "counts": _rle_string(_runs_fortran(mask))}


def rle_to_mask(rle):
    """{"size","counts"} -> (H,W) bool. 검증·후속 분석용 디코더."""
    h, w = rle["size"]
    cnts = rle["counts"]
    if isinstance(cnts, str):
        cnts = _rle_unstring(cnts)
    m = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in cnts:
        if val:
            m[pos:pos + c] = 1
        pos += c
        val ^= 1
    return m.reshape((h, w), order="F").astype(bool)


# ================================================================== 카테고리 게이트
@dataclass(frozen=True)
class CatGate:
    """카테고리 하나의 검증 게이트. 조작 대상 게이트(track3d/profiles)와 완전 분리."""
    area_max: float                     # 2D 박스 화면 점유율 상한
    max_phys: float                     # 깊이 정규화 물리 폭 상한(m). 0=비활성
    max_size3d: float                   # 3D 박스 최대 변 상한(m)
    min_points: int                     # 3D box 를 만들 최소 점 수
    filter_pct: Tuple[float, float]     # 점군 depth 백분위 클리핑
    filter_mad: float                   # 전경 MAD 분리 강도
    z_range: Tuple[float, float]        # 역투영 유효 깊이 범위(m)


GATES = {
    # 로봇 팔: 길이 1m+, 손목캠에서 화면 대부분 차지 — 조작 대상 값(0.60/0.75/0.80)과 분리
    "robot_part": CatGate(area_max=0.90, max_phys=1.5, max_size3d=1.5,
                          min_points=100, filter_pct=(2.0, 98.0), filter_mad=3.0,
                          z_range=(0.05, 5.0)),
    # 대형 배경: tasks/6_큰객체 실측 확정값 (점유율 0.50 / 최대변 2.5m)
    "background": CatGate(area_max=0.50, max_phys=0.0, max_size3d=2.5,
                          min_points=200, filter_pct=(2.0, 98.0), filter_mad=3.0,
                          z_range=(0.05, 8.0)),
    # 추가 참조물: 기존 비대상 트랙 수준
    "reference": CatGate(area_max=0.60, max_phys=0.0, max_size3d=0.80,
                         min_points=30, filter_pct=(10.0, 90.0), filter_mad=1.5,
                         z_range=(0.10, 5.0)),
}


# ================================================================== 순수 헬퍼
# track3d.iou2d / inflate 와 동일 — 기존 파일 무수정 원칙 + cv2 없는 로컬 임포트를
# 위해 복사했다 (track3d 를 임포트하면 geometry->cv2 의존이 함께 딸려온다).

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


def assign_sides(recs, W):
    """robot_part 레코드에 left/right 부여 — **2D 위치 휴리스틱**이다.

    GroundingDINO 는 'left arm' 류의 좌우 수식어를 신뢰성 있게 구분하지 못하므로
    프롬프트에는 좌우를 넣지 않고(spec_allobj.ROBOT_PARTS 주석), 검출 후 부여한다:
      - 같은 라벨 인스턴스 2개: 2D box 중심 x 가 작은 쪽 = left, 큰 쪽 = right
      - 1개: 화면 중심(W/2) 기준 좌/우
    로봇 자체 카메라(머리·1인칭)가 전방을 볼 때 로봇의 왼팔이 화면 왼쪽에 오므로
    대체로 성립하지만, 손목캠 근접 뷰나 팔이 교차한 자세에서는 실제 키네마틱 좌우와
    다를 수 있다. 키네마틱/URDF 정보를 쓰지 않은 근사임을 보고서에 그대로 기술할 것.
    robot_part 외 카테고리는 side="none".
    """
    by_label = {}
    for r in recs:
        if r.get("category") == "robot_part":
            by_label.setdefault(r["label"], []).append(r)
        else:
            r["side"] = "none"
    for group in by_label.values():
        if len(group) == 1:
            r = group[0]
            cx = (r["box2d"][0] + r["box2d"][2]) / 2.0
            r["side"] = "left" if cx < W / 2.0 else "right"
        else:
            group.sort(key=lambda g: (g["box2d"][0] + g["box2d"][2]) / 2.0)
            for r, side in zip(group, ("left", "right")):
                r["side"] = side
            for r in group[2:]:         # 방어적 — propose 가 라벨당 2개로 제한한다
                r["side"] = "none"
    return recs


# ================================================================== 2차 추적기
class ExtraTracker:
    """robot_part / reference / background 의 프레임 간 검출·전파.

    러너(run_sample10)가 조율하는 3단 흐름:
      1) cands = propose(det 결과)            — 검출 매칭 + 미검출 인스턴스의 전파 후보
      2) 러너가 후보 박스를 SAM 에 일괄 투입    — (기존 트래커 박스와 한 호출로 묶는다)
      3) recs = finalize(cands, masks, ...)   — 3D 산출 + 카테고리 게이트 + 상태 갱신

    src 용어(출력 스키마): "det" = GroundingDINO 가 이 프레임에서 찾음,
    "track" = 직전 박스를 SAM 마스크로 이어 전파 (기존 코드의 "prop" 에 해당).
    전파는 track_budget_sec 까지만 잇는다 — 대형 배경·로봇 팔은 거의 매 프레임
    재검출되므로 짧게 잡아 유령 꼬리(ghost tail)를 원천 차단한다.
    """

    def __init__(self, entries, fps=10.0, track_budget_sec=2.0, min_score=0.30):
        # entries: [(표시라벨, 카테고리, [매칭키...])] — spec_allobj.extra_spec() 산출
        self.entries = list(entries)
        self.fps = fps
        self.budget = max(1, int(track_budget_sec * fps))
        self.min_score = min_score
        self.insts = []                 # [{label, category, box2d|None, miss}]
        self.fidx = -1

    # ------------------------------------------------------------ 1단계
    def propose(self, boxes, phrases, scores, WH):
        """검출 결과에서 카테고리 후보를 뽑는다. 반환: 후보 dict 목록 (SAM 투입 전)."""
        W, H = WH
        self.fidx += 1
        cands = []
        for label, cat, keys in self.entries:
            cap = 2 if cat == "robot_part" else 1      # 팔·그리퍼는 좌우 2개까지
            dets = []
            for b, p, s in zip(boxes, phrases, scores):
                if float(s) < self.min_score:
                    continue
                if not any(k in str(p) for k in keys):
                    continue
                dets.append((list(map(float, b)), float(s)))
            dets.sort(key=lambda x: -x[1])
            kept = []
            for b, s in dets:                          # 중복 박스 제거 (bigobj2 방식)
                if any(iou2d(b, kb) > 0.5 for kb, _ in kept):
                    continue
                kept.append((b, s))
                if len(kept) >= cap:
                    break

            insts = [t for t in self.insts if t["label"] == label]
            used = set()
            for b, s in kept:                          # 검출 -> 인스턴스 (IoU 탐욕 매칭)
                best, bi = None, 0.10
                for t in insts:
                    if id(t) in used or t["box2d"] is None:
                        continue
                    v = iou2d(b, t["box2d"])
                    if v > bi:
                        bi, best = v, t
                if best is None:
                    free = [t for t in insts if id(t) not in used and t["box2d"] is None]
                    if free:
                        best = free[0]
                    elif len(insts) < cap:
                        best = dict(label=label, category=cat, box2d=None, miss=0)
                        self.insts.append(best)
                        insts.append(best)
                    else:                              # 슬롯 초과 — 가장 오래 못 본 것 대체
                        stale = [t for t in insts if id(t) not in used]
                        if not stale:
                            continue
                        best = max(stale, key=lambda t: t["miss"])
                used.add(id(best))
                cands.append(dict(inst=best, label=label, category=cat,
                                  src="det", box2d=b, score=s))
            for t in insts:                            # 미검출 인스턴스 -> 전파 후보
                if id(t) in used or t["box2d"] is None:
                    continue
                if t["miss"] >= self.budget:
                    continue
                cands.append(dict(inst=t, label=label, category=cat, src="track",
                                  box2d=inflate(t["box2d"], 1.12, W, H), score=0.0))
        return cands

    # ------------------------------------------------------------ 3단계
    def finalize(self, cands, masks, depth, K, dscale, align_dx, WH):
        """SAM 마스크로 3D box 산출 + 카테고리 게이트 + 인스턴스 상태 갱신.

        반환 레코드: dict(label, category, side, src, box2d, score, mask(bool HxW),
                          box3d({"center","size"}) 또는 None, reason)
        게이트 미통과 객체도 box2d/mask 는 남기고 box3d=None + reason 을 기록한다
        (출력 스키마 요구 — '검출은 됐으나 3D 는 신뢰 불가'를 구분할 수 있어야 한다).
        """
        from geometry import (backproject, filter_points, fit_box3d,
                              largest_component, align_mask_to_depth)
        W, H = WH
        recs = []
        for c, m in zip(cands, masks):
            g = GATES[c["category"]]
            mask = largest_component(np.asarray(m, dtype=bool))
            box2d = list(c["box2d"])
            if c["src"] == "track" and mask.sum() > 50:
                ys, xs = np.nonzero(mask)              # 전파 박스는 마스크로 다시 조인다
                box2d = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            barea = max((box2d[2] - box2d[0]) * (box2d[3] - box2d[1]), 1.0)
            area_ratio = barea / float(W * H)

            box3d, reason = None, ""
            if mask.sum() < 20:
                reason = "no_mask"
            elif area_ratio > g.area_max:
                reason = f"area>{area_ratio:.2f}"
            else:
                if g.max_phys:                         # 물리 폭 게이트 (robot_part 만 활성)
                    d = depth[mask]
                    d = d[np.isfinite(d) & (d > 0)]
                    if d.size > 20:
                        z = float(np.median(d)) * dscale
                        pw = (box2d[2] - box2d[0]) * z / max(K.fx, 1e-6)
                        ph = (box2d[3] - box2d[1]) * z / max(K.fy, 1e-6)
                        if max(pw, ph) > g.max_phys:
                            reason = f"phys>{max(pw, ph):.2f}m"
                if not reason:
                    mk = align_mask_to_depth(mask, dx=align_dx)
                    pts = filter_points(
                        backproject(depth, K, mk, depth_scale=dscale, valid_range=g.z_range),
                        percentile=g.filter_pct, foreground_mad=g.filter_mad)
                    if len(pts) < g.min_points:
                        reason = "no_points"
                    else:
                        b = fit_box3d(pts, pct=1.0)
                        if b is None:
                            reason = "no_points"
                        elif float(np.max(b.size)) > g.max_size3d:
                            reason = f"absmax>{g.max_size3d}"
                        else:
                            box3d = dict(center=[float(v) for v in b.center],
                                         size=[float(v) for v in b.size])

            t = c["inst"]                              # ---- 상태 갱신
            if c["src"] == "det":
                t["box2d"] = list(box2d)
                t["miss"] = 0
            else:
                t["miss"] += 1
                if mask.sum() >= 20:                   # 유효 마스크일 때만 창 갱신
                    t["box2d"] = list(box2d)           # (자기 출력 복리 성장은 budget 이 캡)
                if t["miss"] >= self.budget:
                    t["box2d"] = None                  # 전파 종료 — 재검출까지 휴면
                    t["miss"] = 0
            recs.append(dict(label=c["label"], category=c["category"], side="none",
                             src=c["src"], box2d=[float(v) for v in box2d],
                             score=float(c["score"]), mask=mask, box3d=box3d,
                             reason=reason))
        assign_sides(recs, W)
        return recs


# ================================================================== 렌더
# 카테고리 색 (BGR): 조작 대상=초록, robot_part=파랑, reference=노랑, background=회색
CAT_COLORS = {
    "target": (0, 220, 120),
    "robot_part": (255, 130, 60),
    "reference": (0, 220, 240),
    "background": (160, 160, 160),
}


def draw_objects(vis, objects, K):
    """프레임의 객체 목록(마스크 포함)을 카테고리 색으로 오버레이한다.
    2D box + mask + (있으면) 3D box + 라벨 텍스트. geometry.draw_box3d 재사용."""
    import cv2
    from geometry import Box3D, draw_box3d
    # 마스크 오버레이를 먼저 전부 깔고 박스·텍스트를 위에 그린다
    for o in objects:
        m = o.get("mask")
        if m is None or not m.any():
            continue
        col = CAT_COLORS.get(o["category"], (200, 200, 200))
        ov = vis.copy()
        ov[m] = col
        cv2.addWeighted(ov, 0.35, vis, 0.65, 0, dst=vis)
    for o in objects:
        col = CAT_COLORS.get(o["category"], (200, 200, 200))
        x1, y1, x2, y2 = [int(v) for v in o["box2d"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        name = o["label"] if o.get("side") in (None, "", "none") \
            else f"{o['side']} {o['label']}"
        tag = f"{name} [{o['src']}]"
        b3 = o.get("box3d")
        if b3 is not None:
            c = np.array(b3["center"], dtype=float)
            s = np.array(b3["size"], dtype=float)
            box = Box3D(center=c, size=s, min_xyz=c - s / 2, max_xyz=c + s / 2, n_points=0)
            draw_box3d(vis, box, K, color=col, thickness=2,
                       label=f"{tag} {s[0]*100:.0f}x{s[1]*100:.0f}x{s[2]*100:.0f}cm")
        else:
            cv2.putText(vis, tag, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    return vis
