"""태스크별 처리 프로파일 — r2를 뼈대로, 일부 태스크만 r3/r4 설정을 쓴다.

## 왜 이런 구조인가

라운드 4번(r1~r4)을 돌리며 개선했는데, 육안 검증 결과 **전체적으로는 r2가 가장 좋았다**.
r3에서 얇은 부속물을 살리려 점군 필터를 완화했더니 잘 되던 태스크가 무너졌다
(GraspOreo/머리좌 100%→67%, PickCharger 6×6×2→8×28×30cm).

그러나 **일부 태스크는 r3/r4가 확실히 나았다** — PickDrink(0%→100%),
HE deformable·Locomanip·Tool_use(유령 제거), HRI·Precision(꽃 줄기 포함).

전역 파라미터 하나로 두 요구를 동시에 만족할 수 없으므로, **태스크별로 프로파일을 지정**한다.
이것이 "문제 있는 태스크만 건드리고 나머지는 손대지 않는다"를 코드로 보장하는 방법이다.

## 사용법

    from profiles import profile_for
    prof = profile_for("GraspOreo")        # -> R2 프로파일
    prof = profile_for("PickDrink")        # -> R3 프로파일
"""
from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class Profile:
    """한 태스크를 처리할 때의 알고리즘 설정."""

    # --- 점군 정제 (크기 정확도에 가장 큰 영향) ---
    filter_pct: Tuple[float, float] = (10.0, 90.0)
    """depth 백분위 클리핑. 완화하면(2,98) 얇은 부속물이 살지만 배경·로봇팔이 들어와
    박스가 부푼다(실측: charger 6×6×2 → 8×28×30cm)."""

    filter_mad: float = 1.5
    """전경 분리 강도. 위와 같은 트레이드오프."""

    # --- 마스크 처리 ---
    components: str = "largest"
    """'largest' = 가장 큰 연결 덩어리만 (r2). 케이블·그림자를 확실히 배제하지만
    얇은 부속물(토끼 귀·꽃 줄기)도 잘린다.
    'same_depth' = 깊이가 비슷한 성분의 합집합 (r3). 부속물을 살리되 손이 섞일 위험."""

    erode_kernel: int = 5
    """마스크 침식 커널. 얇고 긴 물체(치약)에서는 3이 적합하다."""

    # --- 검출·추적 ---
    use_distractors: bool = True
    """프롬프트에 'robot hand/arm/gripper' 등을 추가할지 (r2 방식).
    로봇 팔 오검출을 막지만 프롬프트가 길어져 target 검출이 밀릴 수 있다
    (r2에서 5개 유닛이 이것 때문에 0%가 됐다)."""

    anchor_prop: bool = False
    """마스크 전파의 탐색창을 마지막 실검출에 고정할지 (r3 이후).
    자기 출력으로 창을 갱신하면 복리로 붕괴해 유령이 된다."""

    prop_limit_sec: float = 0.0
    """실검출 없이 전파를 이어갈 최대 시간(초). 0이면 무제한(r2/r3).
    r4는 1.0초로 제한했는데 유령은 사라졌지만 정상 구간도 끊겼다."""

    iou_bonus: float = 0.3
    """후보 선정 시 직전 위치와 겹치는 정도에 주는 가중. 높이면 추적이 이어지지만
    엉뚱한 물체에 락온되면 그것도 유지된다."""

    # --- 3D box ---
    mirror_z: bool = False
    """방식 B에서 깊이 방향 두께를 대칭 확장할지 (r3 이후). 단일 시점의
    구조적 두께 결손을 보정하지만 기울어진 물체에서 과대해질 수 있다."""

    min_points: int = 10
    """3D box를 만들 최소 점 수. 얇은 물체에서는 낮춰야 한다."""

    # --- 태스크 고유 ---
    max_phys_size: float = 0.75
    """target 후보의 물리 폭 상한(m). 화면을 채우는 근접 촬영에서도 유효하도록
    거리로 정규화된 값이다."""

    area_max: float = 0.60
    """2D 박스가 화면에서 차지할 수 있는 최대 비율. 근접 촬영(손목 카메라)에서는
    정상 물체도 화면을 채우므로 태스크별로 올려야 한다.
    실측: PickApple 집는 쪽 손목에서 사과의 화면 점유율이 최대 0.627이라
    0.60 임계에 1~3%p 차이로 걸려 56프레임이 통째로 죽었다."""

    size_prior: tuple = ()
    """물체의 알려진 실제 치수(m). 이력이 비어 있는 초반에 크기 게이트가 무력해지는
    것을 막는 시드다. 얇은 물체나 오검출이 잦은 태스크에 쓴다."""

    prior_n: int = 5
    """size_prior를 이력에 몇 개 넣을지."""

    shrink_lo: float = 0.0
    """전파 프레임에서 박스가 이력 대각의 이 비율보다 작아지면 기각(0=비활성).
    전파가 자기를 먹으며 붕괴하는 유령의 주 형태를 막는다."""

    prop_grow_hi: float = 0.0
    """전파 프레임에서 박스 대각이 이력 대각의 이 배수를 넘으면 기각(0=비활성).
    shrink_lo(붕괴 방지)의 반대 방향이다. 전파 창은 자기 출력을 1.12배씩 다시 부풀리므로,
    물체가 로봇 손 안에 있으면 창이 손을 삼키며 복리로 커진다. 전역 상한(2.0배)은
    작은 물체에 너무 헐거워 이를 못 잡는다(실측: charger 6x5x2 -> 10x8x7cm이 1.73배)."""

    max_depth: float = 0.0
    """target 후보의 카메라 거리 상한(m, 0=비활성). 대상이 화면 밖으로 나갔을 때
    배경(벽·천장)의 비슷한 색 물체로 트랙이 옮겨 붙는 것을 막는다.
    max_phys_size가 '크기'를 보는 것과 달리 이것은 '거리'를 본다."""

    grip_follow: float = 0.0
    """잡혀 있는 target의 전파 창을 그리퍼의 2D 이동량만큼 함께 옮긴다(0=비활성).
    값은 '집혀 있다'로 볼 최소 포함율. 검출이 끊긴 이동 구간에서 창이 집는 지점에
    얼어붙어 로봇팔을 잡는 것을 막는다."""

    cold_start_hist: int = 0
    """이력이 빈 초반에 고임계 검출 관측을 가림 여부와 무관하게 이력에 넣을 개수(0=비활성).
    가림 판정이 과민할 때 이력이 굶어 방식 B의 크기가 작아지는 것을 막는다."""

    exclusive_tracks: bool = False
    """한 프레임에서 두 target이 같은 2D 박스를 쓰지 못하게 한다.
    'target이 사라지면 비슷한 다른 물체로 옮겨 붙는' 실패를 막는다
    (HE Precision에서 장미가 화면을 벗어나자 꽃병을 장미로 잡았다)."""


# ---------------------------------------------------------------- 기준 프로파일
R2 = Profile()                                   # 뼈대. 전체적으로 가장 좋았다.

R3 = replace(R2,
             filter_pct=(2.0, 98.0), filter_mad=3.0,
             components="same_depth", use_distractors=False,
             anchor_prop=True, mirror_z=True)

R4 = replace(R3,
             filter_pct=(10.0, 90.0), filter_mad=1.5,   # r3의 완화를 되돌린 것
             prop_limit_sec=1.0, iou_bonus=0.45)


# ------------------------------------------------- 태스크별 지정 (육안 검증 결과)
# 사용자가 라운드별 산출물을 직접 보고 확정한 매핑이다. 근거를 주석으로 남긴다.
TASK_PROFILE = {
    # ---- Brainco ----
    "GraspOreo":       ("R2", "head/wrist, A/B 모두 완벽 — 건드리지 않는다"),
    "GraspRubiksCube": ("R2", "완벽 — 건드리지 않는다"),
    "PickApple":       ("R2", "head 완벽. 집는 쪽 wrist에서 사과가 화면을 채울 때만 보완"),
    "PickCharger":     ("R2", "로봇팔 오인이 남아 있어 별도 보완 필요"),
    "PickDoll":        ("R3", "r3의 head 결과가 가장 좋았다. wrist는 검출 불가로 판단"),
    "PickDrink":       ("R3", "r3에서 물병 검출 0%→100%. head/wrist 모두 양호"),
    "PickTissues":     ("R3", "r3의 head 결과 채택"),
    "PickToothpaste":  ("R2", "전 라운드 실패. 얇은 물체 전용 보완 필요"),
    # ---- Humanoid Everyday ----
    "Articulated":     ("R2", "전반 양호. 노트북이 닫힌 뒤 박스가 작아지는 것만 보완"),
    "Basic":           ("R2", "pink toy 이동 구간 보완 필요"),
    "deformable":      ("R3", "r3 채택"),
    "HRI":             ("R4", "r4 채택"),
    "Locomanip":       ("R3", "r3 채택"),
    "Precision":       ("R4", "r4 채택. 장미가 화면을 벗어나면 꽃병을 장미로 잡는 것만 보완"),
    "Tool_use":        ("R3", "r3 채택"),
}

_BASE = {"R2": R2, "R3": R3, "R4": R4}

# 태스크 고유 보완 — 위 기준선 위에 덧붙인다. 다른 태스크에는 영향이 없다.
TASK_OVERRIDE = {
    # 집는 쪽 손목에서 사과가 화면을 채울 때(점유율 최대 0.627) 면적 상한 0.60에
    # 걸려 56프레임이 죽었다. 기각분의 검출 점수 0.827·재투영 정합 모두 정상이었다.
    "PickApple": dict(area_max=0.75),

    # 로봇 전완이 "white charger"로 검출되고 점수가 진짜 charger와 겹친다(0.388 vs 0.369).
    # 후보가 하나라도 잡히면 재검출·전파 경로가 열리지 않아 팔이 슬롯을 빼앗는다.
    # charger는 손에 들어가는 작은 물체이므로 물리 크기 상한으로 팔을 배제한다.
    # 추가(r5 진단): 접시에 내려놓는 후반 48프레임에서 실검출이 끊기고 전파 창이
    # 자기 출력을 1.12배씩 부풀려 그리퍼를 삼켰다(6x5x2 -> 10x8x7cm). 전역 상한 2.0배는
    # 이 크기의 물체에 너무 헐거워(최대 1.73배) 못 잡았다.
    "PickCharger": dict(max_phys_size=0.15, prop_grow_hi=1.35, shrink_lo=0.55,
                        size_prior=(0.06, 0.05, 0.02), prior_n=5),

    # 얇고 긴 물체(15x4x3cm)라 5x5 침식이 마스크를 지운다. 실제 치수를 이력 시드로 주고,
    # 전파가 붕괴하면 기각한다.
    "PickToothpaste": dict(erode_kernel=3, min_points=8,
                           size_prior=(0.15, 0.04, 0.03), prior_n=5,
                           shrink_lo=0.45, grip_follow=0.5),

    # 노트북이 닫힌 뒤 박스가 작아진다 — 점군 필터만 완화해 얇아진 형상을 살린다.
    # (components는 r2의 'largest' 유지 — r3 전체를 가져오지 않는다)
    "Articulated": dict(filter_pct=(2.0, 98.0), filter_mad=3.0),

    # pink toy 이동 구간 보완. 이력이 얕을 때 시드를 확보한다.
    # pink toy가 그리퍼에 들려 옮겨질 때, 검출기가 접시 하나를 'toy'와 'plate' 양쪽
    # phrase로 내주고 toy에도 접시 박스가 배정된다(IoU=1.000). 후보가 하나라도 잡히면
    # 재검출·전파 경로가 열리지 않아 17프레임이 통째로 죽는다.
    # toy는 손에 들어가는 10cm 물체이므로 물리 크기로 접시를 후보 단계에서 배제한다
    # (실측: 정상 승인 최대 0.159m vs 접시 오배정 0.335~0.384m).
    "Basic": dict(cold_start_hist=5, max_phys_size=0.25, shrink_lo=0.45),

    # 장미가 시야를 벗어나면 꽃병으로 옮겨 붙는다 — 트랙 간 박스 독점으로 막는다.
    "Precision": dict(exclusive_tracks=True),

    # 꽃을 건넨 뒤 꽃봉오리가 화면 위로 벗어나면 2.2~2.7m 벽면의 분홍 물체를 장미로
    # 고임계 검출(0.49~0.58)해 152프레임 중 37건이 엉뚱한 곳에 그려졌다.
    # 책상 위 근접 조작이라 대상이 1m를 넘을 수 없다(정상 구간 실측 0.29~0.60m).
    "HRI": dict(max_depth=1.2),

    # 이동 구간에서 검출이 전혀 나오지 않아(전 라운드 공통) 전파 창이 집는 지점에
    # 얼어붙어 로봇팔을 잡는다. 매 프레임 검출되는 그리퍼의 변위만큼 창을 함께 옮긴다.
}


def profile_for(task: str) -> Profile:
    """태스크 이름(Brainco 태스크명 또는 HE 카테고리명)으로 프로파일을 얻는다."""
    base_name, _ = TASK_PROFILE.get(task, ("R2", "미지정 — 기본값"))
    prof = _BASE[base_name]
    ov = TASK_OVERRIDE.get(task)
    return replace(prof, **ov) if ov else prof


def describe(task: str) -> str:
    base_name, reason = TASK_PROFILE.get(task, ("R2", "미지정"))
    ov = TASK_OVERRIDE.get(task)
    s = f"{task}: {base_name} — {reason}"
    if ov:
        s += f" (+보완 {ov})"
    return s


if __name__ == "__main__":
    for t in TASK_PROFILE:
        print(describe(t))
