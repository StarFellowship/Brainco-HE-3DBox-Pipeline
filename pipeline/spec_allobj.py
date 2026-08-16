"""모드 B(모든 객체 검출) 사양 — 태스크별 4갈래 카테고리 프롬프트.

tasks/6_큰객체/prompt-design.md 의 4갈래 설계를 15개 태스크(Brainco 8 + HE 7) 전부에
정적 데이터로 구현한 것이다:
  (a) 조작 대상(target)   : spec.py 의 is_target=True — **여기서 새로 정의하지 않고
                            spec.py 에서 자동 유도한다** (이중 정의로 인한 drift 방지)
  (b) 참조 물체(reference): 대상과 상호작용하는 이동 가능 물체 (plate, vase, basket ...)
  (c) 대형 배경(background): 태스크가 벌어지는 0.8m+ 지지면·가구 (table, desk, shelf ...)
                            → bigobj2 실측 확정 게이트(점유율 0.50 / 3D 최대변 2.5m) 적용
  (d) 로봇 부위(robot_parts): robot arm / robot gripper — 아래 ROBOT_PARTS 상수

설계 규칙 (prompt-design.md §1.4 를 따른다):
  - phrase 는 소문자 영어 명사구, 관사 없음, " . " 구분 (GroundingDINO 규격)
  - 특정 변형(색·재질) + 상위개념을 병기해 검출기 어휘 편차를 흡수한다
  - **로봇 부위 프롬프트에 left/right 를 넣지 않는다** — GroundingDINO 는 좌우 수식어를
    신뢰성 있게 구분하지 못한다. 좌/우는 검출 후 2D 위치 휴리스틱으로 부여한다
    (multiobj.assign_sides 참조).
  - chair 는 넣지 않는다 — bigobj2 검증에서 유일하게 육안 탈락한 건이
    '로봇 그리퍼를 chair 로 오검출'이었다 (tasks/6_큰객체/README.md §2).
  - HRI 의 human hand 는 4갈래 스키마(target|robot_part|reference|background)에 자리가
    없어 이번 확장 범위에서 제외했다 (사람 신체는 robot_part 가 아니다).

사용처: run_sample10.py 가 extra_spec() 으로 2차 검출 패스의 프롬프트·매칭 사양을 얻는다.
조작 대상과 spec.py 의 기존 참조물은 기존 EpisodeTracker 가 그대로 처리하므로
extra_spec() 결과에 포함되지 않는다 (이중 추적 방지).
"""
from spec import BRAINCO, HE_REP

# ------------------------------------------------------------------ (d) 로봇 부위
# 전 태스크 공통. 좌/우는 프롬프트가 아니라 검출 후 휴리스틱으로 부여한다 (위 주석 참조).
ROBOT_PARTS = [
    ("robot arm", ["robot arm"]),
    ("robot gripper", ["robot gripper"]),
]

# ------------------------------------------------------------------ (b)(c) 태스크별
# 형식: {태스크: {"reference": [(표시라벨, [phrase...])], "background": [...]}}
# Brainco 8태스크는 같은 실험대(흰 테이블 + 접시) 구성이라 배경이 동일하다.
_BC_BACKGROUND = [("table", ["white table", "table", "desk"])]
_BC_REFERENCE = [("plate", ["plate"])]          # spec.py 에 이미 있어 extra 에서는 걸러진다

ALLOBJ = {
    # ---- Brainco (탁상 조작, 머리 2캠 + 손목 2캠) ----
    "GraspOreo":       dict(reference=list(_BC_REFERENCE), background=list(_BC_BACKGROUND)),
    "GraspRubiksCube": dict(reference=list(_BC_REFERENCE), background=list(_BC_BACKGROUND)),
    "PickApple":       dict(reference=list(_BC_REFERENCE), background=list(_BC_BACKGROUND)),
    "PickCharger":     dict(reference=list(_BC_REFERENCE), background=list(_BC_BACKGROUND)),
    "PickDoll":        dict(reference=list(_BC_REFERENCE), background=list(_BC_BACKGROUND)),
    "PickDrink":       dict(reference=list(_BC_REFERENCE), background=list(_BC_BACKGROUND)),
    "PickTissues":     dict(reference=list(_BC_REFERENCE), background=list(_BC_BACKGROUND)),
    "PickToothpaste":  dict(reference=list(_BC_REFERENCE), background=list(_BC_BACKGROUND)),

    # ---- Humanoid Everyday (1인칭 1캠, 카테고리 대표 태스크 기준 장면 구성) ----
    # Basic: 분홍 인형을 주황 접시에 — 장면에 바구니 존재 (bigobj2 he_basic 실측)
    "Basic": dict(
        reference=[("orange plate", ["orange plate"]), ("basket", ["basket"])],
        background=[("table", ["table", "desk"])]),
    # Articulated: 노트북 닫기 — 책상 위
    "Articulated": dict(
        reference=[],
        background=[("desk", ["desk", "table"])]),
    # deformable: 수건 접기 — 책상 위
    "deformable": dict(
        reference=[],
        background=[("desk", ["desk", "table"])]),
    # HRI: 장미 건네기 — 테이블, 장면에 꽃병 존재 (bigobj2 he_hri 실측)
    "HRI": dict(
        reference=[("vase", ["vase", "flower vase"])],
        background=[("table", ["table", "desk"])]),
    # Locomanip: 걸어가 병을 집어 용기에 — 이동 장면이라 가구가 여럿 (bigobj2 he_locomanip)
    "Locomanip": dict(
        reference=[("container", ["container box"])],
        background=[("desk", ["desk", "table"]), ("shelf", ["shelf"]),
                    ("cabinet", ["cabinet"]), ("door", ["door"])]),
    # Precision: 장미를 분홍 꽃병에 — 꽃병은 spec.py 참조물로 이미 추적된다
    "Precision": dict(
        reference=[("vase", ["pink vase", "vase"])],
        background=[("table", ["table", "desk"])]),
    # Tool_use: 먼지떨이로 책상 청소 (prompt-design few-shot 2 와 동일 구성)
    "Tool_use": dict(
        reference=[],
        background=[("table", ["table", "desk"])]),
}


# ------------------------------------------------------------------ 유도 함수
def _keys(phrases):
    """phrase 목록 -> 매칭키(각 phrase 의 핵심 명사 = 마지막 단어, 중복 제거).
    track3d._match 와 같은 '부분 문자열 포함' 매칭에 쓴다."""
    ks = []
    for p in phrases:
        k = p.split()[-1]
        if k not in ks:
            ks.append(k)
    return ks


def _spec(task):
    return BRAINCO.get(task) or HE_REP.get(task)


def spec_labels(task):
    """spec.py 가 이미 추적하는 표시라벨 집합 (target + 기존 참조물)."""
    sp = _spec(task)
    labels = set()
    if sp:
        for v in sp["targets"].values():
            labels.add(v[0] if isinstance(v, (list, tuple)) else v)
    return labels


def target_labels(task):
    """spec.py 에서 유도한 (a) 조작 대상 표시라벨 집합 (is_target=True)."""
    sp = _spec(task)
    labels = set()
    if sp:
        for v in sp["targets"].values():
            if isinstance(v, (list, tuple)) and bool(v[1]):
                labels.add(v[0])
    return labels


def extra_spec(task, mode="all"):
    """run_sample10 2차 검출 패스 사양. 반환 (prompt, entries).

    entries = [(표시라벨, 카테고리, [매칭키...])]
      mode='task' : 로봇 부위만
      mode='all'  : + spec.py 가 아직 다루지 않는 참조 물체 + 대형 배경
    조작 대상과 spec.py 의 기존 참조물(plate 등)은 기존 EpisodeTracker 경로가
    그대로 처리하므로 여기서 제외한다 — 같은 물체를 두 트래커가 잡으면 산출이 중복된다.
    """
    phrases, entries = [], []
    for label, ph in ROBOT_PARTS:
        phrases += ph
        entries.append((label, "robot_part", _keys(ph)))
    if mode == "all":
        ao = ALLOBJ.get(task, {})
        covered = spec_labels(task)
        for label, ph in ao.get("reference", []):
            if label in covered:                      # spec.py 가 이미 추적 -> 중복 방지
                continue
            phrases += ph
            entries.append((label, "reference", _keys(ph)))
        for label, ph in ao.get("background", []):
            phrases += ph
            entries.append((label, "background", _keys(ph)))
    # GroundingDINO 규격: 소문자 phrase 를 " . " 로 구분, 중복 제거(순서 유지)
    uniq = list(dict.fromkeys(p.lower().strip() for p in phrases))
    return " . ".join(uniq) + " .", entries


def describe(task, mode="all"):
    """사람이 읽을 4갈래 요약 (검증·보고용)."""
    sp = _spec(task)
    prompt, entries = extra_spec(task, mode)
    lines = [f"[{task}] mode={mode}",
             f"  (a) target    : {sorted(target_labels(task))}  (spec.py 프롬프트: {sp['prompt'] if sp else '?'})",
             f"  (b) reference : spec.py={sorted(spec_labels(task) - target_labels(task))}"
             f" + extra={[e[0] for e in entries if e[1] == 'reference']}",
             f"  (c) background: {[e[0] for e in entries if e[1] == 'background']}",
             f"  (d) robot_part: {[e[0] for e in entries if e[1] == 'robot_part']}",
             f"  extra prompt  : {prompt}"]
    return "\n".join(lines)


if __name__ == "__main__":
    for t in list(BRAINCO) + list(HE_REP):
        print(describe(t))
        print()
