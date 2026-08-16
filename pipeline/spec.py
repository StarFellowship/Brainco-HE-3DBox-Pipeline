"""태스크 사양 — 검출 프롬프트와 주 조작 대상(target object) 정의.

targets 형식: {매칭키: (표시라벨, is_target)}
  is_target=True 가 그 태스크의 '주 조작 대상'이다. 이 라벨에는 깊이 정규화 물리 크기
  상한(기본 0.75m, 태스크별 조정은 profiles.py)이 걸려, PickCharger처럼 로봇 팔 전체를 물체로 오검출하는 것을 막는다.
  프롬프트에는 track3d.DISTRACTORS(robot hand/arm/gripper/human hand)가 자동으로 붙는다.
"""

# ------------------------------------------------------------------ Brainco
# 8개 태스크, 지시문은 meta/tasks.parquet 실측:
#   "Pick up the Oreo" / "Pick up the red cup on the table." / "Pick up the charger." 등
BRAINCO = {
    "GraspOreo": dict(
        prompt="oreo snack package . plate .",
        targets={"oreo": ("oreo", True), "plate": ("plate", False)}),
    "GraspRubiksCube": dict(
        prompt="rubiks cube . plate .",
        targets={"cube": ("rubiks cube", True), "plate": ("plate", False)}),
    "PickApple": dict(
        prompt="apple . plate .",
        targets={"apple": ("apple", True), "plate": ("plate", False)}),
    # charger: 흰 로봇팔과 색이 겹쳐 라운드1에서 팔 전체(화면 16%, 50,482px)를 잡았다.
    # 깊이 정규화 물리 크기 상한으로 막으므로 어휘를 좁힐 필요가 없다.
    "PickCharger": dict(
        prompt="white charger . power adapter . plate .",
        targets={"charger": ("charger", True), "adapter": ("charger", True),
                 "plate": ("plate", False)}),
    "PickDoll": dict(
        prompt="stuffed animal toy . plate .",
        targets={"toy": ("doll", True), "plate": ("plate", False)}),
    # 공식 지시문은 "Pick up the red cup on the table."이지만 영상 실물은 파란 뚜껑
    # 투명 물병이다(수집 중 물체가 바뀌고 지시문이 갱신되지 않음). r3에서 이 프롬프트로
    # 0%→100%가 됐는데, r2 롤백 때 되돌아가 다시 무너졌다.
    "PickDrink": dict(
        prompt="water bottle . clear plastic bottle . red cup . plate .",
        targets={"bottle": ("bottle", True), "cup": ("bottle", True), "plate": ("plate", False)}),
    "PickTissues": dict(
        prompt="pack of wet wipes . plate .",
        targets={"wipes": ("tissue pack", True), "plate": ("plate", False)}),
    "PickToothpaste": dict(
        prompt="toothpaste tube . plate .",
        targets={"toothpaste": ("toothpaste", True), "plate": ("plate", False)}),
}
BRAINCO_CAMS = ["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"]

# ------------------------------------------------------------------ Humanoid Everyday
# 카테고리별 대표: robot_type=g1 이고 target object가 description에 명시된 태스크를 선정.
# (HE 전체 8,949 에피소드 중 g1은 4,064개. 본 과제 대상이 Unitree G1이므로 g1만 쓴다.)
HE_REP = {
    "Basic": dict(
        ep=3800, task="Basic/put_dumpling_into_plate_g1",
        desc="왼손으로 분홍 원형 인형을 집어 주황 접시에 넣는다",
        prompt="pink round toy . orange plate .",
        targets={"toy": ("pink toy", True), "plate": ("orange plate", False)}),
    "Articulated": dict(
        ep=280, task="Articulated/close_a_laptop_g1",
        desc="오른손으로 노트북 덮개를 눌러 닫는다",
        prompt="laptop . notebook computer .",
        targets={"laptop": ("laptop", True), "computer": ("laptop", True)}),
    "deformable": dict(
        ep=4838, task="deformable/fold_towel",
        desc="왼손으로 흰 체크무늬 수건을 반으로 접는다",
        prompt="towel . cloth . fabric on desk .",
        targets={"towel": ("towel", True), "cloth": ("towel", True), "fabric": ("towel", True)}),
    "HRI": dict(
        ep=5598, task="HRI/hand_over_flower",
        desc="왼손으로 장미를 집어 사람에게 건넨다",
        prompt="rose . flower with stem . bouquet .",
        targets={"rose": ("rose", True), "flower": ("rose", True), "bouquet": ("rose", True)}),
    "Locomanip": dict(
        ep=7120, task="Locomanip/walk_towards_a_desk_and_pick_up_a_bottle_and_put_it_in_a_container",
        desc="책상으로 걸어가 병을 집어 용기에 넣는다",
        prompt="water bottle . plastic bottle . container box .",
        targets={"bottle": ("bottle", True), "container": ("container", False)}),
    "Precision": dict(
        ep=7918, task="Precision/insert_flower_into_vase",
        desc="왼손으로 장미를 집어 분홍 꽃병에 꽂는다",
        prompt="rose . flower with stem . pink vase .",
        targets={"rose": ("rose", True), "flower": ("rose", True), "vase": ("vase", False)}),
    "Tool_use": dict(
        ep=8198, task="Tool_use/clean_a_table_with_duster",
        desc="오른손으로 먼지떨이를 집어 책상을 닦는다",
        prompt="duster . feather brush . cleaning tool .",
        targets={"duster": ("duster", True), "brush": ("duster", True), "tool": ("duster", True)}),
}
