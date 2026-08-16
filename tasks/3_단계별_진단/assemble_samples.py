"""드라이브 공유 폴더 최종 조립 — 육안 검증(에이전트 12개) 통과분만.

선정 원칙: 전 후보를 렌더 → 전량 육안 검증 → '베스트/워스트 라벨에 부합'만 채택.
워스트 후보인데 그럴듯해 보이는 것은 전부 탈락시켰다 (지난 실패의 원인).
"""
import os, shutil, json

B = "/Users/hansangmin/StarFellowship/task3-3dbbox-pipeline"
D = os.path.expanduser("~/Downloads/StarFellowship/신규task_결과")
GD = f"{B}/probe6cpu/gd"; BP = f"{B}/probe6cpu/bp"; SAM = f"{B}/probe6/sam"
DEP = f"{B}/probe6/depth"; GHO = f"{B}/probe6cpu/ghost"; G1 = f"{B}/probe6cpu/g1"
H1 = f"{B}/probe2"; BIG = f"{B}/probe6/bigobj"; P3 = f"{B}/probe3"

def cp(src, dstdir, name):
    os.makedirs(dstdir, exist_ok=True)
    if not os.path.exists(src):
        print(f"  !! 없음: {src}"); return
    shutil.copy(src, os.path.join(dstdir, name))

def wtxt(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)

# ================================================================ 3. 모듈별
M = f"{D}/3_모듈별_베스트워스트"
shutil.rmtree(M, ignore_errors=True)

# ---- Brainco / GroundingDINO
d = f"{M}/Brainco/1_GroundingDINO_2D박스/베스트"
cp(f"{GD}/gd_best_doll_head_f102.png",  d, "베스트1_인형_머리캠_박스정확밀착.png")
cp(f"{GD}/gd_best_drink_head_f147.png", d, "베스트2_물병_머리캠_소형물체정확검출.png")
cp(f"{GD}/gd_best_oreo_head_f6.png",    d, "베스트3_오레오_머리캠_파지중검출성공.png")
cp(f"{GD}/gd_best_cube_head_f198.png",  d, "베스트4_큐브_머리캠_박스밀착.png")
d = f"{M}/Brainco/1_GroundingDINO_2D박스/워스트"
cp(f"{GD}/gd_worst_charger_gap_f109.png",  d, "워스트1_충전기_손에들려있는데_미검출.png")
cp(f"{GD}/gd_worst_charger_gap2_f2.png",   d, "워스트2_충전기_테이블위노출인데_미검출.png")
cp(f"{GD}/gd_worst_tissue_ghosthand_f116.png", d, "워스트3_티슈없는_로봇손목에_티슈박스.png")

# ---- Brainco / SAM
d = f"{M}/Brainco/2_SAM_픽셀분할/베스트"
cp(f"{SAM}/bb_apple1_f70.png",  d, "베스트1_사과_경계픽셀단위정확.png")
cp(f"{SAM}/bb_drink1_f3.png",   d, "베스트2_물병_뚜껑까지단일영역.png")
cp(f"{SAM}/bb_oreo1_f122.png",  d, "베스트3_오레오봉지_구겨진외곽추종.png")
cp(f"{SAM}/bb_tpaste1_f170.png", d, "베스트4_치약상자_모서리정확.png")
cp(f"{SAM}/bb_cube1_f88.png",   d, "베스트5_큐브_블러에도단일영역.png")
cp(f"{SAM}/bb_doll1_f83.png",   d, "베스트6_인형_보풀경계추종.png")
d = f"{M}/Brainco/2_SAM_픽셀분할/워스트"
cp(f"{SAM}/bw_doll_frag_f115.png", d, "워스트1_인형마스크_6조각파편화_손가락침범.png")
cp(f"{SAM}/bw_doll_prop_f163.png", d, "워스트2_인형없는데_과노출배경을_인형마스크로.png")

# ---- Brainco / UniDepth
d = f"{M}/Brainco/3_UniDepthV2_깊이추정/베스트_안정"
cp(f"{DEP}/dp_oreo_head_stable_f77-78.png",   d, "베스트1_머리캠_연속프레임_변화0퍼센트.png")
cp(f"{DEP}/dp_apple_head_stable_f154-155.png", d, "베스트2_머리캠_사과장면_안정.png")
cp(f"{DEP}/dp_doll_head_stable_f195-196.png",  d, "베스트3_머리캠_인형장면_안정.png")
cp(f"{DEP}/dp_cube_wrist_stable_f160-161.png", d, "베스트4_손목캠도_안정구간존재.png")
d = f"{M}/Brainco/3_UniDepthV2_깊이추정/워스트_흔들림"
cp(f"{DEP}/dp_cube_wrist_jump_f195-196.png", d, "워스트1_손목캠_연속프레임인데_깊이270퍼센트점프.png")
cp(f"{DEP}/dp_doll_head_jump_f104-105.png",  d, "워스트2_머리캠_물체깊이실루엣소실_10퍼센트점프.png")

# ---- Brainco / Back-projection
d = f"{M}/Brainco/4_Backprojection_3D박스/베스트"
cp(f"{BP}/bp_best_doll_head_f102.png", d, "베스트1_인형_29x28x19cm_실물30cm부합.png")
cp(f"{BP}/bp_best_oreo_head_f6.png",   d, "베스트2_오레오_9x6x2cm_실물치수일치.png")
d = f"{M}/Brainco/4_Backprojection_3D박스/워스트"
cp(f"{BP}/bp_worst_drink_wrist_f161.png", d, "워스트1_물병없는_화면구석에_45cm박스.png")
cp(f"{BP}/bp_worst_cube_wrist_f187.png",  d, "워스트2_큐브_손목캠_13x16cm_실물5.7cm의2배이상.png")
cp(f"{BP}/bp_worst_apple_wrist_f91.png",  d, "워스트3_사과_손목캠_14x14x3cm_2배과대_두께납작.png")
cp(f"{BP}/bp_worst_tissue_wrist_f33.png", d, "워스트4_티슈_크기붕괴.png")
cp(f"{BP}/bp_worst_doll_wrist_f131.png",  d, "워스트5_인형_손목캠_박스일부만덮고_과소추정.png")

# ---- HE / GroundingDINO
d = f"{M}/HE/1_GroundingDINO_2D박스/베스트"
cp(f"{GD}/gd_best_bottle_he_f61.png", d, "베스트1_물병_박스타이트.png")
cp(f"{GD}/gd_best_towel_he_f62.png",  d, "베스트2_수건_전체정확포함.png")
cp(f"{GD}/gd_best_duster_he_f38.png", d, "베스트3_먼지떨이_정확검출.png")
cp(f"{GD}/gd_best_laptop_f7.png",     d, "베스트4_노트북_저점수0.38이지만_박스정확.png")
cp(f"{GD}/gd_best_toy_f44.png",       d, "베스트5_장난감_그리퍼겹쳐도_검출.png")
d = f"{M}/HE/1_GroundingDINO_2D박스/워스트"
cp(f"{GD}/gd_worst_rose_vase_f156.png", d, "워스트1_장미박스가_화병까지통째로_과확장.png")

# ---- HE / SAM
d = f"{M}/HE/2_SAM_픽셀분할/베스트"
cp(f"{P3}/sam_towel_mask.png",   d, "베스트1_수건_구겨진윤곽_가림홈까지정확.png")
cp(f"{SAM}/hb_laptop1_f43.png",  d, "베스트2_노트북_그리퍼겹침부위만파임.png")
cp(f"{SAM}/hb_toy1_f29.png",     d, "베스트3_장난감_바구니침범없음.png")
cp(f"{SAM}/hb_bottle2_f32.png",  d, "베스트4_물병_실루엣타이트.png")
cp(f"{SAM}/hb_rose2_f165.png",   d, "베스트5_장미_옆송이와혼동없이한송이분할.png")
d = f"{M}/HE/2_SAM_픽셀분할/워스트"
cp(f"{SAM}/hw_duster_speck_f32.png", d, "워스트1_먼지떨이마스크_229조각점묘파편화.png")
cp(f"{SAM}/hw_rose_vase_f156.png",   d, "워스트2_장미마스크가_화병몸통까지삼킴.png")
cp(f"{SAM}/hw_laptop_over_f36.png",  d, "워스트3_노트북마스크가_배경기둥까지과확장.png")

# ---- HE / 깊이
wtxt(f"{M}/HE/3_깊이추정/README.txt",
"""HE(Humanoid Everyday)는 UniDepthV2를 사용하지 않습니다.
데이터셋에 실측 depth(mm)가 포함되어 있어 파이프라인이 실측값을 직접 사용합니다.
따라서 깊이 추정 모듈의 베스트/워스트 사례는 Brainco 폴더에만 있습니다.
(UniDepthV2 정확도 평가는 이 실측 depth를 기준으로 수행했으며 — 근접 43%,
중거리 21.7% 상대오차 — 상세 수치는 보고서 본문 참조)
""")

# ---- HE / Back-projection
d = f"{M}/HE/4_Backprojection_3D박스/베스트"
cp(f"{BP}/bp_best_duster_he_f38.png",  d, "베스트1_먼지떨이_34cm_실물35cm일치.png")
cp(f"{BP}/bp_best_bottle_he_f81.png",  d, "베스트2_물병_10x14x17cm_그럴듯.png")
cp(f"{BP}/bp_best_laptop_he_f7.png",   d, "베스트3_노트북_육면체가형상에밀착.png")
d = f"{M}/HE/4_Backprojection_3D박스/워스트"
cp(f"{BP}/bp_worst_towel_max_f82.png", d, "워스트1_얇은수건에_39cm두께큐브.png")
cp(f"{BP}/bp_worst_duster_he_f98.png", d, "워스트2_박스대부분이_빈테이블_52cm과대.png")
cp(f"{BP}/bp_worst_rose_hri_f130.png", d, "워스트3_장미박스가_화면모서리에_위치이탈.png")

wtxt(f"{M}/README.txt",
"""# 모듈별 베스트 / 워스트 사례

구조: <데이터셋>/<모듈>/{베스트, 워스트}/  — 파일명에 사례 요지를 담았습니다.

모듈별 이미지 종류
  1_GroundingDINO_2D박스 : 2D 박스 오버레이 (초록=조작 대상, 파랑/주황=참조 물체)
  2_SAM_픽셀분할          : 픽셀 마스크 오버레이 (노랑 반투명 + 주황 윤곽선, 상단에 픽셀 수)
  3_UniDepthV2_깊이추정   : 연속 2프레임의 [RGB | 깊이 컬러맵] 비교 (상단에 물체 중앙깊이 m)
  4_Backprojection_3D박스 : 3D 박스 오버레이 (크기 라벨 cm)

선정 방법 (지난번 지적 반영)
  전 후보(약 100장)를 실제 렌더한 뒤 전량을 육안 검증하여,
  "처음 보는 사람이 한 장만 보고도 잘함/못함을 즉시 알 수 있는" 사례만 채택했습니다.
  수치상 워스트였지만 눈으로 보면 멀쩡한 후보 9건은 전부 제외했습니다.

사례 수가 적은 곳의 사유 (숨김 없이 기록)
  - Brainco/SAM 워스트 2장: 워스트 후보 5건을 검증한 결과 3건은 "박스가 틀렸을 뿐
    마스크는 박스 안을 올바르게 분할"한 경우로 판명 → SAM 고유 실패가 아니라 제외.
    Brainco에서 SAM 실패는 대부분 앞단(검출) 문제로 나타납니다.
  - Brainco/Backprojection 베스트 2장: 머리 카메라의 소형 물체는 단일 시점 특성상
    두께 축이 구조적으로 얇게 나와(사과 6x6x2cm 등) "3축 모두 그럴듯" 기준을
    만족하는 사례가 인형·오레오 2건뿐이었습니다.
  - HE/GroundingDINO 워스트 1장: HE 검증 7유닛에서는 5프레임 이상의 미검출 공백이
    발견되지 않았습니다(검출 자체는 안정적). 유일한 실패 유형이 과확장(장미→화병)입니다.
  - 깊이 워스트 2장: 머리 카메라의 프레임 간 ±10~17% 흔들림은 컬러맵으로는 거의
    보이지 않아(수치로만 확인됨) 제외하고, 육안으로 확인 가능한 극단 사례만 남겼습니다.
""")

# ================================================================ 4. G1/H1
G = f"{D}/4_G1_H1_비교"
shutil.rmtree(G, ignore_errors=True)

pairs = [
    ("1_Articulated_관절형물체", [
        (f"{G1}/g1_Articulated_ep280_A_f110.png", "G1_ep280_노트북덮개조절_부분성공_두께축과대.png"),
        (f"{H1}/h1_Articulated_ep120.png",        "H1_ep120_마우스클릭_부분성공_박스가왼쪽으로밀림.png")]),
    ("2_Basic_기본조작", [
        (f"{G1}/g1_Basic_ep3800_A_f110.png", "G1_ep3800_장난감집기_부분성공_장난감박스일부이탈.png"),
        (f"{H1}/h1_Basic_ep1960.png",        "H1_ep1960_주전자들기_성공_21x24x24cm실물부합.png"),
        (f"{H1}/h1_Basic_ep2680.png",        "H1_ep2680_웹캠집기_부분성공_바구니는잡고_웹캠은미검출.png")]),
    ("3_deformable_변형체", [
        (f"{G1}/g1_deformable_ep4838_A_f110.png", "G1_ep4838_수건_부분성공_바닥면정확_높이4배과대.png"),
        (f"{H1}/h1_deformable_ep4718.png",        "H1_ep4718_재킷접기_실패_대표프레임미검출_에피소드검출률43퍼센트.png")]),
    ("4_Precision_정밀조작", [
        (f"{G1}/g1_Precision_ep7918_A_f110.png", "G1_ep7918_장미꽃병_부분성공_꽃병박스하반부만.png"),
        (f"{H1}/h1_Precision_ep7998.png",        "H1_ep7998_인두꽂기_실패_미검출.png")]),
    ("5_Tool_use_도구사용", [
        (f"{G1}/g1_Tool_use_ep8198_A_f110.png", "G1_ep8198_먼지떨이_부분성공_높이3배과대.png"),
        (f"{H1}/h1_Tool_use_ep8398.png",        "H1_ep8398_망치질_실패_미검출.png")]),
    ("6_G1전용_카테고리_H1에는없음", [
        (f"{G1}/g1_HRI_ep5598_A_f110.png",       "G1_HRI_ep5598_장미전달_실패_박스가꽃옆검은영역에.png"),
        (f"{G1}/g1_Locomanip_ep7120_A_f110.png", "G1_Locomanip_ep7120_물병_부분성공_박스겹침혼란.png")]),
]
for sub, files in pairs:
    for src, name in files:
        cp(src, f"{G}/{sub}", name)
cp(f"{B}/probe2/h1.json", G, "H1_실행통계.json")

wtxt(f"{G}/README.txt",
"""# G1 / H1 로봇별 결과 비교

구조: 카테고리별 폴더 안에 G1_*.png 와 H1_*.png 를 나란히 두었습니다.
파일명 = 로봇_에피소드_태스크_판정_근거 순서. 판정은 이미지를 실제로 열어
재검증한 결과이며, 이전 요약("주전자·마우스 100% 성공")보다 엄격합니다.

이미지 형식: 좌=방식A(보이는 부분만), 우=방식B(가림 보정). G1 컷은 방식A 단독.

## 판정 요약 (대표 프레임 1장 기준, 정직 재판정)

카테고리        G1                          H1
Articulated     부분성공 (두께축 과대)       부분성공 (박스 밀림 — 이전 '100%'는 과장)
Basic           부분성공                     성공(주전자) / 부분성공(바구니, 웹캠은 미검출)
deformable      부분성공 (높이 4배 과대)     실패 (대표 프레임 미검출, 에피소드 검출률 43%)
Precision       부분성공                     실패 (미검출)
Tool_use        부분성공 (높이 3배 과대)     실패 (미검출)
HRI/Locomanip   실패/부분성공                (H1에는 이 카테고리가 없음)

## 결론 (보고서와 동일)
- 파일 포맷·카메라·depth 는 동일 → 파이프라인이 코드 수정 없이 H1에서 동작.
- 결과 품질은 혼재: H1 명확한 성공은 주전자 1건. 실패 3건은 물체가 손에 들려
  시야 밖이거나 작고 가장자리에 있는 경우로, G1에서도 같은 조건이면 실패하는 유형.
- G1 쪽도 대표 프레임 기준 "완전 성공"은 드물고 높이(깊이 축) 과대가 공통 약점
  — 로봇 차이보다 단일 시점 깊이 문제가 지배적.
- 단, 표본이 카테고리당 1~2 에피소드라 H1 고유 열화가 없다고 단정할 수 없음.

H1_실행통계.json: H1 6개 에피소드의 검출률·크기 수치 원본.
""")

# ================================================================ 5. ghost 축소
Gh = f"{D}/5_ghost_개선_전후"
keep_units = {"GraspRubiksCube_right_wrist", "GraspOreo_right_wrist", "PickApple_left_wrist"}
old = {}
if os.path.isdir(Gh):
    for u in os.listdir(Gh):
        p = os.path.join(Gh, u)
        if os.path.isdir(p):
            old[u] = p
shutil.rmtree(Gh + "_tmp", ignore_errors=True)
os.makedirs(Gh + "_tmp", exist_ok=True)
# 영상 3유닛만 이관
for u, p in old.items():
    if u in keep_units:
        shutil.copytree(p, f"{Gh}_tmp/영상예시_{u}")
shutil.rmtree(Gh, ignore_errors=True)
os.rename(Gh + "_tmp", Gh)

PAIRS = [
    ("brainco_GraspRubiksCube_ep5_cam_right_wrist_rubikscube_f199", "01_루빅스큐브_손목캠"),
    ("brainco_GraspOreo_ep5_cam_right_wrist_plate_f198",            "02_오레오유닛_접시유령"),
    ("brainco_PickApple_ep5_cam_left_wrist_apple_f176",             "03_사과_손목캠"),
    ("brainco_PickCharger_ep5_cam_right_high_charger_f181",         "04_충전기_머리캠_로봇손위박스"),
    ("brainco_PickCharger_ep5_cam_right_wrist_plate_f198",          "05_충전기유닛_배경허공접시박스"),
    ("brainco_PickDoll_ep5_cam_left_wrist_plate_f179",              "06_인형유닛_그리퍼감싼접시박스"),
    ("brainco_PickDrink_ep5_cam_left_wrist_bottle_f201",            "07_물병_바닥허공박스"),
    ("brainco_PickTissues_ep5_cam_left_wrist_tissuepack_f115",      "08_티슈_빈로봇손위박스"),
    ("brainco_PickCharger_ep5_cam_left_high_charger_f184_B",        "09_방식B예시_충전기_허공추정박스"),
]
d = f"{Gh}/검증통과_전후컷"
for base, name in PAIRS:
    cp(f"{GHO}/{base}_before.png", d, f"{name}_개선전.png")
    cp(f"{GHO}/{base}_after.png",  d, f"{name}_개선후.png")

wtxt(f"{Gh}/README.txt",
"""# Ghost(유령 박스) 개선 전/후 — 검증 통과 샘플만

이 task는 '진행 중'입니다 (꼬리 전파형은 해결, 검출 오류형은 다음 주 SAM 3 실험).
따라서 폴더에는 "개선전에 유령이 실제로 보이고, 개선후에 제거가 확인된" 쌍만 남겼습니다.

검증 방법: 51개 전후 쌍을 전량 육안 대조 → 25쌍 통과. 여기엔 유닛별 대표 1쌍(총 9쌍)만
수록. 탈락 사유는 주로 "개선전 컷에 박스가 안 보임(트리밍이 1~2프레임 잔가지였던 경우)"
또는 "개선전 박스가 실물 위에 있어 유령으로 단정 불가".

파일명 규칙: NN_유닛_상황_개선전/개선후.png — 같은 번호끼리 같은 프레임입니다.
09_방식B예시: 방식B는 마지막 실검출 후 1초까지 추정(주황 점선)을 유지하고 중단합니다.

영상예시_* 3개 폴더: 전체 맥락(꼬리 전파가 자라나는 과정)을 영상으로 확인용.
""")

# ================================================================ 6. 큰객체 축소
Bg = f"{D}/6_큰객체_table"
shutil.rmtree(Bg, ignore_errors=True)
Bg2 = f"{D}/6_큰객체_대형객체샘플"
shutil.rmtree(Bg2, ignore_errors=True)

d = f"{Bg2}/브레인코_인형장면_3종세트"
cp(f"{BIG}/bc_doll_head_2dbox_all.png",          d, "1_전체_2D박스_테이블접시의자.png")
cp(f"{BIG}/bc_doll_head_0_white_table_mask.png", d, "2_테이블_픽셀마스크.png")
cp(f"{BIG}/bc_doll_head_0_white_table_3dbox.png", d, "3_테이블_3D박스_125x63x53cm.png")
cp(f"{BIG}/bc_doll_head_1_plate_mask.png",       d, "4_접시_픽셀마스크.png")
cp(f"{BIG}/bc_doll_head_1_plate_3dbox.png",      d, "5_접시_3D박스_19x15x11cm.png")
cp(f"{BIG}/bc_doll_head_2_chair_3dbox.png",      d, "6_의자_3D박스_45x28x50cm.png")
d = f"{Bg2}/브레인코_오레오장면"
cp(f"{BIG}/bc_oreo_head_1_white_table_mask.png", d, "테이블_픽셀마스크.png")
cp(f"{BIG}/bc_oreo_head_0_plate_mask.png",       d, "쟁반_픽셀마스크.png")
cp(f"{BIG}/bc_oreo_head_2_chair_mask.png",       d, "의자_픽셀마스크_다리까지분할.png")
d = f"{Bg2}/HE_노트북장면"
cp(f"{BIG}/he_articulated_1_laptop_mask.png",  d, "노트북_픽셀마스크.png")
cp(f"{BIG}/he_articulated_1_laptop_3dbox.png", d, "노트북_3D박스_56x28x36cm.png")
cp(f"{BIG}/he_articulated_0_desk_mask.png",    d, "책상_픽셀마스크.png")
d = f"{Bg2}/HE_기타장면"
cp(f"{BIG}/he_basic_2_table_mask.png",     d, "기본조작_테이블마스크.png")
cp(f"{BIG}/he_locomanip_0_table_mask.png", d, "이동조작_테이블마스크.png")
cp(f"{BIG}/meta.json", Bg2, "결과요약.json")

wtxt(f"{Bg2}/README.txt",
"""# 큰 객체(대형 객체) 확장 실험 — 잘 나온 샘플 발췌

이 task는 '진행 중'입니다. 게이트(화면점유율·절대크기 상한)를 대형객체용으로 완화하면
검출이 되는 것까지 확인했고, 파라미터 확정과 VLM 프롬프트 자동화는 다음 주 진행합니다.

table 외에 접시(쟁반)·의자·노트북·책상·바구니 등으로 확장해 시험했고,
객체마다 2D박스 / 픽셀마스크 / 3D박스 3종을 산출했습니다. 여기엔 36장 중
육안 검증을 통과한 14장만 수록 (탈락: floor 505cm 비정상, 바구니 높이 3배 과대,
그리퍼를 의자로 오검출 등 — 전체 수치는 결과요약.json).

브레인코_인형장면_3종세트가 대표 사례: 한 장면에서 테이블(1.25m)·접시·의자가
모두 2D→마스크→3D까지 이어진 경우.
""")

# ================================================================ 루트 README
wtxt(f"{D}/README.txt",
"""# 신규 task 결과물 (2026-08-06 재정리)

1_2D_bounding_box/        [완료] 2D box 오버레이 영상 39편 + boxes2d.json 예시
2_gripper_segmentation/   [완료] 로봇 팔·손 분할 실측
3_모듈별_베스트워스트/     [완료] 데이터셋×모듈×베스트/워스트 폴더. 전량 육안 검증 통과분만
4_G1_H1_비교/             [완료] 카테고리별 G1·H1 나란히 + 정직 재판정 (파일명=판정)
5_ghost_개선_전후/         [진행중] 검증 통과 전후컷 9쌍 + 영상 3유닛만 발췌
6_큰객체_대형객체샘플/     [진행중] 확장 실험에서 잘 나온 14장만 발췌

3·4번은 2026-08-06 피드백을 반영해 전면 재구성:
- 후보 약 100장을 실제 렌더 → 전량 육안 검증 → 라벨에 부합하는 것만 채택
- "워스트인데 멀쩡해 보이는" 후보 9건 제외, "베스트인데 어긋난" 후보 4건 제외
- 파일명에 사례 요지를 담아 이미지만 봐도 무엇인지 알 수 있게 함
""")

# 요약 출력
total = 0
for root, dirs, files in os.walk(D):
    total += len([f for f in files if not f.startswith(".")])
print("조립 완료")
for sub in sorted(os.listdir(D)):
    p = os.path.join(D, sub)
    if os.path.isdir(p):
        n = sum(len(fs) for _, _, fs in os.walk(p))
        print(f"  {sub}: {n}개 파일")
print(f"총 {total}개 파일")
