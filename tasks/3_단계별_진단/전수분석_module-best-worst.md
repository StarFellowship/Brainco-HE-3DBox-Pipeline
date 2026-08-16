# 모듈별 실행 결과 — 베스트/워스트 사례 정리 (r6 기준)

> 이 문서의 경로 표기는 이 저장소의 `results/최종결과/` 기준이다
> (서버 실행 당시 폴더명은 `review/r6/` 였다).

> 근거 데이터: `../../results/최종결과/` 39유닛(Brainco 8태스크×4카메라 + HE 7카테고리×1카메라)의
> `frames.json`(프레임별 det_score·mask_px·dmed·raw_size·reason)과 `stats.json`을 전수 집계했다.
> 프레임 번호는 `frames.json`의 `f` 인덱스(스트라이드 적용 후 처리 프레임 순번)다.
> 이미지 경로의 `AB_compare.png`는 좌=방식 A(관측만), 우=방식 B(가림 보정) 한 프레임 비교다.

## 지표 정의

| 지표 | 의미 | 출처 |
|---|---|---|
| 실검출률 | `src`가 det·redet인 프레임 / 전체 프레임 (GroundingDINO가 실제로 잡은 비율) | frames.json |
| det_score | GroundingDINO 검출 점수 (승인 하한 ~0.25, 저임계 단일 패스) | frames.json |
| mask_px / CV | SAM 마스크 픽셀 수 / 변동계수(표준편차÷평균) | frames.json |
| dmed | 마스크 영역 depth 중앙값(m) | frames.json |
| size_median | 승인 프레임의 3D 박스 크기 중앙값(m) | stats.json |

## 요약표

| 모듈 | 구분 | 사례 | 수치 근거 | 이미지 경로(../../results/최종결과/ 이하) |
|---|---|---|---|---|
| GroundingDINO | 베스트 | PickDoll 머리좌 | 실검출 220/220, det_score 중앙 0.636·최저 0.38 | `brainco/PickDoll_ep5/cam_left_high/AB_compare.png` |
| GroundingDINO | 베스트 | GraspRubiksCube 머리우 | 실검출 218/220(99.1%), 중앙 0.552 | `brainco/GraspRubiksCube_ep5/cam_right_high/AB_compare.png` |
| GroundingDINO | 워스트 | PickCharger 머리 2대 | 실검출 22~32%, 중앙 0.32~0.34, 팔과 점수 겹침 0.388 vs 0.369 | `brainco/PickCharger_ep5/cam_left_high/AB_compare.png` |
| GroundingDINO | 워스트 | PickToothpaste 머리좌 | 이동 구간 f77~143(66프레임) 실검출 0건 → 방식 A 64.9% | `brainco/PickToothpaste_ep5/cam_left_high/AB_compare.png` |
| SAM 2.1 | 베스트 | HE Articulated(노트북) | mask 중앙 69,578px, CV 0.166(39유닛 중 최저) | `he/Articulated_ep280/AB_compare.png` |
| SAM 2.1 | 베스트 | HE HRI(장미) — 줄기 복원 | A 11×11×6cm(꽃봉오리만) vs B 29×12×6cm(줄기 포함) | `he/HRI_ep5598/AB_compare.png` |
| SAM 2.1 | 워스트 | PickDrink 손목우 | mask 중앙 74px, f48=1px·f49=2px, no_points 기각 141건 | `brainco/PickDrink_ep5/cam_right_wrist/AB_compare.png` |
| SAM 2.1 | 워스트 | PickToothpaste 머리우 | f67=1px → f72~74=1.6~1.7만px(중앙값의 16배) 급팽창 | `brainco/PickToothpaste_ep5/cam_right_high/AB_compare.png` |
| UniDepthV2 | 베스트 | 머리캠 절대 스케일 | 물병 5.4×10.3×5.2cm, 오레오 9.3×5.3×2.1cm — 실물과 cm 단위 부합 | `brainco/PickDrink_ep5/cam_right_high/AB_compare.png` |
| UniDepthV2 | 베스트 | intrinsics 동시 출력 | WildCamera 불필요(러너 미배선, 아래 상세) | — |
| UniDepthV2 | 워스트 | 프레임 간 흔들림 (PickDoll 머리우) | 마스크 안정(8.3~9.9천px)인데 dmed가 0.940→0.867→0.950m(±8~10%) | `brainco/PickDoll_ep5/cam_right_high/AB_compare.png`, f100~110 |
| UniDepthV2 | 워스트 | 절대 오차 (HE 실측 대조) | 근접 0.3~0.8m 오차 43%, 중거리 1~3m 21.7% | — (improvement-log.md 기록) |
| Back-projection·3D Box | 베스트 | PickDrink 머리우 | 원통 물병 x≈z(5.4 vs 5.2cm) — 깊이 방향 두께 복원 | `brainco/PickDrink_ep5/cam_right_high/AB_compare.png` |
| Back-projection·3D Box | 베스트 | GraspOreo 머리좌 | 9.3×5.3×2.1cm, 커버리지 A/B 100% | `brainco/GraspOreo_ep5/cam_left_high/AB_compare.png` |
| Back-projection·3D Box | 워스트 | 얇은 두께 결손 (PickApple 머리 2대) | 둥근 사과인데 z=1.9cm (z/min(x,y)=0.28) | `brainco/PickApple_ep5/cam_left_high/AB_compare.png` |
| Back-projection·3D Box | 워스트 | 손목캠 초근접 (GraspRubiksCube 손목우) | 5.7cm 큐브가 42.6×7.4×6.9cm — x가 실물의 7배 | `brainco/GraspRubiksCube_ep5/cam_right_wrist/AB_compare.png` |
| WildCamera | — | 미사용 | UniDepthV2가 depth와 K를 동시 출력해 러너에 배선하지 않음 | — |

---

## 1. GroundingDINO (1단계: 텍스트 → 2D 박스)

### 베스트

| 사례 | 실검출률 | det_score 중앙/최저/최고 | 비고 |
|---|---|---|---|
| **PickDoll 머리좌** | **220/220 (100%)** | 0.636 / 0.38 / 0.712 | 전 프레임 순수 det(redet·전파 0건). 큰 물체+대비 큰 외형 |
| **GraspRubiksCube 머리우** | 218/220 (99.1%) | 0.552 / 0.325 / 0.632 | 전파(prop) 2프레임뿐. 좌측도 98.6% |

- 참고: det_score 중앙값 자체가 가장 높은 유닛은 PickApple 손목좌(0.813)·머리좌(0.667)다.
  색 대비가 큰 물체(빨간 사과, 컬러 큐브, 무늬 인형)가 점수·검출률 모두 상위.
- 이미지: `../../results/최종결과/brainco/PickDoll_ep5/cam_left_high/AB_compare.png`,
  `../../results/최종결과/brainco/GraspRubiksCube_ep5/cam_right_high/AB_compare.png`

### 워스트

**① 흰 팔 위의 흰 충전기 — PickCharger 머리 2대**

| 유닛 | 실검출률 | det_score 중앙/최고 | 전파 의존 |
|---|---|---|---|
| 머리좌 | 49/220 (22.3%) | 0.321 / 0.456 | prop 160프레임(73%) |
| 머리우 | 70/220 (31.8%) | 0.338 / 0.430 | prop 148프레임(67%) |

- 최고 점수조차 0.46으로, 베스트 유닛의 **중앙값**(0.55~0.64)에도 못 미친다.
- 로봇 전완이 "white charger"로 검출되어 진짜 충전기와 점수가 겹친다
  (0.388 vs 0.369 — `pipeline/profiles.py:163` 진단 기록). 물리 크기 상한(0.15m)으로
  팔을 후보에서 배제해 트랙 탈취를 막았다.
- 접시에 내려놓는 후반부(머리좌 f165~182, 머리우 f193~219)는 검출·전파 모두 기각되어
  방식 A 커버리지가 74.5% / 88.6%에 그친다.
- 이미지: `../../results/최종결과/brainco/PickCharger_ep5/cam_left_high/AB_compare.png`
  (A쪽에는 charger 박스가 없고 B쪽만 6×5×2cm 추정 표시 — 검출 끊김 구간의 전형)

**② 이동 중 미검출 — PickToothpaste 머리좌**

- 집어 옮기는 구간 **f77~f143, 연속 66프레임 동안 실검출 0건**(전 194프레임 중).
  이 구간은 전파로 버티다 prop_shrink 기각 68건이 발생, 방식 A 64.9%.
- 얇고 긴 물체(15×4×3cm)가 손에 쥐이면 보이는 면적이 급감하는 것이 원인으로 추정.
- 방식 B는 크기 이력을 유지해 100%를 채웠다.
- 이미지: `../../results/최종결과/brainco/PickToothpaste_ep5/cam_left_high/AB_compare.png`
  (A는 잔존 관측 6×3×2cm, B는 이력 크기 15×4×3cm)

**참고 — 손목캠 0% 유닛 4개**(오레오 좌·큐브 좌·사과 우·치약 좌 손목)는 물체가 시야에
아예 없는 구간이 대부분이라(조작에 관여하지 않는 손) GroundingDINO의 실패로 보기 어렵다.
README·generalization.md에 기록된 데이터 특성이다.

## 2. SAM 2.1 (2단계: 박스 → 마스크)

### 베스트

| 사례 | mask_px 중앙 | CV(변동계수) | 커버리지 |
|---|---|---|---|
| **HE Articulated(노트북)** | 69,578px | **0.166** (39유닛 최저) | 98.6% |
| **PickDoll 머리좌** | 10,095px | 0.197 | 100% |

- 크고 경계가 뚜렷한 물체에서는 마스크가 프레임 간 거의 흔들리지 않는다.
- **얇은 부속물 복원 사례 — HE HRI(장미)**: 마스크 연결 성분을 깊이 근접성으로 합치는
  `same_depth_components`(`pipeline/geometry.py:109`) 덕에 꽃 줄기가 유지된다.
  A(관측)는 꽃봉오리 11×11×6cm, B는 줄기 포함 29×12×6cm(size_median 28.9×12.4×5.4cm).
  이미지: `../../results/최종결과/he/HRI_ep5598/AB_compare.png`

### 워스트

**① 마스크 소실 — PickDrink 손목우**

- mask_px 중앙값이 **74px**, CV 2.51. f48=1px, f49=2px까지 붕괴.
- 점군을 만들 수 없어 `no_points` 기각 141건 → 방식 A 34.1%(방식 B 96.4%로 보완).
- 손목캠에서 물병이 시야 경계에 걸리거나 잡은 손에 대부분 가려지는 구간이 원인.
- 이미지: `../../results/최종결과/brainco/PickDrink_ep5/cam_right_wrist/AB_compare.png`, 프레임 f48~49

**② 소실 직후 과팽창 — PickToothpaste 머리우**

- f61=6px, f67=1px로 붕괴했다가 **f72~74에서 16,281~17,216px로 급팽창**
  (중앙값 1,042px의 약 16배). 저점수 박스가 주변(팔·접시)을 물고 마스크가 넘친 것으로 추정.
- 결과적으로 no_points 44건 + diag>2.0x 21건 + absmax>0.8 20건이 기각되어 방식 A 55.7%.
- CV 1.517로 마스크 안정성 최하위권.
- 이미지: `../../results/최종결과/brainco/PickToothpaste_ep5/cam_right_high/AB_compare.png`, 프레임 f67 vs f72~74

**③ (과거 라운드 확정) 케이블 혼입** — V2 실측에서 충전기 본체+케이블이 한 마스크로 묶여
박스가 8×22×21cm로 부풀었다(`pipeline/geometry.py:182` 기록). `largest_component` 도입 근거.
현행 r6에서는 재발하지 않는다.

## 3. UniDepthV2 (3단계: RGB → depth, Brainco 전용)

HE는 실측 depth를 쓰므로 이 절의 실측치는 Brainco에만 해당한다.

### 베스트

**① 머리캠 절대 스케일이 실물과 부합**

| 물체 | 추출(size_median) | 실물 참고 |
|---|---|---|
| 물병 (PickDrink 머리우) | 5.4×10.3×5.2cm | 직경 ~6cm 물병 |
| 오레오 (머리좌) | 9.3×5.3×2.1cm | 오레오 스낵팩 |
| 큐브 (머리좌) | 7.2×7.3×2.7cm | 한 변 5.7cm (xy +26%) |

- 0.5~0.9m 근접에서 metric depth의 절대 스케일 오차가 cm 수준에 머문다는 뜻이다
  (단, 아래 워스트 ②의 43% 오차와 공존 — 물체·거리에 따라 편차가 크다).

**② intrinsics 동시 출력** — depth와 K(3×3)를 한 번에 반환한다
(`pipeline/models_wrap.py:136` DepthEstimator). 이 덕에 WildCamera가 필요 없다(5절).

### 워스트

**① 정지 물체인데 프레임 간 depth 흔들림 — PickDoll 머리우 f100~110**

| f | mask_px | dmed(m) |
|---|---|---|
| 103 | 9,912 | 0.940 |
| 104 | 9,565 | **0.867** (−7.8%) |
| 105 | 9,545 | **0.950** (+9.6%) |

- 마스크는 8.3~9.9천px로 안정(±3%)인데 dmed가 프레임 간 ±8~10% 튄다 —
  마스크가 아니라 **depth 추정 자체의 흔들림**이다. 25cm 인형의 깊이(z 16cm)보다
  큰 폭(7~8cm)이 매 순간 오간다.
- 전 유닛 비교: 프레임 간 |Δdmed|/dmed 중앙값이 **Brainco 머리캠 1.5~2.6%** vs
  **HE 실측 depth 0.2~0.7%** — 추정 depth가 실측 대비 약 3~7배 흔들린다.
  (손목캠은 6~15%지만 카메라 자체가 움직이므로 UniDepth 단독 탓으로 볼 수 없다.)
- 이미지: `../../results/최종결과/brainco/PickDoll_ep5/cam_right_high/AB_compare.png` (프레임 특정은 위 표)

**② 절대 오차 (HE 실측 depth 대조, STEP 0.5 실험)**

- 근접 0.3~0.8m에서 **오차 43%**, 중거리 1~3m에서 **21.7%** (improvement-log.md "확인된 데이터 특성").
- r6의 HE 유닛에는 실측 depth를 썼으므로 이 오차가 드러나지 않지만, **GT가 없는 Brainco
  결과의 절대 깊이는 이 수준의 불확실성을 안고 있다**고 봐야 한다.

## 4. Back-projection · 3D Box (4~5단계)

### 베스트

| 사례 | 수치 | 근거 |
|---|---|---|
| **PickDrink 머리우** | 5.4×10.3×5.2cm | 원통 물병이라 x≈z여야 하는데 실제로 5.4 vs 5.2 — 깊이 방향 두께가 복원됨(방식 B의 `mirror_extend_z` 대칭 확장 포함). 커버리지 100% |
| **GraspOreo 머리좌** | 9.3×5.3×2.1cm, A/B 100% | 납작한 물체는 관측면이 곧 두께라 결손 없이 정확 |

- 이미지: `../../results/최종결과/brainco/PickDrink_ep5/cam_right_high/AB_compare.png`,
  `../../results/최종결과/brainco/GraspOreo_ep5/cam_left_high/AB_compare.png`

### 워스트

**① 단일 시점 두께 결손(z 붕괴)**

| 물체 | 추출 | 실물 | z/min(x,y) |
|---|---|---|---|
| 사과 (머리좌) | 6.8×6.0×**1.9**cm | 구형 ~7cm | 0.28 |
| 큐브 (머리좌) | 7.2×7.3×**2.7**cm | 정육면체 5.7cm | 0.37 |

- 카메라가 앞면만 보므로 AABB의 깊이 방향이 구조적으로 과소 추정된다.
  `pipeline/geometry.py:158`(mirror_extend_z 주석)의 "z/min(x,y) 0.25~0.47 붕괴" 실측과 일치.
- 방식 B의 대칭 확장은 관측 두께의 1.8배 상한이라 구형 물체를 다 복원하지 못한다
  (사과: 1.9→최대 3.4cm, 실물 7cm에 미달).
- 이미지: `../../results/최종결과/brainco/PickApple_ep5/cam_left_high/AB_compare.png`
  (B 라벨에 7×6×2cm으로 z 결손이 그대로 보임)

**② 손목캠 초근접 + 배경 혼입**

| 유닛 | size_median | 실물 |
|---|---|---|
| GraspRubiksCube 손목우 | **42.6**×7.4×6.9cm | 5.7cm 큐브 |
| PickDoll 손목우 | 40.2×**60.8**×20.9cm | ~25×27×16cm |

- 물체가 화면을 가득 채우면(큐브 손목우 마스크 중앙 40,443px) 경계 밖 배경·그리퍼가
  점군에 섞이고, dmed도 프레임 간 중앙 11.8%씩 튄다(머리캠의 5배).
- 이미지: `../../results/최종결과/brainco/GraspRubiksCube_ep5/cam_right_wrist/AB_compare.png`
  (B 박스가 화면 전체를 가로지르는 퇴화 형태가 그대로 찍혀 있다)

## 5. WildCamera — 미사용

- **파이프라인에서 쓰지 않는다.** UniDepthV2가 depth와 intrinsics를 동시 출력하기 때문
  (README "WildCamera는 쓰지 않는다", `models_wrap.py` DepthEstimator가 K 반환 실측 확인).
- 래퍼 코드(`pipeline/models_wrap.py:176` IntrinsicsEstimator)는 남아 있으나
  `run_review.py`·`track3d.py` 어디서도 호출되지 않는다(grep 0건).
- HE는 intrinsics 메타가 없어 FOV 70° 가정을 쓴다 — HE 3D 크기가 상수배로 어긋날 수 있는
  원인이며, WildCamera로 추정해 보는 것은 시도해볼 만한 개선 후보다(추측).

---

## 해석 시 주의사항

1. **size_prior 시드 유닛은 크기 정확도 근거로 쓰면 안 된다.**
   PickToothpaste·PickCharger는 실물 치수(0.15,0.04,0.03 / 0.06,0.05,0.02)를 이력 시드로
   주입했다(`profiles.py` TASK_OVERRIDE). 실제로 치약 머리좌·충전기 머리좌의 size_median이
   시드값과 **자릿수까지 동일**하다 — 추출이 아니라 시드가 그대로 남은 것이다.
   크기 정확도 주장에는 시드 없는 유닛(오레오·큐브·사과·물병·인형)만 썼다.
2. **프레임 번호는 처리 프레임 인덱스(f)다.** 원본 영상 프레임이 아니라 스트라이드 적용 후
   순번이므로, 영상에서 찾을 때는 `A_visible.mp4`/`B_amodal.mp4`(같은 폴더)의 f번째 프레임을 보면 된다.
3. **손목캠 결과는 모듈 실패와 데이터 특성이 섞여 있다.** 시야에 물체가 없는 0% 유닛 4개,
   화면을 가득 채우는 초근접 유닛은 모듈 성능 평가에서 분리해서 봐야 한다.
