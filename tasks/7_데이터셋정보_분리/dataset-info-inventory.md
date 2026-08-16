# 데이터셋 정보 인벤토리 — 파이프라인 사용 가능/불가능 정보 분리 (초안)

> 목적: 3D BBox 파이프라인 관점에서 두 데이터셋(Brainco, Humanoid Everyday=HE)의
> 모든 정보를 (a) 사용 중 / (b) 존재하지만 미활용 / (c) 부재로 분류하고,
> **"예측 박스의 높이·공간 파악이 어렵다"는 문제에 대한 대체 입력**을 검토한다.
>
> 메타는 2026-08-05 pleiades 서버에서 실측 확인했다(로그인 노드, 메타 읽기만).
> 파이프라인의 알려진 특성(depth 오차 43%/21.7%, RGB-depth 20px 어긋남 등)은
> [README '알려진 데이터 특성'](../../README.md#알려진-데이터-특성)과
> [generalization.md](../../docs/generalization.md)를 따르며 여기서 재서술하지 않는다.

## 1. 실측 확인한 데이터셋 구성

### 1.1 Brainco — `meta/info.json` (GraspOreo 기준, LeRobot v3.0)

경로: `/data2/humanoid_dataset_isangmin/G1_Brainco_*_Dataset/meta/info.json`

| feature | dtype | shape | 내용 |
|---|---|---|---|
| `observation.images.cam_{left,right}_high` | video | 3×480×640 | 머리 좌/우 RGB |
| `observation.images.cam_{left,right}_wrist` | video | 3×480×640 | 손목 좌/우 RGB |
| `observation.state` | float32 | [26] | 팔 14관절 + 손가락 12관절 (이름 명시됨) |
| `action` | float32 | [26] | 위와 동일한 26관절 목표값 |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | — | [1] | 인덱스류 |

**이것이 전부다.** depth 없음, intrinsics 없음, extrinsics 없음.
관절 26개는 **팔·손가락뿐** — 머리(목)·허리·다리 관절이 없어 관절값으로
머리 카메라의 자세 변화를 복원할 수 없다. `meta/` 폴더에도 캘리브레이션 파일이 없음을 확인
(`info.json`, `stats.json`, `tasks.parquet`, `episodes/`가 전부).

### 1.2 HE — `meta/info.json` (LeRobot 2.1, 8,949 에피소드 / 246 태스크)

경로: `/data2/humanoid_dataset_isangmin/humanoid-everyday/meta/info.json`

| feature | dtype | shape |
|---|---|---|
| `observation.images.egocentric` | video | 480×640×3 |
| `observation.depth.egocentric` | float32 | 480×640 |
| `observation.lidar` | float32 | [-1, 3] |
| `observation.imu.{quaternion, accelerometer, gyroscope, rpy}` | float32 | [4],[3],[3],[3] |
| `observation.odometry.{position, velocity, rpy, quat}` | float32 | [3],[3],[3],[4] |
| `observation.{arm,leg,hand}_joints` | float32 | [-1] |
| `observation.tactile.{sensor_id, values}` | int64/float32 | [-1], [-1,-1] |
| `action`, `timestamp`, `frame_index`, `episode_index`, `index`, `next.done`, `task_index` | — | — |

기존에 알려진 목록과 일치. intrinsics·extrinsics·카메라-LiDAR 캘리브레이션은 **여기에도 없다**.

**LiDAR 샘플 실측** (`data/chunk-000/episode_000000.parquet`, g1 로봇, 515프레임):

| 항목 | frame 0 | frame 257 |
|---|---|---|
| 점 수 | 4,694 | 4,912 |
| 거리 범위 | 0.10 ~ 32.6 m (중앙값 4.28 m) | 0.10 ~ 24.2 m (중앙값 3.10 m) |
| z 범위 | -2.04 ~ +1.29 m | -1.72 ~ +1.53 m |

- 단위는 **mm**로 판단(최솟값 ~100 = 0.1m, 최댓값 ~32,600 = 32.6m — Livox Mid-360의 사양 거리와 부합). 사용 시 1e-3 스케일 필요.
- 프레임마다 저장돼 있어 RGB와 프레임 단위 동기로 취급 가능(엄밀한 시간 동기 여부는 미확인 — 추측).
- 점 수 ~4,700/프레임이면 테이블 평면 추정(RANSAC)에는 충분하고, 작은 물체(오레오 등) 표면을 직접 잡기에는 부족할 수 있다(물체당 수 점 수준 예상 — 추측, 실험 필요).

## 2. 3분류 — 파이프라인 관점 정보 인벤토리

### (a) 이미 사용 중

| 정보 | 데이터셋 | 파이프라인에서의 역할 (근거 코드) |
|---|---|---|
| RGB 영상 | 둘 다 | 1~2단계 검출·분할 입력. Brainco는 4캠, HE는 1캠 |
| 실측 depth | HE | 3단계 대체 입력. mm→m(×1e-3), RGB 대비 20px 보정(`run_review.py`의 `dscale=1e-3, adx=-20`) |
| UniDepthV2 추정 intrinsics | Brainco | `K_pred`를 그대로 사용, 실패 시 FOV 70° 폴백(`run_review.py` L121-124) |
| 태스크 지시문 | 둘 다 | 검출 프롬프트의 근거. Brainco `tasks.parquet`, HE `episodes.jsonl`의 instruction/description ([pipeline-design.md §6](../../docs/pipeline-design.md)) |
| `frame_index`/`timestamp` | 둘 다 | 프레임 샘플링(stride)과 depth-영상 프레임 매칭 |

### (b) 존재하지만 미활용 — 활용 방안

| 정보 | 데이터셋 | 활용 방안 | 한계·비고 |
|---|---|---|---|
| **LiDAR** ([-1,3], ~4.7k점/프레임) | HE | ① 실측 depth의 독립 검증(같은 평면까지의 거리 비교) ② **테이블 평면·물체 높이의 절대 앵커**(§3-i) ③ FOV 70° 가정의 스케일 검증 — LiDAR 평면까지 거리 vs depth 역투영 거리 비율로 fx 오차를 상수배로 추정 | 카메라-LiDAR extrinsics가 없어 픽셀 단위 정합은 불가. 평면·거리 같은 **저차원 통계 비교만** 가능 |
| **IMU** (quaternion/rpy) | HE | 중력 방향 획득 → 점군을 **중력 정렬 좌표로 회전 후 AABB** 산출. 현재 AABB는 카메라 축 기준이라 카메라가 기울면 '높이'에 폭·깊이가 섞인다. 정렬하면 높이 축이 물리적 수직과 일치 | IMU-카메라 장착 회전(고정 오프셋)을 1회 추정해야 함. 몸통 IMU라 목 관절 움직임은 미반영(추측 — HE g1 카메라가 머리 장착이면 오차 요인) |
| **Odometry** (position/quat) | HE | Locomanip처럼 로봇이 이동하는 태스크에서 카메라 이동 보정, 멀티프레임 점군 융합(뒷면 결손 보완 → 방식 B의 거울 보정 대체) | 오돔 드리프트 수준 미확인. 짧은 구간(수 초) 융합만 현실적 |
| **hand_joints / tactile** | HE | 파지·접촉 시점 검출 → 가림(occlusion) 판정 보조, 방식 B의 크기 고정 시점 결정 | 현재 손 마스크 겹침 방식이 이미 동작 중이라 우선순위 낮음 |
| **arm/hand 관절 26** (state/action) | Brainco | 파지 시점 검출(손가락 관절 닫힘) → 가림 판정 보조 | 손목 카메라 extrinsics를 FK로 만들려면 URDF+카메라 장착 오프셋이 필요한데 둘 다 없음 → 카메라 자세 복원 용도로는 불가 |
| **leg_joints / imu.accel·gyro** | HE | 당장 활용처 없음 (보행 구간 감지 정도) | — |

### (c) 부재 — 현재 대체 방법과 한계

| 부재 정보 | 데이터셋 | 현재 대체 | 한계 |
|---|---|---|---|
| **카메라 intrinsics** | 둘 다 | HE: FOV 70° 가정(`Intrinsics.from_fov`) / Brainco: UniDepthV2 추정 fx | 틀리면 3D 크기가 **상수배**로 어긋남. 검증 수단이 현재 없음 — 추출 크기가 실물과 부합(README 검증표)한다는 간접 증거뿐 |
| **실측 depth** | Brainco | UniDepthV2 추정 | HE 실측 대조에서 근접 43%, 중거리 21.7% 오차(README). 높이 부정확의 **1차 원인** |
| **카메라 extrinsics(장착 위치·자세)** | 둘 다 | 없음 — 박스를 **카메라 좌표계**로만 산출 | 세계/로봇 좌표 박스 불가. 다중 카메라(Brainco 4캠) 결과의 좌표 통합 불가(현재는 카메라별 독립 산출, 크기비 `peer_ratio`로만 교차 확인) |
| **머리(목)·허리 관절** | Brainco | 없음 | 머리 카메라의 프레임 간 자세 변화 보정 불가 → 멀티프레임 점군 융합 불가 |
| **카메라-LiDAR 캘리브레이션** | HE | 없음 | LiDAR를 픽셀 정합 depth로 쓰지 못함. §3-i처럼 평면·통계 수준 활용만 가능 |
| **IMU/odometry-카메라 오프셋** | HE | 없음 | (b)의 IMU/오돔 활용 전에 오프셋 1회 추정 작업이 선행돼야 함 |

## 3. '높이 부정확' 문제 — 대체 입력 3안 검토

문제 정의: 박스 높이(수직 크기·수직 위치)가 부정확한 원인은
① depth 오차(Brainco 추정 depth), ② intrinsics 부재(상수배 스케일), ③ AABB 축이
카메라 축이라 중력 기준 '높이'가 아님 — 의 세 겹이다. 각 안이 어느 원인을 치는지 명시한다.

### (i) HE LiDAR로 테이블 평면·높이 앵커 (원인 ①② 대응)

- **방법**: 프레임별 LiDAR 점군(mm→m)에서 RANSAC으로 지배 평면(테이블/바닥) 추출 →
  카메라 depth 역투영 점군에서도 같은 평면 추출 → 두 평면까지의 거리 비율로
  depth·fx의 상수배 오차를 추정, 박스 높이를 LiDAR 절대값으로 보정.
- **기대효과**: HE에서 FOV 70° 가정의 정오 판정(현재 검증 수단 0개 → 1개).
  맞다면 그대로 신뢰, 틀리다면 보정 계수를 얻는다. Brainco에는 직접 적용 불가지만,
  HE에서 UniDepthV2를 같이 돌려 **추정 depth의 스케일 편향 계수**를 얻으면 Brainco에 이식 가능(간접).
- **비용**: 낮음. RANSAC은 CPU로 충분. 카메라-LiDAR 회전 정합이 없으므로
  평면 **거리·법선 통계만** 비교(픽셀 정합 시도 안 함).
- **위험**: LiDAR가 몸통 장착이면 테이블 상판이 스캔 음영에 들어갈 수 있음(미확인 — 샘플 시각화로 먼저 확인).

### (ii) 테이블 평면 RANSAC으로 박스 밑면 고정 (원인 ③ + 안정화, 두 데이터셋 모두)

- **방법**: 카메라 depth 점군(HE 실측 / Brainco 추정)에서 물체 마스크 주변 영역의
  지배 평면을 RANSAC으로 추출 → 박스 밑면을 평면에 스냅, 높이 = 물체 최고점 − 평면.
  평면 법선을 '수직 축'으로 쓰면 IMU 없이도 중력 정렬 AABB의 근사가 된다.
- **기대효과**: 프레임 간 높이 흔들림 억제, coast(방식 B) 구간에서도 밑면 고정,
  '테이블 위 어느 높이에 있는가'라는 공간 파악이 가능해짐.
- **비용**: 낮음(numpy/Open3D CPU). 기존 4~5단계(geometry.py, GPU 불필요)에 삽입 가능.
- **위험**: 손목 카메라는 테이블이 화면에 안 잡히는 프레임이 많음 → 머리 카메라 한정으로 시작.
  Brainco는 depth 자체가 43% 오차라 평면도 같이 틀린다 — **상대 높이(물체-테이블)는 개선**되지만 절대 스케일은 (i)·(iii) 없이는 못 고친다.

### (iii) Depth Pro류 모델의 초점거리 추정 (원인 ② 대응)

- **방법**: Apple Depth Pro 등 fx를 함께 추정하는 metric depth 모델을 양쪽 데이터셋 프레임에 돌려,
  Brainco는 UniDepthV2 `K_pred`와 교차검증, HE는 FOV 70° 가정과 비교(3자 대조: Depth Pro fx vs 70° fx vs (i)의 LiDAR 보정 계수).
- **기대효과**: intrinsics 상수배 오차의 정량화. 두 추정치가 수렴하면 그 값을 채택할 근거가 생긴다.
- **비용**: 중간. 모델 설치 + **GPU 추론 필요 → PBS batch job**. 프레임 수는 에피소드당 수 장이면 충분(fx는 상수).
- **위험**: 추정 모델끼리의 합의가 정답 보장은 아님(둘 다 같은 방향으로 틀릴 수 있음). (i)의 LiDAR 앵커와 교차해야 근거가 된다.

### 다음 단계 실험 계획 (GPU는 PBS batch job로만 — 연구실 정책)

| 순서 | 실험 | 데이터 | GPU | 산출물 |
|---|---|---|---|---|
| 1 | LiDAR 샘플 시각화 + 단위·커버리지 확인 | HE g1 에피소드 3개 | 불필요(CPU) | 테이블 상판이 LiDAR에 잡히는지 판정 |
| 2 | (i) LiDAR 평면 vs depth 평면 거리 비교 | 위와 동일 | 불필요 | FOV 70° 가정의 오차 계수 |
| 3 | (ii) 평면 RANSAC 밑면 고정을 geometry.py에 옵션으로 추가 | 대표 15유닛 재실행 | 1~3단계 재실행분만 필요 | 높이 흔들림 전/후 비교(프레임 간 높이 표준편차) |
| 4 | (iii) Depth Pro fx 추정 | 태스크당 프레임 5장 | **필요(PBS)** | fx 3자 대조표 |

- 1·2는 로그인 노드 금지 규칙에 따라 parquet 일부를 로컬로 받아 수행하거나 PBS CPU job으로 돌린다.
- 3은 기존 프로파일 원칙(기본 비활성 옵션, 미지정 태스크 코드 경로 불변 — [generalization.md §3](../../docs/generalization.md))을 따른다.

## 4. 요약

- 두 데이터셋 모두 **intrinsics·extrinsics가 메타에 없다**(2026-08-05 서버 실측 재확인). 현재 파이프라인은 이를 FOV 70° 가정(HE)과 UniDepthV2 추정(Brainco)으로 대체 중이며, 이것이 높이·공간 부정확의 구조적 원인이다.
- HE에는 **미활용 자산이 많다**: LiDAR(~4.7k점/프레임, mm 단위, 0.1~32m 실측 확인)·IMU·odometry. 특히 LiDAR는 현재 0개인 'intrinsics/depth 스케일 검증 수단'을 만들 유일한 실측 소스다.
- Brainco는 RGB 4캠 + 팔·손 관절이 전부라 대체 입력이 없다 — HE에서 얻은 보정 계수를 이식하는 간접 경로와, 데이터 불문 적용 가능한 (ii)·(iii)이 현실적 선택지다.
