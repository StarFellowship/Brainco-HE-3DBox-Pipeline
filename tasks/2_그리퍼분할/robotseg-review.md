# RobotSeg 검토 — Gripper/Hand Segmentation 적용성 평가 및 활용 구상

> 원 질문: "Gripper, Hand Gripper, Hand도 Segmentation이 가능한가? 가능하다면 로봇 학습에
> 어떻게 활용할 수 있는가?" — RobotSeg(showlab) 논문·저장소 검토 결과.
>
> 작성: 2026-08-05 (초안). 출처는 각 절 말미와 문서 하단에 명시.

## 0. 요약

- **가능하다.** RobotSeg는 로봇 팔·그리퍼·로봇 전체를 3단계 granularity로 분할하는
  SAM 2 기반 파운데이션 모델이다(CVPR 2026 Oral). 프롬프트 없이 **완전 자동**으로 동작한다.
- 우리 현 스택(GroundingDINO 텍스트 프롬프트 "robot hand" 등)은 로봇 손을 거의 못 잡는다
  (내부 측정: 39유닛 검증에서 손 겹침 배제 `overlaps_hand` 발동 1회). RobotSeg는 이
  약점을 정확히 겨냥한 모델이라 **대체 후보로 타당**하다.
- 단, 학습된 10개 로봇에 **휴머노이드(G1/H1)와 5지 핸드가 없다**. 도메인 갭이 있으므로
  zero-shot 성능은 다음 단계 추론 실험으로 확인해야 한다(체크포인트 공개됨, 6절).

## 1. 방법 상세

### 1.1 무엇인가

| 항목 | 내용 |
|---|---|
| 정체 | 이미지·비디오에서 로봇을 분할하는 파운데이션 모델. "첫 로봇 분할 파운데이션 모델" 주장 |
| 분할 대상 | **robot arm / gripper / whole robot** 3단계 (카테고리 인자로 선택) |
| 기반 | SAM 2 (초기화 가중치 `sam2.1_hiera_tiny.pt`) |
| 크기·속도 | 41.3M 파라미터, 319.8 GFLOPs, 94.2 ms/frame (RTX A5000, >10 FPS) |
| 프롬프트 | 자동(AU) · 1클릭 · 3클릭 · 박스 · 온라인 인터랙티브. **텍스트 프롬프트는 없음** |
| 라이선스 | Apache 2.0 |
| 발표 | CVPR 2026 Oral, arXiv:2511.22950 (Mei, Huang, Ci, Shou / Show Lab NUS) |

SAM 2가 로봇에서 실패하는 이유로 논문이 드는 4가지: 로봇 형태 다양성, 배경과의 외형
모호성(금속·검정 부품), 관절 구조 복잡성, 조작 중 급격한 형상 변화. 이를 아래 3요소로 푼다.

### 1.2 세 가지 핵심 요소

| 요소 | 무엇을 | 어떻게 | 왜 로봇에 필요한가 |
|---|---|---|---|
| **SEMA** (Structure-Enhanced Memory Associator) | SAM 2의 메모리 매칭에 구조 정보 주입 | Canny 에지 E로 특징을 변조(F⊙(1+E))한 구조 브랜치를 시간 브랜치와 병렬로 두고, 메모리와 cross-attention해 구조 맵 S를 만들어 F′′=F′⊙(1+α·S)로 재변조. S는 학습 시 경계 감독을 받음 | 관절이 꺾이며 프레임마다 실루엣이 급변 → 외형 기억만으로는 매칭이 깨짐. 에지 기반 구조 단서로 경계를 유지 |
| **RPG** (Robot Prompt Generator) | 클릭·박스 없는 자동 프롬프트 생성 | ① 클래스 토큰: arm/gripper/robot별 학습된 토큰 뱅크에서 조회(의미 prior) ② 객체 토큰: 과거 메모리 특징을 계층 군집화(FPS 중심 K-means로 macro → 각 영역 내 micro 하위군집)해 프로토타입 토큰 생성 | 대량 로봇 비디오에 수동 프롬프트는 비현실적. "로봇"이라는 의미 범주를 모델이 스스로 프롬프트로 만들어냄 |
| **LET** (Label-Efficient Training) | 첫 프레임 GT만으로 비디오 전체 감독 | ① cycle loss: 0→t 순전파 후 t→0 역전파해 첫 프레임 GT와 대조 ② semantic loss: 중간 프레임 임베딩을 첫 프레임 의미와 코사인 정렬 ③ patch loss: DINOv3 패치 유사도로 첫 프레임 마스크를 중간 프레임에 의사라벨로 전파(16× 패치 단위) | 프레임 전수 마스크 라벨링 비용 회피. VRS 학습셋 2,707 비디오는 첫 프레임만 라벨됨 |

### 1.3 학습 데이터 — VRS 데이터셋

- 2,812 비디오 / 138,707 프레임. 학습 2,707(첫 프레임만 라벨) + 테스트 105(7,203 프레임 전수 라벨).
- 10개 로봇 · 출처 데이터셋:

| 로봇 | 출처 | 로봇 | 출처 |
|---|---|---|---|
| Franka | DROID | MobileALOHA | MobileALOHA |
| Fanuc Mate | Berkeley Fanuc | xArm | UCSD Kitchen |
| UR5 | Columbia PushT | WidowX | Berkeley Bridge |
| Kuka iiwa | Stanford Kuka | Sawyer | RoboTurk |
| Google Robot | RoboVQA | Hello Stretch | DobbE |

- 어노테이션: arm(적)·gripper(녹) 부위별 + whole robot 계층 마스크.
- **시점: 대부분 3인칭 고정 카메라 계열.** 논문에서 egocentric/wrist 시점을 명시적으로
  다루지 않는다(안전 모니터링 등 3인칭 응용을 전제로 서술).

### 1.4 성능 (논문 수치, arXiv v1 기준)

| 벤치마크 | 설정 | RobotSeg | 비교 |
|---|---|---|---|
| VRS (비디오) | whole robot, 자동(AU), J&F | **85.1** | SAM 2.1 파인튜닝 73.6 / RoboEngine 74.1 |
| VRS | arm, AU | 75.6 | — |
| VRS | gripper, AU | 76.0 | — |
| VRS | whole robot, 1클릭 | 75.5 | SAM 2.1 파인튜닝 66.2 |
| RoboEngine (이미지) | whole robot, 자동 | 87.9 | RoboEngine 85.9 |
| 참고 | SAM 3 (concept seg) | — | whole robot 34.7, 일부 로봇에서 0.0 |

- 요점: **자동 모드에서도 gripper 단독 분할이 J&F 76 수준** — 우리가 필요한 것이 바로 이것.
- 논문 명시 한계: 특이 외형의 로봇·장면에서는 여전히 어려움("embodiment-specific modeling"
  여지 언급). ※ 위 수치는 WebFetch로 논문에서 추출한 값 — 실험 착수 전 원문 표와 대조 권장.

### 1.5 입출력·추론 방법 (repo 기준)

- 환경: Python 3.11, PyTorch 2.5.1, CUDA 12.1. `pip install -e ".[dev]"` 후
  `python setup.py build_ext --inplace` (SAM 2 계열과 동일한 빌드 절차).
- 입력: **프레임 이미지 폴더**(jpg/png 시퀀스). 우리 파이프라인도 프레임 단위 처리라 호환.
- 사용 API (test/demo.py):
  ```python
  predictor = build_robotseg_video_predictor("configs/robotseg-infer", "checkpoints/robotseg.pt")
  # CATEGORY ∈ {"arm", "gripper", "robot"}
  predictor.add_new_robot(...)        # 프레임 0에서 자동 시작 — 클릭·박스 불필요
  predictor.propagate_in_video(...)   # 이후 프레임 전파
  ```
- 출력: 프레임별 이진 마스크(데모는 오버레이 jpg 저장, `guided_refine_mask()` 후처리 옵션).
- 배치 스크립트: `test/infer_auto_semi.sh` (RobotSeg 저장소 내부 경로)(자동/반자동), `infer_interactive.sh`(인터랙티브).
- 커스텀 데이터 평가에는 `mask_gt_info` 메타 준비 필요(추론만 할 때는 불필요).

출처: [showlab/RobotSeg](https://github.com/showlab/RobotSeg), [arXiv:2511.22950](https://www.arxiv.org/abs/2511.22950)

## 2. 우리 데이터 적용성 평가

### 2.1 도메인 갭

| 축 | VRS(학습분포) | 우리 데이터 | 갭 평가 |
|---|---|---|---|
| 로봇 종류 | 산업/연구용 팔 10종 (전부 parallel gripper 또는 단순 EEF) | Unitree G1 휴머노이드 + Brainco **5지 핸드**, (H1도 후보) | **큼.** 휴머노이드·다지 핸드는 학습에 전무. 다만 SAM3(34.7)와 달리 RobotSeg는 로봇 일반의 구조 prior를 학습했으므로 미지 로봇에도 부분 일반화 기대 |
| 시점 | 3인칭 고정 카메라 위주 | Brainco 머리 2 + **손목 2**, HE **1인칭 머리 캠** | **중간~큼.** 1인칭은 자기 팔이 화면 하단 밖에서 잘려 들어옴 — VRS 분포 밖 |
| 형상 변화 | 팔 관절 위주 | 5지 핸드 파지 시 손가락 형상 급변 | SEMA가 겨냥한 문제이나 손가락 스케일까지 검증 안 됨 |
| 사람 혼재 | 명시 없음 | HE HRI(handover)에 사람 손 등장 | RobotSeg는 로봇 전용 — **human hand는 여전히 별도 처리 필요** |

### 2.2 예상 문제 (추측 명시)

1. **1인칭 특유의 등장 패턴**: 데모는 프레임 0에서 `add_new_robot`으로 시작한다. 1인칭
   영상은 초반에 로봇이 아예 안 보이다가 팔이 중간에 진입하는 경우가 많다. RPG가 로봇
   부재 프레임에서 거짓 양성을 내는지, 진입 시점을 잡는지는 미검증(추측: 불안정 가능).
2. **손목 카메라**: 자기 손이 초근접·대면적·부분만 보임. VRS에 이런 프레이밍이 없어
   성능 저하 예상(추측).
3. **5지 핸드의 "gripper" 범주 해석**: 클래스 토큰 뱅크가 parallel gripper로 학습됨.
   5지 핸드를 gripper로 잡을지 arm에 흡수할지 불명 — whole robot 모드가 더 안전할 수 있음.
4. **G1 외형**: 검정+흰색 휴머노이드 몸통은 VRS의 팔 로봇과 다르나, "appearance ambiguity"
   대응이 설계 목표였던 만큼 whole robot 모드는 상대적으로 기대 가능.

→ 결론: **그대로 신뢰할 수 없고, 39유닛 대표 프레임에 zero-shot 추론을 돌려 육안 검증 +
소수 프레임 수동 마스크 IoU로 판단해야 한다** (다음 단계 실험 계획, 6절).

## 3. 현 스택 대비 — 왜 대체가 필요한가

### 3.1 현 스택의 실측 한계

- 현 파이프라인은 모든 태스크 프롬프트에 방해물 어휘를 자동으로 붙인다
  (`pipeline/track3d.py:55` `DISTRACTORS = ["robot hand", "robot arm", "gripper", "human hand"]`).
- 이 방해물 트랙의 용도 2가지: ① target 후보가 손과 IoU>0.75로 겹치면 기각
  (`overlaps_hand`, `track3d.py:366`) ② 손 마스크 합집합을 가림 판정에 사용
  (`track3d.py:497-501`).
- **실측: 39유닛(15태스크) 검증에서 `overlaps_hand` 발동 1회** — GroundingDINO가
  "robot hand/arm/gripper" 텍스트로 로봇 손·팔을 거의 검출하지 못한다는 뜻이다.
  결과적으로 가림 판정은 손 마스크가 아니라 **depth 증거**(대상 앞쪽에 더 가까운 픽셀
  비율, `track3d.py:535-550`)에 사실상 의존하고 있다.
- ※ 현 스택의 gripper 검출률 정량 재측정이 별도 GPU job으로 진행 중 —
  **[결과 별도 첨부 예정]**.

### 3.2 RobotSeg로 대체 시 기대

| 항목 | 현재 (GroundingDINO 텍스트) | RobotSeg 대체 시 |
|---|---|---|
| 로봇 손 검출 | 사실상 0에 가까움 (39유닛 중 발동 1회) | 학습분포 내 gripper J&F 76 — 도메인 갭 감안해도 개선 여지 큼 |
| 프롬프트 | 텍스트 어휘 튜닝 필요, 취약 | 자동(프롬프트 불필요) |
| 시간 일관성 | 프레임별 독립 검출 | SAM 2 메모리 + SEMA로 비디오 전파 |
| human hand | 같은 텍스트 경로로 시도 (역시 저조) | **커버 안 됨** — 사람 손은 별도 모델 필요 (예: hand seg 전용 모델) |
| 추가 비용 | — | +94 ms/frame 수준 (파이프라인 stride 처리에는 수용 가능) |

- 판단: **로봇 손·팔에 한해서는 RobotSeg가 현 텍스트 프롬프트 경로의 직접 대체 후보다.**
  마스크(boolean) 출력이라 현 `occ_mask` 인터페이스에 그대로 끼울 수 있다.
  human hand는 커버하지 못하므로 DISTRACTORS에서 로봇 부분만 분리 대체하는 형태가 된다.

## 4. 로봇 학습 활용 구상 (우리 프로젝트 맥락)

### (a) 로봇 마스크 제거 후 world model / 표현 학습
- 로봇 픽셀을 마스킹·인페인팅하면 **물체 중심(object-centric) 장면 표현**을 학습시킬 수 있다
  — world model이 "로봇 팔의 움직임"이 아니라 "물체 상태 변화"를 예측하도록 유도.
- RoboEngine이 실증한 방향(로봇 마스크 + 배경 생성 = 데이터 증강): 같은 시연을 다른
  배경·로봇 외형으로 증강해 시각 정책의 배경 과적합을 줄인다. 우리 G1 데이터에도 로봇
  마스크만 있으면 동일 레시피 적용 가능.
- 반대 방향도 유효: 로봇 마스크를 **어텐션 prior**로 줘서 정책이 자기 몸과 환경을 구분하게 함.

### (b) gripper–물체 접촉·파지 시점 판정
- gripper 마스크와 target 마스크(현 파이프라인이 이미 산출)의 2D 인접 + 두 영역 depth 차
  < 임계 → **접촉 프레임 자동 라벨**.
- 파지 시작/해제 keyframe이 자동으로 나오면: ① 에피소드를 reach/grasp/transport/place
  구간으로 자동 분절(phase 라벨) ② 모방학습의 subgoal·reward 신호 ③ 우리 방식 B의
  "가림 구간" 시작점을 접촉 시점으로 정초(현재는 occ_frac 임계 0.25로만 판정).

### (c) 가림 판정 신호로 직접 사용 — 최소 수정 통합점
- 현재: `occ_mask` = GroundingDINO가 잡은 손 마스크 합집합 (`track3d.py:497-501`) — 위
  실측대로 거의 비어 있고, depth 증거가 대신 일하고 있다. depth 증거는 "앞에 무언가
  있다"만 알지 "그것이 로봇 손인지"는 모른다.
- 제안: RobotSeg gripper(또는 whole robot) 마스크를 `occ_mask`에 OR. **코드 변경이
  한 지점**이고 출력 형식(HxW boolean)이 동일해 통합 비용이 가장 낮다.
- 효과: 가림 판정이 "물리적 증거 + 의미적 증거"가 되어, 배경 물체에 의한 가림과 자기
  손에 의한 가림을 구분 가능 → 방식 B의 크기 유지/갱신 결정이 정확해진다.

### (d) ghost 방지 — 로봇 영역 위 target 박스 기각
- 현 `overlaps_hand` 게이트(IoU>0.75 기각)는 손 박스가 검출돼야만 작동 → 사실상 휴면 상태.
- RobotSeg 마스크로 대체: target 후보 박스 내부의 **로봇 마스크 픽셀 비율 > 임계**면 기각.
  박스 IoU보다 정밀하다(손에 쥔 물체는 로봇 픽셀 비율이 낮아 살아남고, 팔을 물체로
  오인한 후보만 걸림 — 현 설계 의도와 동일하되 실제로 작동하게 됨).
- 추가로 4단계 역투영 전에 **마스크에서 로봇 픽셀을 빼면** 점군 오염(손가락 픽셀이 물체
  점군에 섞여 3D 박스가 부풀는 문제)을 원천 차단할 수 있다.

### (e) 그 외
- **자기가림(wrist 캠) 처리**: 손목 카메라 4유닛 중 방식 A 0% 문제의 일부는 자기 손이
  프레임을 채우는 상황 — 로봇 마스크로 "유효 관측 영역"을 정의하면 wrist 캠 판정 개선 여지.
- **real-to-sim**: 로봇 픽셀 분리 후 배경만 재구성 → 시뮬레이터 장면 이식(논문이 든 응용).
- **안전/품질 필터링**: 로봇 마스크 면적 시계열로 "손이 시야 밖" 에피소드를 사전 걸러내
  wrist 캠 저품질 유닛을 자동 분류.

## 5. 통합 시나리오 요약 (권장 순서)

1. **1단계(다음 단계)**: robotseg.pt zero-shot — 39유닛 대표 프레임 × 3 카테고리(arm/gripper/robot)
   추론, 육안 + 소수 수동 마스크 IoU. 1인칭·손목·5지 핸드에서의 성능 확인이 목적.
2. **2단계**: 성능이 쓸 만하면 (c)+(d) 통합 — 가림 판정과 ghost 게이트를 마스크 기반으로 교체.
   현 스택과 A/B 비교(방식 A/B 검출률, 크기 안정성).
3. **3단계**: 부족하면 LET 활용 — **G1 첫 프레임 수십 장만 라벨**해 파인튜닝. LET가 정확히
   이 시나리오(첫 프레임만 라벨)를 위해 설계됐다는 점이 이 모델의 실용적 장점.

## 6. 체크포인트 다운로드 (다음 단계 추론 실험 준비)

| 항목 | 링크 | 비고 |
|---|---|---|
| **robotseg.pt** | OneDrive: `https://1drv.ms/u/c/f6d9d790b8550d3f/IQDc3mfIAQRETb7zmyhO-BG5AU-cIxzPnUwBDlsrCgcEQ3k?e=oT7NtR` | BaiduDisk 대안(pwd: cvpr). **크기 미공개** — 41.3M 파라미터이므로 fp32 기준 약 170MB 추정(추측) |
| VRS 데이터셋 | OneDrive: `https://1drv.ms/f/c/f6d9d790b8550d3f/IgCB128DB7eUQo9PDO8bkfSxAau1flNmBRe3441a5IyKkGg?e=mG6e3j` | 파인튜닝 단계에서만 필요 |
| SAM2.1 초기가중치 | `sam2.1_hiera_tiny.pt` (Meta 공식) | 학습 시에만 필요, 추론은 robotseg.pt만 |

- 배치: repo의 `checkpoints/` 폴더. 서버에서는 OneDrive 직접 다운로드가 막힐 수 있어
  로컬 수령 후 `scp -P 3022` 업로드가 안전하다.
- 환경: 기존 `task3` conda env(PyTorch 2.5.1/cu121 계열)와 요구사항이 겹치므로 **별도
  env(`robotseg`, Python 3.11) 권장** — SAM 2 빌드(`build_ext --inplace`)가 기존 SAM 2.1
  설치와 충돌할 수 있다(추측, 설치 시 확인).

## 출처

- 저장소: https://github.com/showlab/RobotSeg (README, test/demo.py — 2026-08-05 열람)
- 논문: RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video,
  arXiv:2511.22950, CVPR 2026 Oral. https://www.arxiv.org/abs/2511.22950
- 내부 근거: `pipeline/track3d.py` (DISTRACTORS L55, overlaps_hand L366, 가림 판정 L497-550),
  `review/r6/*/stats.json` 39유닛, README.md 검증 표.
- 현 스택 gripper 검출률 정량 재측정: **[결과 별도 첨부 예정 — GPU job 진행 중]**
