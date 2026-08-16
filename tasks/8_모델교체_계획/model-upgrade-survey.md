# 모듈별 최신 모델 교체 조사·권고안 (초안)

- 작성일: 2026-08-05, 조사 기준 시점: 2026-08 웹 조사
- 범위: **조사 + 권고안만**. 설치·구동 실험은 **다음 단계** — GPU는 연구실 정책상 PBS batch job으로만 쓸 수 있고, 이번 주는 GPU 슬롯을 확보하지 못했기 때문
- 서버 제약: RTX 3090 24GB, CUDA **12.3**, PBS batch 전용
- 현 파이프라인(5단계): GroundingDINO → SAM 2.1 → UniDepthV2/실측 depth → 역투영 → AABB
- 참고: 로봇 특화 분할(RobotSeg)은 별도 담당자가 정리 중이라 본 문서에서 제외

## 0. 요약 (한 장)

| 우선순위 | 대상 | 후보 | 한 줄 근거 |
|---|---|---|---|
| **1** | 3단계 (depth) | **MoGe-2**, Depth Pro, UniK3D, DA3-Metric | HE 실측 depth로 **정량 비교가 즉시 가능**(현 기준선: 근접 오차 43%, 중거리 21.7%). 설치 리스크 낮고 GPU 시간도 적다 |
| **2** | 1·2단계 통합 | **SAM 3 / SAM 3.1** | 검출+분할+추적을 한 모델로 — 유령·이동 중 검출 끊김의 **구조적 원인**을 건드린다. 단 CUDA 12.6+ 요구가 서버(12.3)와 충돌 → 설치 검증 필요 |
| **3** | 1단계 단독 | MM-Grounding-DINO, OWLv2 | SAM 3이 막힐 때의 fallback. 드롭인 교체에 가깝지만 기대 이득이 상대적으로 작다 |

---

## 1. 1·2단계 통합 — SAM 3 / SAM 3.1

### 개요

| 항목 | 내용 |
|---|---|
| 발표 | SAM 3: 2025-11-19 (Meta, arXiv 2511.16719). **SAM 3.1: 2026-03-27** drop-in 업데이트 |
| 무엇 | Promptable **Concept** Segmentation — 명사구 텍스트 프롬프트 하나로 이미지·비디오에서 **검출+분할+추적**을 단일 모델(848M, detector와 tracker가 vision encoder 공유)로 수행 |
| 공개 | 가중치·추론/파인튜닝 코드 공개. GitHub `facebookresearch/sam3`, HF `facebook/sam3.1` (HF 인증 필요한 gated 저장소) |
| 라이선스 | **SAM License** (Meta 자체) — 연구·상업 이용 허용하되 제한 조항 있음. 실사용 전 원문 확인 필요 |
| 요구 환경 | Python **3.12+**, PyTorch **2.7+**, **CUDA 12.6+** (repo 명시) |
| VRAM | 이미지 추론 8~24GB. 비디오 추적은 메모리뱅크 크기에 비례(1080p 장편·3트랙·bank 30 기준 55GB+ 사례 보고). **SAM 3.1은 FP16 기준 8→4GB로 절반**, 최대 16객체 단일 패스(Object Multiplex) |

### 우리 파이프라인과의 관계

- 현 구조의 유령(ghost)은 "GroundingDINO 검출이 끊긴 구간을 SAM 2.1 마스크 전파로 메꾸는" 이음새에서 발생했다(개선 이력 V5: 승인 프레임의 7.5%가 유령, 유령 트랙 13개 중 12개가 손목 카메라 — `docs/improvement-log.md`). SAM 3은 검출과 추적이 한 모델 안에서 같은 표현을 공유하므로 **이 이음새 자체가 없어진다**.
- 이동 중 검출 끊김(PickToothpaste 방식 A 65%, PickCharger 89%)도 추적 내장으로 개선을 기대할 수 있다. 다만 "유령이 사라진다"는 보장은 없다 — **track3d.py의 검증 게이트(물리 크기·재투영·대각)는 그대로 유지**하고, SAM 3 출력을 게이트 앞단에 꽂는 구성을 권한다.
- 교체 범위: 1·2단계만 대체하고 3~5단계(depth·역투영·박스)는 재사용. `models_wrap.py`에 SAM3 래퍼를 추가하고 `track3d.py`의 재검출/전파 로직을 우회하는 경로를 만들면 된다.

### 리스크

| 리스크 | 내용 | 대응 |
|---|---|---|
| **CUDA 버전** | repo는 CUDA 12.6+ 명시, 서버는 12.3 | PyTorch cu126 휠은 CUDA minor version compatibility로 12.x 드라이버에서 동작할 **가능성**이 있다(추측 — 스모크로 확인). 안 되면 conda로 cuda-toolkit 12.6 설치 또는 보류 |
| 속도 | 848M 단일 모델 — 현 GDINO(Swin-T)+SAM2.1 조합보다 프레임당 느릴 수 있음 | 15유닛 비교에서 프레임당 시간 실측 후 전수 처리 비용 재산정 |
| 접근성 | HF gated — 계정 인증·라이선스 동의 필요 | 사전에 계정으로 접근 신청 |
| 재튜닝 | 태스크별 프로파일(`profiles.py`)의 임계값이 GDINO 점수 기준 → SAM 3 점수 체계로 재보정 필요 | 15유닛 비교 단계에서 함께 조정 |

- 출처: [Meta 블로그(SAM 3/3.1)](https://ai.meta.com/blog/segment-anything-model-3/), [GitHub facebookresearch/sam3](https://github.com/facebookresearch/sam3), [arXiv 2511.16719 해설(Roboflow)](https://blog.roboflow.com/what-is-sam3/), [MarkTechPost 릴리스 기사](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/), [Spheron 배포 가이드(VRAM)](https://www.spheron.network/blog/deploy-sam-3-gpu-cloud/)

---

## 2. 1단계 단독 후보 (SAM 3이 막힐 때의 fallback)

| 후보 | 공개/라이선스 | 요구 환경 | 기대 이점 | 리스크 |
|---|---|---|---|---|
| **MM-Grounding-DINO** (OpenMMLab) | 가중치 공개, **Apache-2.0** (MMDetection) | mmdet + mmcv — mmcv가 torch/CUDA 버전에 민감 | GDINO 재현+개선판. zero-shot 성능이 원본 GDINO보다 우수하다는 비교 보고 다수. 출력이 같은 형식(텍스트→2D 박스)이라 **드롭인 교체** | mmcv 빌드가 CUDA 12.3에서 번거로울 수 있음. 이득 폭이 태스크 의존적 |
| **OWLv2** (Google) | 가중치 공개, **Apache-2.0**, HF `transformers`로 바로 사용 | transformers만 있으면 됨 — 설치 리스크 최소 | CLIP 계열 self-training으로 zero-shot 검출 강함. 설치가 가장 쉬움 | 구문(색·수식어) 이해나 소형 물체에서 GDINO 대비 경향이 다름 — 우리 프롬프트("oreo snack package" 류)에서의 우열은 실측 필요 |
| **YOLO-World** | 가중치 공개, **GPL-3.0** | 가벼움(수십M) | 실시간급 속도 | **라이선스 제약**(GPL) + 정확도는 GDINO 계열보다 낮은 편. 우리는 오프라인 배치라 속도 이득의 가치가 낮음 → **비권장** |

- 판단: 1단계 단독 교체는 "검출 끊김"의 근본 원인(검출-추적 분리)을 해결하지 못한다. SAM 3 실험이 우선이고, 이 줄은 SAM 3이 환경 문제로 좌초했을 때만 집행.
- 출처: [MM-Grounding-DINO 논문](https://arxiv.org/pdf/2401.02361), [Roboflow GDINO vs OWLv2](https://roboflow.com/compare/grounding-dino-vs-owlv2), [Roboflow GDINO vs YOLO-World](https://playground.roboflow.com/models/compare/grounding-dino-vs-yolo-world), [YOLO-World 라이선스(Roboflow)](https://roboflow.com/model-licenses/yolo-world)

---

## 3. 3단계 — depth 모델 후보

먼저: **UniDepth V3는 없다**(2026-08 검색 기준 미발표). UniDepthV2(arXiv 2502.20110)가 그 계열의 최신이고, 같은 저자의 사실상 후속작은 UniK3D다.

현 기준선(UniDepthV2, HE 실측 depth 대비): **근접(0.3~0.8m) 오차 43%, 중거리(1~3m) 21.7%** (`README.md` 알려진 데이터 특성). 로봇 조작은 근접 구간이 본질이므로 여기가 가장 아픈 지점이다.

| 후보 | 공개/라이선스 | 요구 환경 | intrinsics | 기대 이점 | 리스크 |
|---|---|---|---|---|---|
| **Depth Pro** (Apple, arXiv 2410.02073) | 가중치 공개. **Apple 자체 라이선스(ASCL 계열)** — 연구 사용은 무난, 상업 이용은 조항 확인 필요(GitHub issue #66에서 논란) | Python 3.9+, torch (버전 명시 느슨) | **초점거리 추정 head 내장** — HE의 intrinsics 부재(현재 화각 70° 가정 → "3D 크기 상수배 오차" 리스크) 문제에 직결 | 2.25MP 고해상도 sharp metric depth, 0.3s/장 | 근접 로봇 시점에서의 정확도는 미검증. 라이선스가 4후보 중 가장 제한적 |
| **MoGe-2** (Microsoft, NeurIPS 2025, arXiv 2507.02546) | 가중치 공개, **MIT** (DINOv2 부분 Apache-2.0) | torch 2.x, "대부분 버전과 호환" 명시. **RTX 3090 FP16 기준 60ms/장** 공식 수치 | **metric point map + intrinsics(FOV) 복구**. FOV를 알면 입력으로 줄 수도 있음 | depth map이 아니라 **3D point map을 직접 출력** → 우리 4단계(역투영)를 "마스크로 point map을 자르는" 것으로 대체 가능. 라이선스·속도·서버 궁합 모두 최상 | point map 좌표계 관례가 우리 geometry.py와 맞는지 배선 확인 필요 |
| **UniK3D** (동일 저자, CVPR 2025, arXiv 2503.16591) | 가중치 공개(ViT-S/B/L), **CC BY-NC 4.0(비상업)** | Python 3.10+, **CUDA 11.8+**(cu121 휠 안내) — 서버와 궁합 확인됨 | 카메라 ray 출력, 임의 카메라 모델 지원 | API가 UniDepth와 유사 → **교체 비용 최소**. 저자 벤치마크에서 UniDepth 대비 큰 폭 우위(단, 우위 폭이 가장 큰 것은 광각·파노라마 쪽) | 비상업 라이선스. 핀홀 근접 구간에서의 이득 폭은 실측 필요 |
| **Depth Anything 3 - Metric** (ByteDance-Seed, 2025-11-14, arXiv 2511.10647) | **DA3Metric-Large(0.35B)는 Apache-2.0** (Giant/Large 계열은 CC BY-NC 4.0) | torch ≥ 2 + xformers | 별도 출력 없음(단안 metric 특화) | 단안 metric depth 특화 모델, DA2 대비 향상 보고 | intrinsics를 주지 않음 → HE에는 여전히 화각 가정 필요. metric 특화 변형이 Large 1종뿐 |
| Metric3D v2 (arXiv 2404.15506) | 가중치 공개, BSD-2 | torch (요건 완만) | **intrinsics를 입력으로 요구** | zero-shot metric depth+normal 강함 | 우리 상황(intrinsics 부재)과 반대 방향 → **비권장**. UniDepthV2 출력 intrinsics를 넣는 조합은 가능하나 복잡도만 늘어남 |

### 판단

- **본선 4종: MoGe-2, Depth Pro, UniK3D, DA3-Metric.** HE 실측 depth가 있으므로 넷 다 같은 프레임 집합에서 MAE·구간별 상대오차로 **하루 안에 정량 결판**이 난다.
- Brainco(실측 depth 없음)에는 intrinsics까지 주는 모델이 필요하므로 MoGe-2·Depth Pro가 유리. UniK3D도 가능. DA3-Metric은 intrinsics 별도 조달 필요.
- 출처: [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro), [라이선스 논의 issue #66](https://github.com/apple/ml-depth-pro/issues/66), [microsoft/MoGe](https://github.com/microsoft/moge), [MoGe-2 arXiv](https://arxiv.org/abs/2507.02546), [lpiccinelli-eth/UniK3D](https://github.com/lpiccinelli-eth/UniK3D), [ByteDance-Seed/Depth-Anything-3](https://github.com/bytedance-seed/depth-anything-3), [YvanYin/Metric3D](https://github.com/YvanYin/Metric3D), [UniDepthV2 arXiv](https://arxiv.org/abs/2502.20110)

---

## 4. 비교 우선순위와 검증 방법

### 왜 depth부터인가

1. **정답이 있다** — HE 실측 depth로 MAE·상대오차를 바로 계산할 수 있다. 검출/분할 비교(커버리지·유령률)는 사람 눈 검수가 섞여 오래 걸린다.
2. **GPU가 싸다** — 프레임 수백 장 × 모델 4종이면 3090 한 장에서 1~2시간.
3. **설치 리스크가 낮다** — 4종 모두 CUDA 12.3 서버에서 막힐 사유가 발견되지 않았다(SAM 3만 CUDA 12.6+ 명시).
4. **아픈 곳을 직접 때린다** — 근접 43% 오차와 HE intrinsics 가정은 3D 박스 크기 신뢰도의 최대 병목이다.

### 검증 방법

| 실험 | 데이터 | 지표 | 판정 |
|---|---|---|---|
| depth 4종 비교 | HE 7카테고리 × 대표 에피소드 × 50프레임 샘플(약 350장, 20px 정렬 보정 적용) | 전체 MAE, **구간별 상대오차(근접 0.3~0.8m / 중거리 1~3m)**, 추론 시간/장 | 근접 43% 기준선을 유의하게 깨는 모델 채택. intrinsics 출력 모델은 추정 FOV vs 70° 가정 비교 병기 |
| 채택 depth로 E2E | 15개 대표 유닛 재실행 | 추출 크기 vs 실물(루빅스 5.7cm 등 기지 물체), 방식 A/B 커버리지 회귀 여부 | 크기 오차 감소 + 커버리지 비회귀 |
| SAM 3(3.1) 비교 | 15개 대표 유닛(39 검증 유닛 전부) | 방식 A 커버리지(특히 PickToothpaste 65%·PickCharger 89% 구간), 추출 크기, **유령률**(`coverage_det` 계측 재사용), 손목 카메라 4개 0% 유닛의 변화, 프레임당 처리 시간 | 유령률·커버리지가 현 파이프라인 이상이면 1·2단계 대체 확정 |
| (fallback) 1단계 단독 | 위와 동일 유닛 | 검출 recall/커버리지 | SAM 3 좌초 시에만 |

---

## 5. 다음 단계 실험 계획 (안)

전제: GPU는 PBS batch job으로만 사용. 아래 GPU 시간은 **추정치**이며, 15개 대표 검증 1회가 현 파이프라인에서 소요되는 시간을 기준으로 SAM 3을 2배로 잡았다(848M 단일 모델 가정 — 실측 전 추측).

| 일차 | 작업 | GPU 시간(추정) |
|---|---|---|
| 1 | depth 4종 설치 + 스모크(각 1장 추론, conda env는 충돌 방지 위해 모델별 분리) | ~1h |
| 2 | depth 정량 비교(350장 × 4종) + 결과 표 | 1~2h |
| 3 | SAM 3 설치 스모크 — **관건은 cu126 torch 휠이 CUDA 12.3 드라이버에서 도는지**. 되면 이미지·짧은 클립 추론까지 | ~1h |
| 4~5 | SAM 3 러너 작성(`models_wrap.py` 래퍼 + 4~5단계 재사용) → 15유닛 비교 → 유령률·커버리지 집계 | 4~8h |

- 산출물: depth 비교 표(모델 × 구간 상대오차), SAM 3 vs 현행 39유닛 비교 표, 채택/보류 권고.
- SAM 3이 3일차에 막히면 4~5일차를 1단계 fallback(OWLv2 → MM-GDINO 순, 설치 쉬운 순)으로 전환.

---

## 6. 한계·주의

- 본 문서의 성능 수치는 각 논문·저장소의 자기 보고이며, **우리 데이터(근접 로봇 조작 시점)에서의 우열은 전부 실측 전**이다.
- SAM 3의 CUDA 12.6+ 요구 vs 서버 12.3: minor version compatibility로 동작할 가능성이 있다는 것은 **추측**이다. 스모크가 판정한다.
- SAM License·Apple 라이선스는 요약 기사 기반 — 데이터셋 공개/상업 활용 계획이 생기면 원문 재검토 필요.
- SAM 3.1의 VRAM 절감(FP16 4GB)·속도 2배 수치는 H100 기준 Meta 발표치로, 3090에서의 값은 다를 수 있다.
