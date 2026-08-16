# 세부 설정 — 실행 파라미터와 태스크별 프로파일

> 파이프라인이 무엇이고 어떻게 동작하는지는 [pipeline-design.md](pipeline-design.md)에 있다.
> 이 문서는 **값과 조정 방법**만 다룬다.

## 1. 실행 설정

전체 데이터셋 적용 시 조정하는 값들이다. 알고리즘이 아니라 **처리량과 산출물 구성**을 바꾼다.

> **현재 상태**: 러너는 아래 값을 직접 읽는다 —
> `run_review.py` 상단 상수(`STRIDE`, `BC_MAX_FRAMES`, `HE_MAX_FRAMES`)와
> `run_robust.py`의 환경변수(`BC_PER_TASK`, `HE_PER_TASK`, `FPS`, `BC_CAP`, `HE_CAP`, `CAMS`).
> `pipeline/config.py`는 **프리셋 정의와 소요 시간 추정 전용**이며 러너와 아직 배선되지 않았다.

| 항목 | 기본 | 효과 |
|---|---|---|
| `fps` | 6.0 | 초당 처리 프레임. **완료 시간에 정비례** |
| `max_frames_per_ep` | 0 | 에피소드당 상한(0=무제한). 긴 에피소드가 전체를 좌우하는 것을 막는다 |
| `cameras` | 머리 1대 | Brainco 카메라 목록. **시간에 정비례** |
| `episodes_per_task` | 0 | 태스크당 에피소드 수(0=전수) |
| `amodal` | True | 방식 B 산출 여부 |
| `save_video` | False | 오버레이 영상 저장. 대량 실행 시 끈다 |
| `save_frames_json` | True | 프레임별 3D box 기록(실질 산출물) |
| `robot_type` | "g1" | HE는 g1 4,064 + h1 4,885가 섞여 있다. 본 과제는 G1 |
| `align_dx_he` | -20 | HE의 RGB-depth 정렬 보정(px) |
| `hfov_he` | 70.0 | HE intrinsics 가정. 메타에 없어 필요 |

### 프리셋과 예상 소요 (GPU 2장, 실측 0.66초/프레임)

| 프리셋 | fps | 카메라 | 에피소드 | 소요 |
|---|---|---|---|---|
| `review` | 10 | 4대 | 전수 | 161h |
| `balanced` | 6 | 머리 2대 | 전수 | 78h |
| `fast` | 3 | 머리 1대 | 전수 (5,662) | **25h** |
| `full` | 6 | 4대 | 전수 | 137h |
| `robust3` | 6 | 머리 2대 | **태스크당 3개** | **4.5h** |
| `robust5` | 6 | 머리 2대 | 태스크당 5개 | 7.4h |
| `visible_only` | 6 | 머리 2대 | 전수 | 78h (방식 B 없음) |

```bash
python pipeline/config.py     # 위 표를 계산해 출력한다
```

```bash
# 실제 실행은 환경변수로 조정한다
BC_PER_TASK=5 HE_PER_TASK=3 FPS=6 CAMS=cam_left_high,cam_right_high \
  python pipeline/run_robust.py rb1
```

## 2. 태스크별 프로파일 (`pipeline/profiles.py`)

전역 파라미터 하나로는 모든 태스크를 만족시킬 수 없어, **태스크마다 설정을 지정**한다.
(한 태스크를 고치려고 전역 값을 바꿨다가 잘 되던 태스크가 무너진 이력이 있다)

### 기준 프로파일

`R2`·`R3`·`R4`는 **각 개선 라운드에서 확정된 설정 묶음**의 이름이다
(라운드 2·3·4 → 결과 폴더 `review/r2`·`r3`·`r4`. 최종 산출은 `review/r6`).

| 옵션 | R2 (기본) | R3 | R4 |
|---|---|---|---|
| `filter_pct` | (10,90) | (2,98) | (10,90) |
| `filter_mad` | 1.5 | 3.0 | 1.5 |
| `components` | largest | same_depth | same_depth |
| `use_distractors` | True | False | False |
| `anchor_prop` | **False** | True | True |
| `prop_limit_sec` | **0 (무제한)** | 0 | 1.0 |
| `mirror_z` | False | True | True |
| `iou_bonus` | 0.3 | 0.3 | 0.45 |

전파 앵커링(`anchor_prop`)과 시간 제한(`prop_limit_sec`)은 **R3·R4 프로파일에서만** 켜진다.
R2를 쓰는 태스크(오레오·루빅스·사과·충전기·치약·Articulated·Basic)는 적용되지 않는다.

### 태스크 지정

| 태스크 | 기준 | 추가 설정 |
|---|---|---|
| GraspOreo, GraspRubiksCube | R2 | — (완벽 확인, 손대지 않음) |
| PickApple | R2 | `area_max=0.75` |
| PickCharger | R2 | `max_phys_size=0.15`, `prop_grow_hi=1.35`, `shrink_lo=0.55`, `size_prior=(0.06,0.05,0.02)` |
| PickDoll, PickDrink, PickTissues | R3 | — |
| PickToothpaste | R2 | `erode_kernel=3`, `min_points=8`, `size_prior=(0.15,0.04,0.03)`, `shrink_lo=0.45`, `grip_follow=0.5` |
| Articulated | R2 | `filter_pct=(2,98)`, `filter_mad=3.0` |
| Basic | R2 | `cold_start_hist=5`, `max_phys_size=0.25`, `shrink_lo=0.45` |
| deformable, Locomanip, Tool_use | R3 | — |
| HRI | R4 | `max_depth=1.2` |
| Precision | R4 | `exclusive_tracks=True` |

### 개별 옵션

| 옵션 | 기본 | 하는 일 |
|---|---|---|
| `max_phys_size` | 0.75 | 후보의 물리 폭 상한(m). 거리에 **불변**이라 카메라가 달라도 같은 값을 쓴다 |
| `area_max` | 0.60 | 화면 점유율 상한. 근접 촬영에서는 정상 물체도 화면을 채운다 |
| `max_depth` | 0 | 후보 거리 상한(m). 대상이 화면을 벗어났을 때 배경 물체로 옮겨 붙는 것을 막는다 |
| `prop_grow_hi` | 0 | 전파 박스가 이력의 이 배수를 넘으면 기각. 창이 손을 삼키는 것을 막는다 |
| `shrink_lo` | 0 | 전파 박스가 이력의 이 비율보다 작아지면 기각. 유령 방지 |
| `grip_follow` | 0 | 잡혀 있는 물체의 전파 창을 그리퍼 이동량만큼 함께 옮긴다 |
| `size_prior` | () | 알려진 실제 치수를 이력 시드로. 초반에 크기 기준이 서지 않는 것을 막는다. **주의: 시드가 `stats.json`의 `size_median`에 그대로 남을 수 있어, 크기 정확도를 판단할 때는 `frames.json`의 `raw_size`(실검출 프레임)를 봐야 한다** |
| `cold_start_hist` | 0 | 초반 N개 관측을 가림 판정과 무관하게 이력에 넣는다 |
| `exclusive_tracks` | False | 한 프레임에서 두 대상이 같은 박스를 쓰지 못하게 한다 |
| `erode_kernel` | 5 | 마스크 침식. 얇은 물체는 3 |
| `min_points` | 10 | 3D box를 만들 최소 점 수 |

**신규 옵션의 기본값은 전부 비활성**이다. 지정하지 않은 태스크는 코드 경로가 바뀌지 않는다.

## 3. GPU 실행 (연구실 정책)

**PBS batch job으로만 실행한다.** interactive job과 job 없는 직접 실행은 금지다.

- Node 1은 job당 CPU 4개, Node 3은 8개 제한
- job 이름: `G{GPU수}C{CPU수}_{이니셜}_{프로젝트}`
- job 내부 재시도 루프 + 결과 파일 존재 확인 → **중단 시 이어서 진행**

```bash
qsub -q pleiades1 -l select=1:ncpus=4:ngpus=1 -l walltime=05:00:00 \
     -v KIND=all,RND=r7 pipeline/pbs_review.sh
qstat -u <user>
```

## 4. 실측 리소스

| | 값 |
|---|---|
| VRAM | 약 3.4GB (GroundingDINO 0.9 + SAM2.1 1.1 + UniDepthV2 1.4) |
| 처리 속도 | 0.57~0.78초/프레임 (RTX 3090) |
| 산출 용량 | 에피소드당 영상 2편 약 3MB + JSON 압축 약 100KB |
