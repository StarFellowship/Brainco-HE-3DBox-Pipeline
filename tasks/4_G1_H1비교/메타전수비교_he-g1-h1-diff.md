# HE(Humanoid Everyday)의 G1 / H1 로봇별 차이 정리

작성일: 2026-08-05. 원 task ④ "현재까지의 구현 내용 중, HE에서 G1/H1 로봇 별 차이나는 경우가 있는지 확인하여 정리"에 대한 초안.

- 근거: 서버 `/data2/humanoid_dataset_isangmin/humanoid-everyday`의 `meta/tasks.jsonl`(246행)·`meta/episodes.jsonl`(8,949행)·`meta/info.json` **전수 분석** + parquet **실측 5개**(h1 3, g1 2) + 첫 프레임 추출 3장.
- 결론 요약: **파일 포맷·카메라·depth는 g1/h1 동일**하고, 차이는 (1) 태스크 구성이 거의 겹치지 않음(공통 8쌍뿐), (2) 관절 차원(파이프라인 미사용), (3) 영상 속 로봇 손 노출 양상. 파이프라인 코드 변경은 사실상 **태스크 사양(spec) 추가**가 전부다.

---

## 1. 태스크 구성 — g1/h1 쌍은 8개뿐, 나머지는 서로 다른 태스크

`tasks.jsonl` 246개 태스크 전수. 태스크명 접미사는 3종이다.

| 명명 방식 | 개수 | 비고 |
|---|---|---|
| `_g1` 접미 | 32 | 전부 g1 에피소드만 가짐 (episodes.jsonl 대조, 불일치 0건) |
| `_h1` 접미 | 14 | 전부 h1 에피소드만 가짐 |
| 무접미 | 200 | g1 92개 / h1 108개 — **접미사가 없어도 한 태스크는 한 로봇 전용** |

- **로봇이 섞인 태스크는 0개** — 246개 모두 g1 전용(124) 아니면 h1 전용(122)이다. 태스크의 로봇은 접미사가 아니라 `episodes.jsonl`의 `robot_type`으로 판별해야 한다(무접미 200개가 있으므로).
- 특례: HRI에 `HRI/h1-grab_a_book...` 등 **`h1-` 접두** 태스크 5개가 있다(접미가 아니라 접두, 무접미로 분류됨).

### 같은 내용의 g1/h1 쌍 (base명 기준 8쌍)

`Articulated/close_a_kettle_lid`, `Articulated/flip_close_a_diary`, `Articulated/flip_open_a_diary`, `Articulated/pull_out_chips_tray`, `Basic/stack_two_cubes`, `Tool_use/remove_a_soldering_gun_from_its_base`, `deformable/fold_a_toolkit`, `deformable/unfold_a_tablet_cover`

- 접미 붙은 것 중 쌍 없이 g1에만 있는 태스크 24개(예: `Articulated/close_a_laptop_g1` — **우리 Articulated 대표 태스크**, `Basic/put_dumpling_into_plate_g1` — **Basic 대표**), h1에만 있는 태스크 6개(예: `Articulated/close_a_laptop_lid_h1`, `Precision/place_a_soldering_gun_into_its_base_h1`).
- 즉 **h1로 확장해도 지금의 g1 대표 태스크와 같은 물체를 보는 경우는 거의 없다.** h1용 대표 태스크·프롬프트를 새로 골라야 한다(6절).

### 카테고리별 태스크 수 / 에피소드 수 (g1 / h1)

| 카테고리 | 태스크 g1 | 태스크 h1 | 에피소드 g1 | 에피소드 h1 |
|---|---:|---:|---:|---:|
| Articulated | 22 | 26 | 878 | 1,040 |
| Basic | 15 | 49 | 668 | 1,970 |
| HRI | 16 | 25 | 640 | 998 |
| **Locomanip** | **47** | **0** | **879** | **0** |
| Precision | 3 | 5 | 120 | 200 |
| Tool_use | 11 | 8 | 439 | 317 |
| deformable | 10 | 9 | 440 | 360 |
| 계 | 124 | 122 | 4,064 | 4,885 |

- **Locomanip은 h1이 아예 없다** — h1 검증은 6카테고리가 상한이다.
- h1은 Basic 비중이 크다(에피소드의 40%). 대부분 "pick X and place it in a container"류.

## 2. 에피소드 통계 — h1이 수는 많고 길이는 짧다

`episodes.jsonl` 전수, 30fps(`info.json`) 기준 환산.

| | g1 | h1 |
|---|---:|---:|
| 에피소드 수 | 4,064 | 4,885 |
| 길이 중앙값 | 409 프레임 (13.6초) | 316 프레임 (10.5초) |
| 평균 | 438 (14.6초) | 339 (11.3초) |
| 범위 | 116~2,402 (3.9~80.1초) | 110~1,255 (3.7~41.8초) |
| p10~p90 | 260~644 | 226~472 |
| parquet 실존 | (기존 확인) | **4,885/4,885 전부 디스크에 있음** |

- h1이 평균 22% 짧다 → 같은 에피소드 수라면 h1 처리 시간이 오히려 적게 든다.
- `run_robust.py`의 길이 필터 `200 <= length <= 800`은 h1 분포(p10=226)에도 그대로 쓸 수 있다.

## 3. 데이터 스키마 — 컬럼·depth·해상도는 동일, 관절 차원만 다르다

parquet 실측 5개: h1 = ep1960(Basic/lift_a_kettle_from_base), ep40(Articulated/adjust_the_tilt_angle_of_a_monitor), ep5398(HRI/h1-grab_a_book...) / g1 = ep0(Articulated/adjust_the_angle_of_a_phone_stand), ep3800(Basic/put_dumpling_into_plate_g1 — 우리 Basic 대표).

| 항목 | g1 (2개 실측) | h1 (3개 실측) | 파이프라인 영향 |
|---|---|---|---|
| 컬럼 수·구성 | 22개 | 22개 (동일, 누락 없음) | 없음 |
| `observation.depth.egocentric` | 있음, float32 307,200 (=480×640) | **동일하게 있음**, 동일 shape | **없음 — 실측 depth 그대로 사용 가능** |
| depth 값 (mm) | 유효 최소 463~509, 중앙 1,261~1,697 | 유효 최소 463~510, 중앙 790~1,069 | 단위(mm, `dscale=1e-3`) 동일 |
| depth 유효 픽셀 비율 | 92~94% | 84~89% | 표본 2~3개라 단정 못 함. 약간 낮은 경향 |
| 영상 해상도·fps (ffprobe) | 640×480, 30fps | 640×480, 30fps | 없음 |
| `observation.arm_joints` | 14 | 14 | 미사용 |
| `observation.leg_joints` | 15 | **13** | 미사용 |
| `observation.hand_joints` | 14 | **12** | 미사용 |
| `action` | 28 | **26** | 미사용 |
| tactile | 센서 18개 × 4값 | **비어 있음(0개)** | 미사용 |
| lidar (첫 프레임 점수) | 4,583~4,694점 | 3,831~4,900점 | 미사용 |

- `info.json`은 `robot_type: "mixed"`이고 arm/leg/hand/action의 shape을 `[-1]`(가변)로 선언 — 로봇별 차원 차이를 포맷 차원에서 이미 흡수해 둔 구조다.
- **우리 파이프라인이 읽는 것은 영상(mp4) + depth 열 + 내부 파라미터뿐**이므로, 관절·tactile 차원 차이는 어느 단계에도 닿지 않는다. 로더(`frames_he`) 수정 불요.

## 4. 영상 차이 — h1은 자기 손이 화면 하단에 상시 노출

h1 ep1960·ep40, g1 ep3800의 첫 프레임을 서버에서 ffmpeg로 추출해 실측 비교했다.

| 관찰 | g1 (ep3800) | h1 (ep1960, ep40) |
|---|---|---|
| 시점 | 머리 카메라로 책상면을 내려다봄. 프레임에 로봇 신체 없음 | 동일하게 책상을 봄. **화면 하단 1/3에 로봇의 양손(5지 흰/검 dexterous hand)과 몸통 앞 구조물이 크게 들어옴** |
| 장면 | 같은 실험실(흰 책상, 회색 바닥)로 보임 | 동일 환경 |
| 물체 거리감 | depth 중앙값 1.3~1.7m | 0.8~1.1m — 표본상 물체가 더 가까워 보임 |

- 표본 3장 기준의 관찰이며 일반화는 추측이다. 다만 "h1 첫 프레임에 양손 노출"은 2/2 에피소드에서 일치했고, 대기 자세(양손을 앞으로 든 텔레옵 rest pose)에서 비롯된 것으로 보인다.

## 5. 파이프라인을 h1으로 확장할 때 필요한 변경

코드 관점에서 로봇 분기가 있는 곳은 단 한 줄 — `run_robust.py:79`의 `robot_type != "g1"` 필터다. 그 외 변경·확인 사항:

| # | 항목 | 판단 | 근거 |
|---|---|---|---|
| 1 | 로더·depth 처리 (3단계) | **변경 불요** | 3절 — depth 열·단위·해상도 동일 |
| 2 | 내부 파라미터 | 기존 화각 70° 가정 유지 가능하나, **h1 머리 카메라가 같은 기종인지는 메타로 확인 불가** | `info.json`에 intrinsics 없음(g1 때와 동일한 한계) |
| 3 | RGB-depth 20px 정렬 보정 | g1에서 실측한 값 — **h1에서 재확인 필요** | 정렬 오프셋은 카메라 장착에 의존, h1 미검증 |
| 4 | 태스크 사양(`spec.py` HE_REP) | **h1용 대표 태스크·프롬프트 신규 작성 필요** | 1절 — g1 대표 7개 중 h1에 같은 태스크가 있는 것이 없음. Locomanip은 h1 부재로 6카테고리만 가능 |
| 5 | distractor 어휘 | 기존 `robot hand/arm/gripper/human hand` 유지하되 **h1에서 오검출률을 별도 확인** | 4절 — h1은 5지 손이 하단에 상시 노출, "human hand"류로 잡힐 가능성. 잡혀도 distractor로 흡수되면 무해, 대상 물체로 오인되는지가 관건 |
| 6 | 프로파일(`profiles.py`) | 카테고리 키 그대로 재사용 가능하나 값은 **재검증 필요** | HRI `max_depth=1.2` 등은 g1 장면에서 튜닝. 장면(같은 실험실)은 유사하므로 초기값으로는 유효할 것으로 추측 |
| 7 | 에피소드 샘플링 | 길이 필터·seed 로직 그대로 사용 가능 | 2절 |

### h1 실행 결과

> **(자리) h1 파이프라인 실제 실행은 별도 GPU job으로 진행 중 — 결과는 별도 첨부.**

## 6. 근거 파일

| 근거 | 위치 |
|---|---|
| 태스크·에피소드 메타 | 서버 `/data2/humanoid_dataset_isangmin/humanoid-everyday/meta/{tasks.jsonl, episodes.jsonl, info.json}` |
| parquet 실측 | 같은 루트 `data/chunk-*/episode_{000000,003800,000040,001960,005398}.parquet` |
| 영상 실측 | `videos/chunk-*/egocentric/episode_*.mp4` (ffprobe + 첫 프레임 추출) |
| 로봇 분기 코드 | `pipeline/run_robust.py:79`, `pipeline/config.py:74`, `pipeline/spec.py` HE_REP |
