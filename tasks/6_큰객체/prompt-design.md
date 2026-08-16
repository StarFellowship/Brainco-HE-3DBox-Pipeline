# 카테고리 추출 프롬프트 정리 + VLM 프롬프팅 설계 (초안)

> 원 요구: "탐지하지 않는 '큰 객체'(e.g. Table)에 대한 전처리 시도.
> 현 카테고리를 추출하는 VLM의 Prompt 구성 정리. VLM Prompting으로
> Segmentation용 Object Categories를 뽑는 Prompt의 설계내용 정리."

## 0. 먼저 바로잡아야 할 사실 관계

**현재 파이프라인의 카테고리(검출 프롬프트)는 VLM이 뽑는 것이 아니라 전부 수작업 지정이다.**

- 위치: `pipeline/spec.py` — 15개 태스크(Brainco 8 + HE 대표 7)의 프롬프트·target을 사람이 직접 작성.
- 작성 근거: 데이터셋 지시문(Brainco `meta/tasks.parquet`, HE description) + **영상 실물 육안 확인**.
- 따라서 이 문서는 (1) 현 수작업 프롬프트의 구성 규칙을 전수 정리하고,
  (2) 큰 객체가 왜 빠지는지 코드 수준에서 분석하고,
  (3) 이 수작업을 대체할 **VLM 자동 추출 프롬프트를 신규 설계**한다. (3)은 아직 미구현 설계안이다
  (다음 단계 시연 목표, 후보 모델 Qwen2.5-VL-7B).

---

## 1. 현 프롬프트 구성 전수 정리 (spec.py 실측)

### 1.1 형식

```python
"태스크명": dict(
    prompt="oreo snack package . plate .",          # GroundingDINO 텍스트 프롬프트
    targets={"oreo": ("oreo", True),                # 매칭키: (표시라벨, is_target)
             "plate": ("plate", False)})
```

- `prompt`: GroundingDINO 규격 — 소문자 영어 명사구를 `" . "`로 구분.
- `targets`의 **매칭키**: 검출 phrase에 부분 문자열로 포함되면 그 트랙에 배정된다
  (`track3d.py _match()`의 `a in str(p)` 매칭).
- **is_target=True** = 주 조작 대상. 이 트랙에만 물리 크기 상한(`max_phys_size`)·거리 상한(`max_depth`)·
  손 겹침 배제(`DIST_IOU_MAX`)가 걸린다. False는 참조물(접시 등)로, 게이트가 느슨하다.
- 프롬프트에는 `track3d.DISTRACTORS = ["robot hand", "robot arm", "gripper", "human hand"]`가
  **자동으로 덧붙는다**(프로파일 `use_distractors=True`일 때). 이 트랙은 렌더링하지 않고
  (1) 검출기가 손을 target으로 흡수하는 것 방지, (2) 가림 판정용 손 마스크로만 쓴다.

### 1.2 Brainco 8태스크 전수

| 태스크 | 지시문(parquet 실측) | prompt | targets (T=주 대상) | 프로파일 | 비고·알려진 실패 |
|---|---|---|---|---|---|
| GraspOreo | "Pick up the Oreo" | `oreo snack package . plate .` | oreo(T), plate | R2 | 완벽, 건드리지 않음 |
| GraspRubiksCube | (동일 계열) | `rubiks cube . plate .` | cube→rubiks cube(T), plate | R2 | 완벽 |
| PickApple | (동일 계열) | `apple . plate .` | apple(T), plate | R2 + area_max 0.75 | 손목캠에서 사과 화면점유 0.627이 상한 0.60에 걸려 56프레임 사망 → 태스크별 상향 |
| PickCharger | "Pick up the charger." | `white charger . power adapter . plate .` | charger(T), adapter(T), plate | R2 + max_phys 0.15m, prop_grow 1.35, size_prior 6×5×2cm | **흰 로봇 전완이 "white charger"로 오검출**(점수 0.388 vs 실물 0.369). 물리 크기 상한 0.15m로 배제. 후반 전파 창이 그리퍼를 삼킴(6×5×2→10×8×7cm) → prop_grow_hi |
| PickDoll | (동일 계열) | `stuffed animal toy . plate .` | toy→doll(T), plate | R3 | wrist는 검출 불가 판정 |
| PickDrink | **"Pick up the red cup on the table."** | `water bottle . clear plastic bottle . red cup . plate .` | bottle(T), cup(T)→같은 트랙, plate | R3 | **지시문 라벨 오류** — 실물은 파란 뚜껑 투명 물병(수집 중 물체 교체, 지시문 미갱신). 지시문만 믿은 라운드에서 0%, 실물 어휘 추가로 100% |
| PickTissues | (동일 계열) | `pack of wet wipes . plate .` | wipes→tissue pack(T), plate | R3 | — |
| PickToothpaste | (동일 계열) | `toothpaste tube . plate .` | toothpaste(T), plate | R2 + erode 3, size_prior 15×4×3cm, grip_follow 0.5 | 얇고 긴 물체라 5×5 침식이 마스크를 지움. 이동 구간 검출 전무 → 그리퍼 변위 추종 |

### 1.3 HE 대표 7태스크 전수

| 카테고리 | 태스크(ep) | prompt | targets | 프로파일 | 비고·알려진 실패 |
|---|---|---|---|---|---|
| Basic | put_dumpling_into_plate_g1 (3800) | `pink round toy . orange plate .` | toy→pink toy(T), plate→orange plate | R2 + max_phys 0.25m, cold_start 5 | 이동 중 검출기가 접시를 toy/plate 양쪽 phrase로 내줌(IoU=1.0) → 물리 크기로 접시 배제(정상 ≤0.159m vs 접시 0.335~0.384m) |
| Articulated | close_a_laptop_g1 (280) | `laptop . notebook computer .` | laptop(T), computer(T)→같은 트랙 | R2 + filter 완화 | 고임계 det이 에피소드에 1회뿐 → det1+redet3 확정 경로 신설. 닫힌 뒤 박스 축소 → 점군 필터만 완화 |
| deformable | fold_towel (4838) | `towel . cloth . fabric on desk .` | towel/cloth/fabric(T)→같은 트랙 | R3 | 펼친 수건 실측 0.65~0.71m — 전역 상한을 0.5m로 두면 잘림(0.75m 유지 근거) |
| HRI | hand_over_flower (5598) | `rose . flower with stem . bouquet .` | rose/flower/bouquet(T)→같은 트랙 | R4 + max_depth 1.2m | 꽃이 화면 위로 벗어나면 **2.2~2.7m 벽면 분홍 물체를 장미로 고임계 오검출**(0.49~0.58, 152중 37프레임) → 거리 상한 |
| Locomanip | walk_towards_a_desk_and… (7120) | `water bottle . plastic bottle . container box .` | bottle(T), container | R3 | — |
| Precision | insert_flower_into_vase (7918) | `rose . flower with stem . pink vase .` | rose/flower(T), vase | R4 + exclusive_tracks | 장미가 화면을 벗어나면 **꽃병으로 트랙이 옮겨 붙음** → 트랙 간 2D 박스 독점 |
| Tool_use | clean_a_table_with_duster (8198) | `duster . feather brush . cleaning tool .` | duster/brush/tool(T)→같은 트랙 | R3 | — |

### 1.4 여기서 추출한 수작업 프롬프트 설계 규칙

VLM 프롬프트가 이 규칙들을 그대로 재현해야 한다.

| # | 규칙 | 근거 사례 |
|---|---|---|
| 1 | **지시문 어휘를 쓰되 영상 실물을 우선한다.** 지시문 라벨은 그대로 믿으면 안 된다 | PickDrink: 지시문 "red cup" vs 실물 물병. 실물 어휘 없이는 0% |
| 2 | **동의어·상위개념을 병기**하고 같은 표시라벨로 한 트랙에 병합한다 (`key_alias`) | rose/flower/bouquet, charger/adapter, laptop/computer — 검출기 어휘 편차 흡수 |
| 3 | **색·재질 수식어**로 특정성을 올린다 | "pink round toy", "white charger", "clear plastic bottle", "orange plate" |
| 4 | **주 조작 대상(is_target)과 참조물을 구분**한다 — target에만 크기·거리·손겹침 게이트 | plate/vase/container는 전부 False. Basic·Precision 실패가 이 구분 덕에 수리 가능했다 |
| 5 | 로봇 손/팔 어휘(DISTRACTORS)는 태스크 지식이 아니라 **상수로 자동 부착** | G1 데이터에 항상 존재. 단 프롬프트가 길어지면 target 검출이 밀린다(r2에서 5개 유닛 0% — R3는 use_distractors=False) |
| 6 | 형식: 소문자 영어 명사구, 관사 없음, `" . "` 구분, 물체당 phrase 1~4개 | GroundingDINO 규격 |

---

## 2. 큰 객체(table/desk)가 빠지는 이유 분석

### 2.1 1차 원인 — 프롬프트에 아예 없다 (설계 의도)

- 15개 태스크 프롬프트 전수에 `table`/`desk`가 **독립 카테고리로 등장하지 않는다**
  (deformable의 "fabric on desk"는 수건을 특정하는 수식일 뿐 desk 트랙이 없다).
- 3D BBox 산출 대상을 "조작 대상 + 직접 상호작용하는 참조물"로 좁게 정의했기 때문이다.
  GroundingDINO는 프롬프트에 없는 것은 검출하지 않으므로, 여기서 이미 배제가 확정된다.

### 2.2 2차 원인 — 프롬프트에 넣어도 크기 게이트 체인이 막는다 (코드 실측)

테이블(폭 1m 이상)을 프롬프트에 추가했다고 가정하고, 후보가 죽는 지점을 코드 순서대로 추적했다.

| 순서 | 위치 | 게이트 | 값 | 테이블(폭≥1m)에의 작용 |
|---|---|---|---|---|
| ① | `track3d.py:364` `_match()` | `area_max` — 2D 박스 화면 점유율 상한 | 0.60 (Profile 기본) | HE 1인칭에서 테이블이 화면 하단을 크게 차지하면 후보 단계에서 기각 가능. 근접 시 확정 기각 |
| ② | `track3d.py:380-383` `_match()` | `max_phys_size` — 깊이 정규화 물리 폭 상한. **is_target=True일 때만** | 0.75m (Profile 기본) | target으로 넣으면 z≈0.8m에서 폭 1m+ → 즉시 기각(`phys>1.00m`). **참조물(False)로 넣으면 이 검사는 건너뜀** |
| ③ | `track3d.py:156-157` `gate()` | `area_max` 재검사 | 0.60 | ①과 동일 기준을 3D 산출 후 한 번 더 |
| ④ | `track3d.py:159-160` `gate()` | **`MAX_SIZE_GLOBAL` — 3D 박스 최대 변 상한. 모듈 전역 상수, is_target 무관** | **0.80m** | **결정적 차단 지점.** 테이블 상판 긴 변 ≥1m → `absmax>0.8` 기각. 참조물로 우회해도 여기서 반드시 죽는다 |
| ⑤ | `geometry.py:87-97` `filter_points()` | depth MAD 전경 분리(1.5) + 백분위 클리핑(10,90) | R2 기본 | 기각 이전에 크기 자체가 왜곡된다 — 비스듬히 보는 상판은 depth 범위가 넓어(예: 0.4~1.5m) 주 모드 밖과 앞뒤 20%가 잘려 **상판 일부만 남는다** → 통과해도 과소 측정 (추측: 실측으로 확인 필요) |
| ⑥ | `track3d.py:167-168` `gate()` | `fill > 0.995` — 마스크가 2D 박스를 통째로 채우면 배경 락온으로 간주 | 0.995 | 화면을 꽉 채운 상판 마스크가 걸릴 수 있음 (경계 조건, 추측) |

- 참고: `track3d.py:37`의 `MAX_PHYS_SIZE = 0.75` 모듈 상수는 **정의만 있고 실제로는 안 쓰인다**
  (실사용은 `Profile.max_phys_size`). 코드 정리 대상.
- 요약: **target으로 넣으면 ②에서, 참조물로 넣으면 ④에서 반드시 죽는다.**
  이 상한들은 "로봇 팔·배경을 물체로 오인하는 것"을 막으려고 도입한 안전판이라
  (README·improvement-log 이력), 단순 상향은 기존 15태스크를 다시 깨뜨릴 위험이 있다.

### 2.3 대응 방향 — 대형객체 전용 프로파일 (별도 트랙 클래스)

- 원칙: 기존 게이트를 건드리지 않고, **카테고리 등급별로 게이트 세트를 분리**한다.
  현 구조가 이미 target/참조물 2등급이므로 3등급(large_background)을 추가하는 확장이다.
- 필요한 코드 변경(설계안):
  - `MAX_SIZE_GLOBAL`을 모듈 상수에서 트랙(등급)별 값으로 — large는 예: 3.0m
  - large 등급: `area_max` 0.90~0.95, `max_phys_size` 미적용, `filter_mad` 완화 또는 비활성
    (상판의 넓은 depth 범위 보존), `mirror_z` 비활성, 렌더 색 구분, is_target 게이트 전부 미적용
  - 정적 물체이므로 이력 게이트(대각 2배)·coasting은 오히려 단순해진다
- **실측 결과**: 게이트 조합을 훑어 화면 점유율 0.50 / 3D 최대변 2.5m로 확정했다.
  통과 13건 중 12건이 실물로 타당한 크기다. 상세는 [README](README.md)와
  [검증통과샘플/](검증통과샘플/) 참조.

---

## 3. VLM 카테고리 자동 추출 설계 (신규 — 다음 단계 시연 목표)

### 3.1 개요

| 항목 | 내용 |
|---|---|
| 목적 | spec.py 수작업을 대체 — 에피소드당 1회 VLM 호출로 검출 프롬프트·등급을 자동 생성 |
| 모델 후보 | **Qwen2.5-VL-7B-Instruct**. bf16 가중치 약 15~16GB로 RTX 3090(24GB) 단일 카드 추론 가능 추정(단일 이미지 기준, 필요 시 AWQ 4bit). 실측 전 |
| 입력 | 에피소드 **첫 프레임 1장** + 지시문(있으면). Brainco는 ffmpeg 추출 프레임, HE는 parquet RGB |
| 출력 | 아래 JSON 스키마. 우리 파이프라인 등급 구조와 1:1 대응 |
| 호출 시점 | 에피소드 전처리 단계 1회 (프레임별 아님 — 비용 무시 가능) |

### 3.2 출력 스키마 — 파이프라인과의 1:1 대응

```json
{
  "manipulation_target": [ {"label": "...", "phrases": ["...", "..."]} ],
  "reference_objects":   [ {"label": "...", "phrases": ["..."]} ],
  "large_background":    [ {"label": "...", "phrases": ["..."]} ],
  "robot_parts":         [ "robot hand", "robot arm" ]
}
```

| 스키마 필드 | 파이프라인 대응 | 변환 규칙 |
|---|---|---|
| manipulation_target | `targets`의 is_target=True 트랙 | label→표시라벨, 각 phrase의 핵심 명사(마지막 단어)→매칭키. phrases 전체를 prompt에 `" . "` join |
| reference_objects | is_target=False 트랙 | 동일 |
| large_background | **신규 3등급 트랙** (2.3의 대형객체 프로파일) | 동일. 현 코드에는 받을 자리가 없으므로 프로파일 확장과 함께 배선 |
| robot_parts | `DISTRACTORS` | 초기에는 **VLM 출력을 쓰지 않고 고정 상수 유지** 권장(현 방식이 검증됨). VLM 출력은 human hand 유무(HRI) 확인 등 교차검증용 |

### 3.3 프롬프트 전문

설계 의도(한국어): 1.4의 수작업 규칙 6개를 시스템 프롬프트의 Rules로 옮겼다.
특히 규칙 1(실물 우선)은 PickDrink 교훈을 그대로 문장화했고, few-shot 1이 이를 시연한다.
동의어 병기(규칙 2)는 "phrases of one object"로, 등급 구분(규칙 4)은 4개 필드로 강제한다.

**System prompt (영어 원문):**

```text
You are an object-category extractor for a robot-manipulation 3D bounding-box
pipeline. You will be given the first frame of a robot episode (RGB, taken by
the robot's own camera) and, when available, the task instruction text.

Return exactly one JSON object and nothing else, with this schema:

{
  "manipulation_target": [ {"label": "<short name>", "phrases": ["<p1>", ...]} ],
  "reference_objects":   [ {"label": "<short name>", "phrases": ["<p1>", ...]} ],
  "large_background":    [ {"label": "<short name>", "phrases": ["<p1>", ...]} ],
  "robot_parts":         [ "<part>", ... ]
}

Rules:
1. TRUST THE IMAGE OVER THE INSTRUCTION. If the instruction names an object
   that does not match what is visible, describe the visible object first and
   append the instruction's word as an additional phrase of the SAME object.
   Never invent an object that is not visible.
2. manipulation_target: the object(s) the robot hand is about to grasp, move,
   press, fold, or hand over. Usually exactly one. Must fit in a robot hand or
   be directly manipulated (max ~0.7 m).
3. reference_objects: movable objects the target interacts with (plate, vase,
   container, laptop base...). Not manipulated directly.
4. large_background: supporting surfaces and furniture larger than ~0.8 m that
   the task happens on or against (table, desk, shelf, cabinet, door).
5. robot_parts: visible robot or human body parts. Choose ONLY from:
   ["robot hand", "robot arm", "gripper", "human hand"].
6. Each "phrases" list: 1-4 short English noun phrases, lowercase, no articles,
   detector-friendly. Include one specific variant with color or material
   (e.g. "clear plastic bottle") and one generic hypernym (e.g. "water bottle").
7. Do not list walls, floor, ceiling, lights, cables, or anything that never
   interacts with the task.
8. If unsure whether two words are the same object, output ONE object with
   both words as phrases — never two objects.
```

**Few-shot 1 — Brainco PickDrink (실물-지시문 불일치 시연):**

```text
[image: PickDrink 에피소드 첫 프레임 — 접시 위 파란 뚜껑 투명 물병, 흰 로봇 팔]
Instruction: "Pick up the red cup on the table."

{
  "manipulation_target": [
    {"label": "bottle",
     "phrases": ["water bottle", "clear plastic bottle", "red cup"]}
  ],
  "reference_objects": [
    {"label": "plate", "phrases": ["plate"]}
  ],
  "large_background": [
    {"label": "table", "phrases": ["table", "desk"]}
  ],
  "robot_parts": ["robot hand", "robot arm"]
}
```

**Few-shot 2 — HE Tool_use / clean_a_table_with_duster (large_background 시연):**

```text
[image: HE ep8198 첫 프레임 — 책상 위 먼지떨이, 1인칭 시점]
Instruction: "Clean a table with duster."

{
  "manipulation_target": [
    {"label": "duster",
     "phrases": ["duster", "feather brush", "cleaning tool"]}
  ],
  "reference_objects": [],
  "large_background": [
    {"label": "table", "phrases": ["table", "desk"]}
  ],
  "robot_parts": ["robot hand", "robot arm"]
}
```

- few-shot은 이미지 포함 멀티턴으로 넣는다(Qwen2.5-VL은 대화 내 다중 이미지 지원).
  기대 출력 JSON은 위처럼 실제 spec.py 값과 정합하게 고정한다.
- 후처리: JSON 파싱 실패 시 1회 재시도(“Return valid JSON only”) → 그래도 실패면 해당
  에피소드는 spec.py 폴백. phrases는 소문자·중복 제거 후 prompt 조립, DISTRACTORS는 기존대로 자동 부착.

### 3.4 검증 방법 — 수작업 spec.py 대비 일치율

대상: 검증에 쓰던 15개 대표(Brainco 8 + HE 7)의 첫 프레임 + 지시문. 정답은 spec.py(+ 2.3의 large 추가분).

| 측정 항목 | 방법 | 통과 기준(제안) |
|---|---|---|
| target 라벨 일치율 | VLM manipulation_target label vs spec.py is_target=True 표시라벨. 동의어는 사람 판정 | 15/15 (핵심 지표 — 하나라도 틀리면 그 태스크는 0%로 직결) |
| phrase 유효율 | 각 phrase를 GroundingDINO에 단독 입력, 첫 프레임에서 해당 물체에 박스가 잡히는 비율 | ≥80% |
| PickDrink 함정 통과 | "red cup" 지시문에도 물병 어휘가 1순위로 나오는가 | 필수 통과 |
| 참조물/large recall | reference+large 합집합 vs 정답(plate·vase·container·table 등) | 누락 ≤1개/태스크 |
| 환각률 | 화면에 없는 물체를 출력한 비율 | 0 |
| end-to-end 등가성 | VLM 산출 프롬프트로 15개 유닛 파이프라인 재실행, 방식 A/B 검출률을 수작업 결과(README 표)와 비교 | 태스크별 차이 ≤5%p |

- 일정: 다음 단계 시연 전 15개 일치율 → 통과 시 robustness 세트(태스크당 3~5 에피소드)로 확대.
- 리스크(추측 포함): (a) 7B 모델이 소형 물체(충전기)를 첫 프레임에서 못 볼 수 있음 → 프레임을
  첫 1장이 아니라 초반 3장 샘플로 늘리는 옵션, (b) phrase가 GroundingDINO 어휘와 안 맞을 수 있음
  → phrase 유효율 측정으로 조기 검출, (c) HE는 지시문이 태스크명·description에 분산 —
  description을 지시문 자리에 넣는다.

---

## 4. 참고 파일

| 파일 | 내용 |
|---|---|
| `pipeline/spec.py` | 15개 태스크 프롬프트·target 원본 |
| `pipeline/track3d.py` | 게이트 체인(`_match` L351-387, `gate` L147-188), DISTRACTORS L55, MAX_SIZE_GLOBAL L43 |
| `pipeline/profiles.py` | 태스크별 프로파일·오버라이드(max_phys_size, area_max 등) |
| `pipeline/geometry.py` | 점군 정제 `filter_points` L68-97 |
| `docs/improvement-log.md` | 실패 사례 상세 이력 |
