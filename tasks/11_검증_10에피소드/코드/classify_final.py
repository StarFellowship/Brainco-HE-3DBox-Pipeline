# -*- coding: utf-8 -*-
"""육안 판정을 병합해 task별 잠정 분류표를 만든다.

분류 기준 (사용자 지정):
  완벽   = 10개 에피소드 전부, 5개 모듈 모두 깔끔하게 통과
  개선중 = 하자 있는 에피소드가 3개 이하
  못함   = 하자 있는 에피소드가 4개 이상
손목 카메라는 분류에서 제외하고 별도 표로 정리한다 (사용자 검토 예정).

하자 정의 (잠정 — 사용자가 육안 검토 후 기준을 주면 재분류):
  - 박스가 빈 공간이나 엉뚱한 물체 위에 있음 (육안 확정)
  - 검출기가 화면에 보이는 실물을 놓침 (육안 확정)
  - 물체가 보이는데 검출 0회
  - 마지막 검출 이후 남은 박스를 후처리가 잘랐는데 실물이 있었음 (잘못 자름)
  ※ ghost가 후처리로 제거된 것 자체는 최종 산출물 하자로 치지 않되 별도 집계
  ※ '판단 애매'는 하자로 안 세고 별도 집계
"""
import json, os
from collections import defaultdict

SC = os.path.dirname(os.path.abspath(__file__))
verdicts = {v["unit"]: v for v in json.load(open(f"{SC}/verdicts.json"))}
scan = json.load(open(f"{SC}/유닛별_자동스캔.json"))
task_rows = [r for r in scan if r["mode"] == "task"]


def cam(unit):
    return "손목" if "wrist" in unit else ("머리" if "high" in unit else "1인칭")


def epi(unit):                      # task/brainco/GraspOreo/ep00006_cam_left_high -> (GraspOreo, 6)
    parts = unit.split("/")
    return parts[2], int(parts[3].split("_")[0][2:])


# 영상(에피소드×카메라)별 판정 정리
per_video = {}
for r in task_rows:
    unit = r["unit"]
    v = verdicts.get(unit)
    defects, notes, ghost_removed, unclear = [], [], False, []
    if v:
        if v["empty_or_wrong_box"] == "yes":
            defects.append("박스가 빈 공간/엉뚱한 물체 위")
        if v["object_visible_when_missed"] == "yes":
            defects.append("보이는 실물을 검출기가 놓침")
        if v["target_absent_confirmed"] == "no":
            defects.append("물체가 보이는데 검출 0회")
        if v["tail_ghost"] == "yes":
            ghost_removed = True                      # 후처리로 제거됨 — 하자 아님, 집계만
        if v["tail_ghost"] == "no":
            defects.append("후처리가 실물 프레임을 잘못 자름")
        for k in ["empty_or_wrong_box", "object_visible_when_missed"]:
            if v[k] == "unclear":
                unclear.append(k)
        notes.append(v.get("notes", ""))
    elif r["flags"]:
        unclear.append("판정누락")
    per_video[unit] = {"task": r["task"], "cam": cam(unit), "defects": defects,
                       "unclear": unclear, "ghost_removed": ghost_removed,
                       "box_on": (v or {}).get("box_on_what", ""), "note": " ".join(notes)}

# 후보에 없던 영상(코드 훑기에서 깔끔) 도 포함해야 함 — scan의 flags 없는 것
# task_rows에 전부 있으므로 defects=[]로 이미 포함됨

# ── 에피소드 단위 합산 (머리/1인칭만)
ep_def = defaultdict(lambda: defaultdict(list))    # task -> ep -> defects
ep_unc = defaultdict(lambda: defaultdict(list))
for unit, pv in per_video.items():
    if pv["cam"] == "손목":
        continue
    t, e = epi(unit)
    ep_def[t][e] += pv["defects"]
    ep_unc[t][e] += pv["unclear"]

lines = ["# task별 잠정 분류 (머리·1인칭 카메라, 육안 판정 반영)", "",
         "기준: 완벽=10개 에피소드 모두 깔끔 / 개선중=하자 에피소드 3개 이하 / 못함=4개 이상",
         "('판단 애매'는 하자로 세지 않고 따로 표기. 최종 기준은 사용자 검토 후 확정)", "",
         "| task | 에피소드 | 하자 에피소드 | 애매 | 잠정 분류 | 대표 하자 |", "|---|---|---|---|---|---|"]
summary = {}
for t in sorted(ep_def):
    eps = ep_def[t]
    n = len(eps)
    bad = [e for e in eps if eps[e]]
    unc = [e for e in eps if not eps[e] and ep_unc[t][e]]
    cls = "완벽" if not bad else ("개선중" if len(bad) <= 3 else "못함")
    fc = defaultdict(int)
    for e in bad:
        for d in set(eps[e]): fc[d] += 1
    top = ", ".join(f"{k}({v}ep)" for k, v in sorted(fc.items(), key=lambda x: -x[1])[:2])
    lines.append(f"| {t} | {n} | {len(bad)} | {len(unc)} | **{cls}** | {top or '—'} |")
    summary[t] = {"분류": cls, "하자ep": sorted(bad), "애매ep": sorted(unc),
                  "하자유형": dict(fc)}

# ── 하자 유형 → 원인 모듈 매핑
lines += ["", "## 하자 유형별 발생 영상 수 (머리·1인칭)", "",
          "| 하자 | 영상 수 | 원인 모듈 |", "|---|---|---|"]
tc = defaultdict(int)
for unit, pv in per_video.items():
    if pv["cam"] == "손목": continue
    for d in set(pv["defects"]): tc[d] += 1
mod = {"박스가 빈 공간/엉뚱한 물체 위": "1단계 검출이 끊긴 동안 2단계 SAM 추적이 어긋남",
       "보이는 실물을 검출기가 놓침": "1단계 GroundingDINO",
       "물체가 보이는데 검출 0회": "1단계 GroundingDINO",
       "후처리가 실물 프레임을 잘못 자름": "후처리 규칙(검출 실패를 소멸로 오인)"}
for k, v in sorted(tc.items(), key=lambda x: -x[1]):
    lines.append(f"| {k} | {v} | {mod.get(k,'')} |")

gr = sum(1 for pv in per_video.values() if pv["cam"] != "손목" and pv["ghost_removed"])
lines += ["", f"- ghost가 있었지만 후처리로 제거된 영상: {gr}개 (하자로 안 셈)",
          "", "## 손목 카메라 (분류 제외, 별도 검토용)", "",
          "| 판정 | 영상 수 |", "|---|---|"]
wc = defaultdict(int)
for unit, pv in per_video.items():
    if pv["cam"] != "손목": continue
    if pv["defects"]: wc["하자 확인"] += 1
    elif pv["unclear"]: wc["판단 애매"] += 1
    else: wc["하자 후보였으나 육안상 문제 없음 또는 물체가 원래 안 보임"] += 1
for k, v in wc.items(): lines.append(f"| {k} | {v} |")

json.dump({"per_video": per_video, "task_summary": summary},
          open(f"{SC}/판정표_전체.json", "w"), ensure_ascii=False, indent=1)
open(f"{SC}/판정표.md", "w").write("\n".join(lines))
print("\n".join(lines))
