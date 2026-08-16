# -*- coding: utf-8 -*-
"""자동 스캔 + ghost 꼬리 제거 + 육안 판정을 합쳐 최종 판정표를 만든다.

입력:
  verify_units.json          하자 후보 139유닛 메타
  verdicts.json              육안 판정 결과 (워크플로 산출)
  스틸/index.json            스틸 목록
  (서버) trim_결과_유닛별.json  꼬리 제거 통계 — verify_units에 이미 tail 포함

출력:
  판정표_유닛별.json / 판정표.md   유닛×판정 전체표 + task 분류 확정
"""
import json, os, sys
from collections import defaultdict

SC = os.path.dirname(os.path.abspath(__file__))
units = {u["unit"]: u for u in json.load(open(f"{SC}/verify_units.json"))}
verdicts = {v["unit"]: v for v in json.load(open(f"{SC}/verdicts.json"))}
scan = {r["unit"]: r for r in json.load(open(f"{SC}/유닛별_자동스캔.json"))} if os.path.exists(f"{SC}/유닛별_자동스캔.json") else {}

rows = []
for unit, u in units.items():
    v = verdicts.get(unit, {})
    r = {
        "unit": unit, "task": u["task"],
        "cam": "손목" if "wrist" in unit else ("머리" if "high" in unit else "1인칭"),
        "꼬리제거_프레임": sum(u["tail"].values()) if u["tail"] else 0,
        "육안_꼬리가ghost맞음": v.get("tail_ghost", "미판정"),
        "육안_빈공간또는엉뚱한곳": v.get("empty_or_wrong_box", "미판정"),
        "육안_놓친구간에실물보임": v.get("object_visible_when_missed", "미판정"),
        "육안_대상부재확인": v.get("target_absent_confirmed", "미판정"),
        "박스위치": v.get("box_on_what", ""),
        "메모": v.get("notes", ""),
    }
    rows.append(r)

json.dump(rows, open(f"{SC}/판정표_유닛별.json", "w"), ensure_ascii=False, indent=1)

# ── 집계
def agg(rows, key):
    c = defaultdict(int)
    for r in rows: c[r[key]] += 1
    return dict(c)

lines = ["# 육안 판정 통합 결과", ""]
for cam in ["머리", "손목", "1인칭"]:
    sub = [r for r in rows if r["cam"] == cam]
    if not sub: continue
    lines += [f"## {cam} 카메라 ({len(sub)}유닛)", "",
              f"- 꼬리가 ghost 맞음(제거 정당): {agg(sub,'육안_꼬리가ghost맞음')}",
              f"- 추적 구간 박스가 빈 공간/엉뚱한 곳: {agg(sub,'육안_빈공간또는엉뚱한곳')}",
              f"- 검출기가 놓친 구간에 실물 보임: {agg(sub,'육안_놓친구간에실물보임')}", ""]

# 잘못 잘린 것(꼬리인데 실물 있음) — 반드시 확인해야 하는 목록
danger = [r for r in rows if r["육안_꼬리가ghost맞음"] == "no"]
lines += ["## ⚠ 꼬리를 잘랐는데 실물이 있던 유닛 (재확인 필요)", ""]
lines += [f"- {r['unit']} — {r['메모']}" for r in danger] or ["- 없음"]

open(f"{SC}/판정표.md", "w").write("\n".join(lines))
print("\n".join(lines))
