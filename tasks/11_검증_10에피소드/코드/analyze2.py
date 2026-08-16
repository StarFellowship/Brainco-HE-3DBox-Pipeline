# -*- coding: utf-8 -*-
"""자동 스캔 2차: 카메라 종류 분리 + 잔상 실제 구간 + 모든객체 모드의 라벨 구성."""
import json, glob, os
from collections import defaultdict

ROOT = os.path.expanduser("~/task3/sample10/s10")
TH_DET, TH_BOX3D, TH_TRACK = 0.90, 0.95, 0.30


def camkind(unit):
    if "wrist" in unit: return "손목"
    if "high" in unit:  return "머리"
    return "1인칭"       # HE


def scan(sp):
    st = json.load(open(sp))
    n = st.get("n_frames", 0)
    if not n: return None
    unit = os.path.dirname(sp).replace(ROOT + "/", "")
    f = []
    tgt = {k: v for k, v in st.get("objects", {}).items() if k.startswith("target/")}
    if not tgt: f.append("대상없음")
    for k, v in tgt.items():
        fr = max(1, v.get("frames", 0))
        if v.get("det", 0) / n < TH_DET:      f.append("검출끊김")
        if v.get("box3d", 0) / fr < TH_BOX3D: f.append("3D생성실패")
        if v.get("track", 0) / n > TH_TRACK:  f.append("추적의존")
    coast = 0
    for lb, t in st.get("tracker_stats", {}).items():
        if t.get("is_target"):
            coast = max(coast, t.get("coasting_frames", 0))
    if coast > 0: f.append("잔상꼬리")
    return {"unit": unit, "task": st.get("task"), "mode": st.get("mode"),
            "cam": camkind(unit), "n": n, "coast": coast,
            "flags": sorted(set(f)), "labels": sorted(st.get("objects", {}).keys())}


rows = [r for r in (scan(p) for p in sorted(glob.glob(f"{ROOT}/*/**/stats.json", recursive=True))) if r]
tk = [r for r in rows if r["mode"] == "task"]

print("## 1) 카메라 종류별 — 무결 비율\n")
print("| 카메라 | 유닛 | 무결 | 하자 | 무결률 |")
print("|---|---|---|---|---|")
for c in ["머리", "손목", "1인칭"]:
    rs = [r for r in tk if r["cam"] == c]
    if not rs: continue
    ok = [r for r in rs if not r["flags"]]
    print(f"| {c} | {len(rs)} | {len(ok)} | {len(rs)-len(ok)} | {len(ok)/len(rs):.0%} |")

print("\n## 2) 머리·1인칭 카메라만 — task 분류\n")
print("| task | 유닛 | 무결 | 하자 | 분류 | 주요 하자 |")
print("|---|---|---|---|---|---|")
head = [r for r in tk if r["cam"] != "손목"]
bt = defaultdict(list)
for r in head: bt[r["task"]].append(r)
cls_count = defaultdict(list)
for t in sorted(bt):
    rs = bt[t]; ok = [r for r in rs if not r["flags"]]
    ratio = len(ok) / len(rs)
    cls = "잘함" if ratio == 1.0 else ("개선필요" if ratio >= 0.4 else "잘못함")
    cls_count[cls].append(t)
    fc = defaultdict(int)
    for r in rs:
        for x in r["flags"]: fc[x] += 1
    top = ", ".join(f"{k}×{v}" for k, v in sorted(fc.items(), key=lambda x: -x[1])[:3])
    print(f"| {t} | {len(rs)} | {len(ok)} | {len(rs)-len(ok)} | **{cls}** | {top or '—'} |")
print()
for c in ["잘함", "개선필요", "잘못함"]:
    print(f"- {c}: {len(cls_count[c])}종 — {', '.join(cls_count[c]) or '없음'}")

print("\n## 3) 잔상(마지막 검출 이후 이어진 프레임) 분포 — 머리·1인칭\n")
cs = sorted([r["coast"] for r in head if r["coast"] > 0], reverse=True)
print(f"- 잔상 있는 유닛: {len(cs)}/{len(head)}")
if cs:
    print(f"- 길이: 최대 {cs[0]}, 중앙 {cs[len(cs)//2]}, 최소 {cs[-1]} 프레임")
    print(f"- 10프레임 이상: {len([x for x in cs if x>=10])}유닛")

print("\n## 4) 손목 카메라 하자 유형\n")
fc = defaultdict(int)
for r in tk:
    if r["cam"] == "손목":
        for x in r["flags"]: fc[x] += 1
for k, v in sorted(fc.items(), key=lambda x: -x[1]): print(f"- {k}: {v}유닛")

print("\n## 5) 모든 객체 모드 — 실제로 어떤 라벨이 잡혔나\n")
al = [r for r in rows if r["mode"] == "all"]
lab = defaultdict(int)
for r in al:
    for l in r["labels"]: lab[l.split("/")[0] + "/" + l.split("/")[1]] += 1
print(f"- 유닛 {len(al)}개에서 나온 라벨 종류: {len(lab)}")
for k, v in sorted(lab.items(), key=lambda x: -x[1])[:20]:
    print(f"  - {k}: {v}유닛")

print("\n## 6) 대상없음 유닛\n")
for r in tk:
    if "대상없음" in r["flags"]: print(f"- {r['unit']}")
