# -*- coding: utf-8 -*-
"""10에피소드 샘플링 결과 자동 스캔.
자동으로 '하자 후보'를 찾아 표로 만든다. 최종 판정은 육안 검증으로 확정한다."""
import json, glob, os, re
from collections import defaultdict

ROOT = os.path.expanduser("~/task3/sample10/s10")
OUT  = os.path.expanduser("~/task3/sample10/분석")
os.makedirs(OUT, exist_ok=True)

# 하자 판정 기준 (자동 스캔용 — 육안 검증 전 후보 추출)
TH_DET   = 0.90   # 대상 물체를 GroundingDINO가 찾은 프레임 비율이 이 아래면 '검출 끊김'
TH_BOX3D = 0.95   # 3D Box가 만들어진 비율이 이 아래면 '3D Box 생성 실패'
TH_TRACK = 0.30   # SAM이 직전 mask를 이어받은 비율이 이 위면 'detection 의존 낮음'
MAX_SIZE = 0.50   # 조작 대상의 최대 변이 이 이상이면 '크기 비현실'


def unit_report(sp):
    st = json.load(open(sp))
    unit = os.path.dirname(sp)
    n = st.get("n_frames", 0)
    flags, detail = [], {}
    if n == 0:
        return None

    # ── 대상 물체 (target)
    tgt = {k: v for k, v in st.get("objects", {}).items() if k.startswith("target/")}
    if not tgt:
        flags.append("대상없음")
    for k, v in tgt.items():
        det, trk, b3 = v.get("det", 0), v.get("track", 0), v.get("box3d", 0)
        fr = max(1, v.get("frames", 0))
        detail[k] = {"det율": round(det / n, 3), "track율": round(trk / n, 3),
                     "3D율": round(b3 / fr, 3), "출현": fr}
        if det / n < TH_DET:                       flags.append("검출끊김")
        if b3 / fr < TH_BOX3D:                     flags.append("3D생성실패")
        if trk / n > TH_TRACK:                     flags.append("추적의존")

    # ── tracker_stats 로 잔상(ghost 꼬리)·크기 확인
    ts = st.get("tracker_stats", {})
    for lb, t in ts.items():
        if not t.get("is_target"):
            continue
        if t.get("coasting_frames", 0) > 0:
            flags.append("잔상꼬리")
            detail.setdefault("_잔상", {})[lb] = t["coasting_frames"]
        sm = t.get("size_median") or []
        if sm and max(sm) > MAX_SIZE:
            flags.append("크기비현실")
            detail.setdefault("_크기", {})[lb] = [round(x, 3) for x in sm]

    # ── 로봇 부위 (팔·손)
    rp = {k: v for k, v in st.get("objects", {}).items() if k.startswith("robot_part/")}
    detail["_로봇부위수"] = len(rp)
    if not rp:
        flags.append("로봇부위미검출")

    return {
        "unit": unit.replace(os.path.expanduser("~/task3/sample10/s10/"), ""),
        "task": st.get("task"), "mode": st.get("mode"),
        "n_frames": n, "flags": sorted(set(flags)), "detail": detail,
        "n_objects": len(st.get("objects", {})),
    }


rows = []
for sp in sorted(glob.glob(f"{ROOT}/*/**/stats.json", recursive=True)):
    r = unit_report(sp)
    if r:
        rows.append(r)

json.dump(rows, open(f"{OUT}/유닛별_자동스캔.json", "w"), ensure_ascii=False, indent=1)

# ── task 분류 (모드 task 기준)
by_task = defaultdict(list)
for r in rows:
    if r["mode"] == "task":
        by_task[r["task"]].append(r)

lines = ["# 10에피소드 샘플링 — 자동 스캔 결과",
         "",
         "판정은 자동 스캔이며, 육안 검증으로 확정해야 한다.",
         f"기준: 검출율<{TH_DET} / 3D생성<{TH_BOX3D} / 추적의존>{TH_TRACK} / 최대변>{MAX_SIZE}m / 잔상꼬리>0",
         "", "## task 분류", "",
         "| task | 유닛 | 무결 | 하자 | 분류 | 주요 하자 |", "|---|---|---|---|---|---|"]

summary = {}
for t in sorted(by_task):
    rs = by_task[t]
    clean = [r for r in rs if not r["flags"]]
    bad   = [r for r in rs if r["flags"]]
    ratio = len(clean) / len(rs)
    cls = "잘함" if ratio == 1.0 else ("개선필요" if ratio >= 0.4 else "잘못함")
    fc = defaultdict(int)
    for r in bad:
        for f in r["flags"]:
            fc[f] += 1
    top = ", ".join(f"{k}×{v}" for k, v in sorted(fc.items(), key=lambda x: -x[1])[:3])
    lines.append(f"| {t} | {len(rs)} | {len(clean)} | {len(bad)} | **{cls}** | {top or '—'} |")
    summary[t] = {"분류": cls, "무결": len(clean), "전체": len(rs), "하자유형": dict(fc)}

# ── 하자 유형 전체 집계
allf = defaultdict(int)
for r in rows:
    if r["mode"] == "task":
        for f in r["flags"]:
            allf[f] += 1
lines += ["", "## 하자 유형 빈도 (task 모드 전체)", "", "| 유형 | 유닛 수 |", "|---|---|"]
for k, v in sorted(allf.items(), key=lambda x: -x[1]):
    lines.append(f"| {k} | {v} |")

# ── 모드 all: 객체를 몇 개나 뽑았나
all_rows = [r for r in rows if r["mode"] == "all"]
if all_rows:
    lines += ["", "## 모든 객체 모드 — 유닛당 검출 객체 수", "",
              "| task | 유닛 | 객체수 중앙값 | 최소 | 최대 |", "|---|---|---|---|---|"]
    bt = defaultdict(list)
    for r in all_rows:
        bt[r["task"]].append(r["n_objects"])
    for t in sorted(bt):
        v = sorted(bt[t])
        lines.append(f"| {t} | {len(v)} | {v[len(v)//2]} | {v[0]} | {v[-1]} |")

open(f"{OUT}/자동스캔_요약.md", "w").write("\n".join(lines))
json.dump(summary, open(f"{OUT}/task분류.json", "w"), ensure_ascii=False, indent=1)
print("\n".join(lines))
print(f"\n총 유닛 {len(rows)} (task {len([r for r in rows if r['mode']=='task'])} / all {len(all_rows)})")
