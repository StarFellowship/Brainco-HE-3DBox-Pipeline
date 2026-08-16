# -*- coding: utf-8 -*-
"""ghost 꼬리 제거를 10에피소드 샘플 결과에 적용하고, 제거 전후를 집계한다.

규칙 (기존 postprocess.py ghosttrim과 동일한 원리, 새 저장 형식에 맞춰 재구현):
- 대상 물체(target)마다, GroundingDINO가 실제로 찾은(src=="det") 마지막 프레임을 찾는다.
- 그 프레임 이후에 SAM 2.1이 이어서 추적(src=="track")만 하고 있는 박스는 전부 제거한다.
  (물체가 화면에서 사라진 뒤 추적 창만 남는 것이 ghost 꼬리이기 때문)
- 단, det 횟수가 8회 미만인 라벨은 신뢰할 수 없으므로 추적 구간 전체를 제거한다.
- 에피소드 중간의 추적 구간은 이후 det로 회복되는 정상 구간이므로 건드리지 않는다.

산출:
- 유닛마다 frames_trimmed.json (제거 반영본. 원본 frames.json은 그대로 둔다)
- 전체 집계 trim_summary.json / trim_summary.md
"""
import json, glob, os
from collections import defaultdict

ROOT = os.path.expanduser("~/task3/sample10/s10")
OUT  = os.path.expanduser("~/task3/sample10/분석")
os.makedirs(OUT, exist_ok=True)
CONFIRM_N = 8          # 이 횟수 이상 det 된 라벨만 '확립'으로 본다


def trim_unit(unit_dir):
    fp = f"{unit_dir}/frames.json"
    frames = json.load(open(fp))
    n = len(frames)
    # 대상 라벨별 det 프레임 목록
    det_at = defaultdict(list)
    for i, fr in enumerate(frames):
        for o in fr["objects"]:
            if o["category"] == "target" and o["src"] == "det":
                det_at[o["label"]].append(i)

    removed = defaultdict(int)      # label -> 제거 프레임 수
    tail_len = {}                   # label -> 꼬리 길이
    for i, fr in enumerate(frames):
        keep = []
        for o in fr["objects"]:
            if o["category"] != "target" or o["src"] != "track":
                keep.append(o); continue
            lb = o["label"]
            dets = det_at.get(lb, [])
            if len(dets) < CONFIRM_N:            # 확립 안 된 라벨: 추적 전부 제거
                removed[lb] += 1; continue
            if i > dets[-1]:                     # 마지막 det 이후의 추적 = ghost 꼬리
                removed[lb] += 1
                tail_len[lb] = tail_len.get(lb, 0) + 1
                continue
            keep.append(o)                       # 중간 추적 구간은 유지
        fr["objects"] = keep

    json.dump(frames, open(f"{unit_dir}/frames_trimmed.json", "w"), ensure_ascii=False)
    return {"unit": unit_dir.replace(ROOT + "/", ""), "n_frames": n,
            "removed": dict(removed), "tail": dict(tail_len),
            "last_det": {lb: (v[-1] if v else -1) for lb, v in det_at.items()},
            "n_det": {lb: len(v) for lb, v in det_at.items()}}


rows = []
for sp in sorted(glob.glob(f"{ROOT}/*/**/frames.json", recursive=True)):
    rows.append(trim_unit(os.path.dirname(sp)))

json.dump(rows, open(f"{OUT}/trim_결과_유닛별.json", "w"), ensure_ascii=False, indent=1)

# ── 집계 (task 모드만; 카메라 구분)
def camkind(u):
    return "손목" if "wrist" in u else ("머리" if "high" in u else "1인칭")

tk = [r for r in rows if "/task/" in "/" + r["unit"]]
with_tail = [r for r in tk if r["tail"]]
lines = ["# ghost 꼬리 제거 — 적용 결과", "",
         f"- 적용 유닛: {len(tk)} (task 모드)",
         f"- 꼬리가 있었던 유닛: {len(with_tail)}",
         f"- 제거된 프레임 합계: {sum(sum(r['tail'].values()) for r in with_tail)}",
         "", "| 카메라 | 유닛 | 꼬리 있던 유닛 | 제거 프레임 |", "|---|---|---|---|"]
bc = defaultdict(lambda: [0, 0, 0])
for r in tk:
    c = camkind(r["unit"]); bc[c][0] += 1
    if r["tail"]:
        bc[c][1] += 1; bc[c][2] += sum(r["tail"].values())
for c in ["머리", "손목", "1인칭"]:
    if bc[c][0]:
        lines.append(f"| {c} | {bc[c][0]} | {bc[c][1]} | {bc[c][2]} |")

lines += ["", "## 꼬리가 길었던 유닛 상위 15", "", "| 유닛 | 라벨 | 꼬리 길이 | 마지막 det | 총 프레임 |", "|---|---|---|---|---|"]
flat = []
for r in with_tail:
    for lb, tl in r["tail"].items():
        flat.append((tl, r["unit"], lb, r["last_det"].get(lb, -1), r["n_frames"]))
for tl, u, lb, ld, nf in sorted(flat, reverse=True)[:15]:
    lines.append(f"| {u} | {lb} | {tl} | {ld} | {nf} |")

open(f"{OUT}/trim_요약.md", "w").write("\n".join(lines))
print("\n".join(lines[:30]))
print(f"\n[완료] frames_trimmed.json {len(rows)}개 생성, 집계는 {OUT}/")
