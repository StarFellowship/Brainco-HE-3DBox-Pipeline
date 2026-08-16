# -*- coding: utf-8 -*-
"""육안 검증용 정지컷 추출.

하자 후보 유닛마다 판정에 필요한 프레임을 ALLOBJ.mp4(박스·mask가 그려진 렌더)에서 뽑는다.
- 꼬리 유닛: 마지막 det 직전 / 마지막 det / 꼬리 중간 / 꼬리 끝  → 꼬리 구간에 물체가 정말 없는지
- 검출끊김 유닛: 에피소드 중간의 가장 긴 추적(track) 구간 중간 2장 + det 1장 → 빈 공간 박스, 엉뚱한 물체 여부
- 대상없음 유닛: 초반·중간 1장씩 → 물체가 정말 시야 밖인지
결과: 분석/스틸/<unit별 폴더>/*.jpg + index.json (판정 안내 포함)
"""
import json, glob, os, subprocess
from collections import defaultdict

ROOT = os.path.expanduser("~/task3/sample10/s10")
OUT  = os.path.expanduser("~/task3/sample10/분석/스틸")
os.makedirs(OUT, exist_ok=True)
FPS = 10.0
TH_DET = 0.90

trim = {r["unit"]: r for r in json.load(open(os.path.expanduser("~/task3/sample10/분석/trim_결과_유닛별.json")))}

index = []

def grab(mp4, fidx, dst):
    if os.path.exists(dst): return True
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{fidx/FPS:.3f}",
                        "-i", mp4, "-frames:v", "1", "-q:v", "3", dst])
    return r.returncode == 0

for sp in sorted(glob.glob(f"{ROOT}/task/**/stats.json", recursive=True)):
    ud = os.path.dirname(sp)
    unit = ud.replace(ROOT + "/", "")
    st = json.load(open(sp))
    n = st.get("n_frames", 0)
    mp4 = f"{ud}/ALLOBJ.mp4"
    if not n or not os.path.exists(mp4):
        continue

    tr = trim.get(unit, {})
    tgt = {k: v for k, v in st.get("objects", {}).items() if k.startswith("target/")}
    shots = []          # (태그, 프레임번호)

    # ── 유형별 프레임 선택
    if tr.get("tail"):
        for lb, tl in tr["tail"].items():
            ld = tr["last_det"].get(lb, -1)
            if ld < 0: continue
            shots += [("det직전", max(0, ld - 2)), ("마지막det", ld),
                      ("꼬리중간", min(n - 1, ld + max(1, tl // 2))), ("꼬리끝", min(n - 1, ld + tl))]
    low_det = any(v.get("det", 0) / n < TH_DET for v in tgt.values())
    if low_det or not tgt:
        # frames.json에서 대상의 가장 긴 track 연속 구간을 찾는다
        frames = json.load(open(f"{ud}/frames.json"))
        run_best, run_cur, run_start, best_start = 0, 0, 0, 0
        det_frame = -1
        for i, fr in enumerate(frames):
            is_track = any(o["category"] == "target" and o["src"] == "track" for o in fr["objects"])
            is_det   = any(o["category"] == "target" and o["src"] == "det"   for o in fr["objects"])
            if is_det: det_frame = i
            if is_track:
                if run_cur == 0: run_start = i
                run_cur += 1
                if run_cur > run_best: run_best, best_start = run_cur, run_start
            else:
                run_cur = 0
        if run_best >= 5:
            shots += [("추적구간1", best_start + run_best // 3), ("추적구간2", best_start + 2 * run_best // 3)]
        if det_frame >= 0:
            shots.append(("det예시", det_frame))
        if not tgt:
            shots += [("초반", min(5, n - 1)), ("중간", n // 2)]

    if not shots:
        continue
    sd = f"{OUT}/{unit.replace('/', '__')}"
    os.makedirs(sd, exist_ok=True)
    done = []
    for tag, fi in dict.fromkeys(shots):        # 중복 제거, 순서 유지
        dst = f"{sd}/{tag}_f{fi:04d}.jpg"
        if grab(mp4, fi, dst):
            done.append({"tag": tag, "frame": fi, "file": os.path.basename(dst)})
    index.append({"unit": unit, "task": st.get("task"), "n_frames": n,
                  "tail": tr.get("tail", {}), "last_det": tr.get("last_det", {}),
                  "n_det": tr.get("n_det", {}),
                  "target_labels": sorted(tgt.keys()), "shots": done})

json.dump(index, open(f"{OUT}/index.json", "w"), ensure_ascii=False, indent=1)
print(f"[완료] {len(index)}유닛, 스틸 {sum(len(x['shots']) for x in index)}장 → {OUT}")
