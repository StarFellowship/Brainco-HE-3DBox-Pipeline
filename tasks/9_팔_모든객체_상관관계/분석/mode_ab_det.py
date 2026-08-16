# -*- coding: utf-8 -*-
"""같은 영상에서 '대상만 모드'와 '모든 객체 모드'의 대상 검출률 비교.
검출 문구가 길어지면 대상 검출이 흔들리는지 실측."""
import json, glob, os
from collections import defaultdict

ROOT = os.path.expanduser("~/task3/sample10/s10")
pair = defaultdict(dict)
for mode in ["task", "all"]:
    for sp in glob.glob(f"{ROOT}/{mode}/brainco/*/*/stats.json"):
        st = json.load(open(sp))
        unit = "/".join(sp.split("/")[-3:-1])
        n = st.get("n_frames", 0)
        if not n:
            continue
        det = sum(v.get("det", 0) for k, v in st.get("objects", {}).items()
                  if k.startswith("target/"))
        pair[unit][mode] = det / n

both = {u: v for u, v in pair.items() if "task" in v and "all" in v}
diffs = [v["all"] - v["task"] for v in both.values()]
diffs.sort()
n = len(diffs)
worse = sum(1 for d in diffs if d < -0.05)
better = sum(1 for d in diffs if d > 0.05)
print(f"양쪽 모드 다 있는 영상: {n}")
print(f"대상 검출률 평균: 대상만 {sum(v['task'] for v in both.values())/n:.3f} / "
      f"모든객체 {sum(v['all'] for v in both.values())/n:.3f}")
print(f"차이 중앙값 {diffs[n//2]:+.3f}, 5%p 이상 나빠짐 {worse}개, 좋아짐 {better}개")
print("나빠진 상위 5:")
for u, v in sorted(both.items(), key=lambda x: x[1]["all"] - x[1]["task"])[:5]:
    print(f"  {u}: {v['task']:.2f} -> {v['all']:.2f}")
