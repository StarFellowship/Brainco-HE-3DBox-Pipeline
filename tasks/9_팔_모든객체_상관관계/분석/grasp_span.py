# -*- coding: utf-8 -*-
"""상관관계 정보의 활용 검증: 대상-그리퍼 표면 간격(gap)<5cm 연속 구간 = '잡고 있는 구간' 후보."""
import json, os, glob

res = []
for up in sorted(glob.glob(os.path.expanduser(
        "~/task3/sample10/s10/task/brainco/*/ep*_cam_left_high/relations.json"))):
    unit = "/".join(up.split("/")[-3:-1])
    d = json.load(open(up))
    spans = {}
    for fr in d["frames"]:
        for p in fr["pairs"]:
            a, b, g = p["a"], p["b"], p.get("gap")
            if g is None:
                continue
            if "gripper" in b and "robot" not in a and a not in ("plate", "table"):
                if g < 0.05:
                    spans.setdefault(a + "~" + b, []).append(fr["frame"])
    out = []
    for k, fs in spans.items():
        runs = []
        s = prev = fs[0]
        for f in fs[1:]:
            if f - prev > 3:
                runs.append((s, prev)); s = f
            prev = f
        runs.append((s, prev))
        runs = [r for r in runs if r[1] - r[0] >= 5]
        if runs:
            out.append((k, runs[:3]))
    res.append((unit, out))

ok = [u for u, o in res if o]
print("머리좌 카메라 %d개 영상 중 접촉 구간이 잡힌 것: %d개" % (len(res), len(ok)))
for u, o in res:
    if o:
        parts = []
        for k, runs in o:
            seg = ", ".join("%d~%d" % r for r in runs)
            parts.append("%s: %s" % (k, seg))
        print("  [%s] %s" % (u, " | ".join(parts)))
