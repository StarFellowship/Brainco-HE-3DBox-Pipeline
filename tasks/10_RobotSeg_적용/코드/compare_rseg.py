# -*- coding: utf-8 -*-
"""RobotSeg vs 현행 방식 — 같은 프레임 나란히 비교 이미지 생성 (CPU).

왼쪽: 원본 프레임에 우리 방식(SAM 2.1 + 글자 프롬프트)의 로봇 팔·손 mask를 빨강으로 칠함
오른쪽: RobotSeg 자동 모드(arm 카테고리)의 오버레이(파랑)
"""
import os, sys, glob, json
import numpy as np
import cv2

HOME = os.path.expanduser("~")
sys.path.insert(0, f"{HOME}/task3/pipeline")
from multiobj import rle_to_mask

S10 = f"{HOME}/task3/sample10/s10/task/brainco"
FR2 = f"{HOME}/rseg_night/frames2"
AUTO = f"{HOME}/rseg_night/out2"
OUT = f"{HOME}/rseg_night/비교_현행vs자동"

UNITS = [
    ("PickDrink_ep26_left_high",  f"{S10}/PickDrink/ep00026_cam_left_high"),
    ("PickDrink_ep26_left_wrist", f"{S10}/PickDrink/ep00026_cam_left_wrist"),
    ("GraspOreo_ep6_left_high",   f"{S10}/GraspOreo/ep00006_cam_left_high"),
    ("PickCharger_ep28_left_high", f"{S10}/PickCharger/ep00028_cam_left_high"),
]
PICK = list(range(10, 220, 20))          # 프레임 12장 표본


def put(img, txt):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


for unit, s10dir in UNITS:
    frames = json.load(open(f"{s10dir}/frames.json"))
    files = sorted(glob.glob(f"{FR2}/{unit}/*.jpg"))
    od = f"{OUT}/{unit}"; os.makedirs(od, exist_ok=True)
    made = 0
    for i in PICK:
        if i >= len(frames) or i >= len(files): break
        base = cv2.imread(files[i])
        if base is None: continue
        ours = base.copy()
        n_part = 0
        for o in frames[i]["objects"]:
            if o["category"] == "robot_part" and o.get("mask_rle"):
                m = rle_to_mask(o["mask_rle"])
                if m.shape[:2] != ours.shape[:2]:
                    m = cv2.resize(m.astype(np.uint8), (ours.shape[1], ours.shape[0]))
                ours[m > 0] = (0.55 * ours[m > 0] + 0.45 * np.array((40, 40, 255))).astype(np.uint8)
                n_part += 1
        ours = put(ours, f"our method: SAM2.1+text prompt ({n_part} parts, red)")
        ap = f"{AUTO}/{unit}/arm/{i+1:05d}.jpg"
        auto = cv2.imread(ap)
        auto = put(auto if auto is not None else base.copy(),
                   "RobotSeg auto arm (blue)" if auto is not None else "RobotSeg: no output")
        cv2.imwrite(f"{od}/f{i:04d}.jpg", np.hstack([ours, auto]))
        made += 1
    print(f"[{unit}] {made}장")
print("COMPARE_DONE")
