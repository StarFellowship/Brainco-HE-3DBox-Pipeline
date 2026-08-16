# -*- coding: utf-8 -*-
"""RobotSeg 품질 개선 시도 — box 프롬프트로 양팔을 명시 지정.

자동 모드는 오른팔을 놓쳤다. 여기서는 우리 파이프라인(GroundingDINO)이 찾은
왼팔/오른팔 2D box를 RobotSeg 첫 프레임에 프롬프트로 넣고 영상 전체를 추적시킨다.
산출: 프레임별 오버레이(왼팔=파랑, 오른팔=빨강) + 팔별 마스크 PNG(0/255)
"""
import os, sys, glob, json
import numpy as np

HOME = os.path.expanduser("~")
S10 = f"{HOME}/task3/sample10/s10/task/brainco"
FR2 = f"{HOME}/rseg_night/frames2"
OUT = f"{HOME}/rseg_night/out3_boxprompt"

UNITS = [  # (frames2 이름, s10 결과 폴더)
    ("PickDrink_ep26_left_high",  f"{S10}/PickDrink/ep00026_cam_left_high"),
    ("PickDrink_ep26_left_wrist", f"{S10}/PickDrink/ep00026_cam_left_wrist"),
    ("GraspOreo_ep6_left_high",   f"{S10}/GraspOreo/ep00006_cam_left_high"),
    ("PickCharger_ep28_left_high", f"{S10}/PickCharger/ep00028_cam_left_high"),
]

sys.path.insert(0, f"{HOME}/RobotSeg/test")
os.chdir(f"{HOME}/RobotSeg/test")
import torch
import cv2
from robotseg.build_robotseg import build_robotseg_video_predictor

torch.cuda.set_device(0)
predictor = build_robotseg_video_predictor("../robotseg/configs/robotseg-infer", "../checkpoints/robotseg.pt")

COLORS = {0: (255, 80, 40), 1: (40, 60, 255)}   # BGR: obj0 왼팔=파랑 표기용 반전 주의 → 아래 라벨로 명시


def first_arm_boxes(unit_dir):
    """우리 결과에서 왼팔·오른팔 box2d가 둘 다 있는 첫 프레임을 찾는다."""
    frames = json.load(open(f"{unit_dir}/frames.json"))
    for i, fr in enumerate(frames):
        arms = {}
        for o in fr["objects"]:
            if o["category"] == "robot_part" and o["label"] == "robot arm" and o.get("box2d"):
                arms[o.get("side", "none")] = o["box2d"]
        if "left" in arms and "right" in arms:
            return i, arms
    return None, None


for unit, s10dir in UNITS:
    vdir = f"{FR2}/{unit}"
    if not os.path.isdir(vdir):
        print(f"[스킵] 프레임 없음: {unit}"); continue
    fi, arms = first_arm_boxes(s10dir)
    if arms is None:
        print(f"[스킵] 팔 box 없음: {unit}"); continue
    print(f"[{unit}] 프롬프트 프레임={fi}, 왼팔={arms['left']}, 오른팔={arms['right']}", flush=True)

    od = f"{OUT}/{unit}"
    os.makedirs(f"{od}/overlay", exist_ok=True)
    os.makedirs(f"{od}/mask_left_arm", exist_ok=True)
    os.makedirs(f"{od}/mask_right_arm", exist_ok=True)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=vdir)
        for oid, side in [(0, "left"), (1, "right")]:
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=fi, obj_id=oid,
                box=np.array(arms[side], dtype=np.float32))
        files = sorted(glob.glob(f"{vdir}/*.jpg"))
        for out_fi, obj_ids, masks in predictor.propagate_in_video(state):
            img = cv2.imread(files[out_fi])
            for k, oid in enumerate(obj_ids):
                m = (masks[k] > 0.0).squeeze().cpu().numpy().astype(np.uint8)
                side = "left_arm" if oid == 0 else "right_arm"
                cv2.imwrite(f"{od}/mask_{side}/{out_fi:05d}.png", m * 255)
                col = (255, 120, 40) if oid == 0 else (40, 40, 255)   # 왼팔=파랑, 오른팔=빨강
                img[m > 0] = (0.55 * img[m > 0] + 0.45 * np.array(col)).astype(np.uint8)
            cv2.putText(img, "left arm=blue  right arm=red (box prompt)", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imwrite(f"{od}/overlay/{out_fi:05d}.jpg", img)
    print(f"[{unit}] 완료", flush=True)

print("BOXPROMPT_DONE")
