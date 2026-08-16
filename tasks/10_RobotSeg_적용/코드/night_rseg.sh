#!/bin/bash
# RobotSeg zero-shot 야간 파일럿 — 프레임 추출(CPU, 로그인 노드) 후 PBS job 제출
set -x
DATA=/data2/humanoid_dataset_isangmin
OUT=$HOME/rseg_night
mkdir -p $OUT/frames $OUT/logs $OUT/out

prep() { # $1 데이터셋 $2 캠 $3 파일 $4 유닛명
  d=$OUT/frames/$4
  [ -d "$d" ] && [ "$(ls "$d" 2>/dev/null | wc -l)" -gt 10 ] && return
  mkdir -p "$d"
  ffmpeg -y -i "$DATA/$1/videos/observation.images.$2/chunk-000/$3" \
    -vf fps=6 -frames:v 180 -q:v 2 "$d/%05d.jpg" </dev/null
}

# 3인칭 머리캠 2유닛 + 1인칭 손목캠 1유닛 (도메인 갭 실측용 구성)
prep G1_Brainco_PickDrink_Dataset cam_left_high  file-005.mp4 PickDrink_ep5_left_high
prep G1_Brainco_PickDrink_Dataset cam_left_wrist file-005.mp4 PickDrink_ep5_left_wrist
prep G1_Brainco_GraspOreo_Dataset cam_left_high  file-005.mp4 GraspOreo_ep5_left_high

for d in $OUT/frames/*/; do echo "$d: $(ls "$d" | wc -l) frames"; done
qsub -q pleiades1 $HOME/rseg_night/pbs_rseg.sh
qstat -u isangmin
