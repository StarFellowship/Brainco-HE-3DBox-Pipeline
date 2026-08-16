#!/bin/bash
#PBS -N RSEG_HSM_HER
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /home/isangmin/rseg_night/logs/pbs.log

# RobotSeg zero-shot: 3유닛 x 3카테고리(arm/gripper/robot)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate robotseg
cd $HOME/RobotSeg/test

for UNIT in PickDrink_ep5_left_high PickDrink_ep5_left_wrist GraspOreo_ep5_left_high; do
  for CAT in arm gripper robot; do
    sed -e "s|^INPUT_DIR = .*|INPUT_DIR = \"$HOME/rseg_night/frames/$UNIT\"|" \
        -e "s|^OUTPUT_ROOT = .*|OUTPUT_ROOT = \"$HOME/rseg_night/out/$UNIT\"|" \
        -e "s|^CATEGORY = .*|CATEGORY = \"$CAT\"|" demo.py > night_variant.py
    echo "=== $UNIT / $CAT $(date)"
    python night_variant.py || echo "FAILED $UNIT $CAT"
  done
done
echo NIGHT_DONE
