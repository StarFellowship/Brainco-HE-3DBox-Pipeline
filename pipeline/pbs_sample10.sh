#!/bin/bash
#PBS -N S10_HSM_HER
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

# 10에피소드 샘플 러너 (PBS job) — pbs_review.sh 를 복사·확장.
# 3분할 제출 전제: SPLIT=bc1|bc2|he 각 1 GPU (3090 노드 — 제출 시 -q pleiades1 또는 pleiades3).
#
#   qsub -q pleiades1 -v SPLIT=bc1,RND=s10,MODE=all pbs_sample10.sh
#   qsub -q pleiades3 -v SPLIT=bc2,RND=s10,MODE=all pbs_sample10.sh
#   qsub -q pleiades1 -v SPLIT=he,RND=s10,MODE=all  pbs_sample10.sh
#
# job명은 -N 인자로 변경 (기본 S10_HSM_HER):  qsub -N S10_BC1 -q pleiades1 -v ... pbs_sample10.sh
# 스모크(유닛 2개 x 60프레임):                qsub -q pleiades1 -v SPLIT=he,RND=smoke,MODE=all,SMOKE=1 pbs_sample10.sh
cd $HOME/task3/pipeline
source $(conda info --base)/etc/profile.d/conda.sh
conda activate task3
export PYTHONPATH=$HOME/task3/pipeline

export SPLIT=${SPLIT:-bc1}
export RND=${RND:-s10}
export MODE=${MODE:-all}
export SMOKE=${SMOKE:-0}

echo "=== PBS job 시작 ==="
echo "job: $PBS_JOBID  node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "split=$SPLIT rnd=$RND mode=$MODE smoke=$SMOKE"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
ls -d /data2/humanoid_dataset_isangmin >/dev/null 2>&1 && echo "데이터 접근 OK" || echo "데이터 접근 실패!"

python $HOME/task3/pipeline/run_sample10.py
RC=$?

# CPU 후처리 — 객체 쌍 상관관계 (GPU 불필요, 실패해도 본 산출물은 보존)
python $HOME/task3/pipeline/relations.py --tree $HOME/task3/sample10/$RND/$MODE || true

echo "=== 종료코드 $RC ==="
exit $RC
