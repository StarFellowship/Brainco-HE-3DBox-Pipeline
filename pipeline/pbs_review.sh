#!/bin/bash
#PBS -N G1C4_HSM_3dbbox
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe

# 3D BBox 파이프라인 검증 러너 (PBS job)
# 인자는 qsub -v 로 전달: KIND(bc|he), RND(라운드), SH(샤드), NSH(샤드수)
cd $HOME/task3/pipeline
source $(conda info --base)/etc/profile.d/conda.sh
conda activate task3
export PYTHONPATH=$HOME/task3/pipeline

echo "=== PBS job 시작 ==="
echo "job: $PBS_JOBID  node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "kind=$KIND round=$RND shard=$SH/$NSH"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
ls -d /data2/humanoid_dataset_isangmin >/dev/null 2>&1 && echo "데이터 접근 OK" || echo "데이터 접근 실패!"

if [ "${KIND:-bc}" = "all" ]; then          # bc 전부 -> he 전부 순차
  python $HOME/task3/pipeline/run_review.py bc ${RND:-r1} ${SH:-0} ${NSH:-1}
  RC=$?
  python $HOME/task3/pipeline/run_review.py he ${RND:-r1} 0 1
  RC=$((RC + $?))
else
  python $HOME/task3/pipeline/run_review.py ${KIND:-bc} ${RND:-r1} ${SH:-0} ${NSH:-1}
  RC=$?
fi
echo "=== 종료코드 $RC ==="
exit $RC
