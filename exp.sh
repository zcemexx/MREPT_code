#!/bin/bash -l
#$ -S /bin/bash
#$ -N eptPhase5
#$ -l h_rt=15:59:00
#$ -l mem=5G
#$ -l tmpfs=16G
#$ -pe smp 8
#$ -t 1-136
#$ -wd /myriadfs/home/zcemexx/Scratch
#$ -o /myriadfs/home/zcemexx/Scratch/logs/
#$ -j y
#$ -M zcemexx@ucl.ac.uk

set -euo pipefail

CODE_DIR="/myriadfs/home/zcemexx/projects/MREPT_code/matlab/code"
MATLAB_SCRIPT="exp.m"
LOG_DIR="/myriadfs/home/zcemexx/Scratch/logs"
mkdir -p "$LOG_DIR"

AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"
TASK_ID="${SGE_TASK_ID:-NA}"
JOB_ID_SAFE="${JOB_ID:-NOJOB}"
RUN_DATE="$(date '+%Y-%m-%d %H:%M:%S')"

if [[ "$AGGREGATE_ONLY" == "1" ]]; then
    export MREPT_AGGREGATE=1
    # 如果误用数组提交聚合模式，仅 task=1 真正执行，其他任务直接退出
    if [[ "$TASK_ID" != "1" ]]; then
        echo "[INFO] AGGREGATE_ONLY=1 and SGE_TASK_ID=$TASK_ID -> skip." 
        exit 0
    fi
    LOG_FILE="${LOG_DIR}/phase5_aggregate_JOB${JOB_ID_SAFE}_TASK${TASK_ID}.log"
else
    export MREPT_AGGREGATE=0
    LOG_FILE="${LOG_DIR}/phase5_task_JOB${JOB_ID_SAFE}_TASK${TASK_ID}.log"
fi

exec >"$LOG_FILE" 2>&1

echo "================================================="
echo "Run time: $RUN_DATE"
echo "Host: $(hostname)"
echo "JOB_ID: ${JOB_ID:-NA}"
echo "SGE_TASK_ID: ${SGE_TASK_ID:-NA}"
echo "NSLOTS: ${NSLOTS:-NA}"
echo "AGGREGATE_ONLY: $AGGREGATE_ONLY"
echo "MREPT_AGGREGATE: ${MREPT_AGGREGATE:-0}"
echo "Log: $LOG_FILE"
echo "================================================="

module purge
module load matlab

if [[ ! -f "$CODE_DIR/$MATLAB_SCRIPT" ]]; then
    echo "[ERROR] MATLAB script not found: $CODE_DIR/$MATLAB_SCRIPT"
    exit 1
fi

matlab -nodisplay -nodesktop -r "cd('$CODE_DIR'); try, run('$MATLAB_SCRIPT'); catch ME, disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"

echo "[DONE] exp.m finished successfully."
