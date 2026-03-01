#!/bin/bash -l
#$ -S /bin/bash
#$ -N eptSweep
#$ -l h_rt=23:59:00
#$ -l mem=5G
#$ -l tmpfs=16G
#$ -pe smp 32
#$ -t 1-136
#$ -wd /myriadfs/home/zcemexx/Scratch
#$ -o /myriadfs/home/zcemexx/Scratch/logs/
#$ -j y
#$ -m abe
#$ -M zcemexx@ucl.ac.uk

set -euo pipefail

CODE_DIR="${CODE_DIR:-/myriadfs/home/zcemexx/projects/MREPT_code/matlab/code}"
MATLAB_SCRIPT="${MATLAB_SCRIPT:-sweep_radius_metrics_online.m}"
LOG_DIR="${LOG_DIR:-/myriadfs/home/zcemexx/Scratch/logs}"
mkdir -p "$LOG_DIR"

export SWEEP_PHASE5_ROOT="${SWEEP_PHASE5_ROOT:-/myriadfs/home/zcemexx/Scratch/outputs/phase5}"
export SWEEP_GT_ROOT="${SWEEP_GT_ROOT:-/myriadfs/home/zcemexx/Scratch/data/ADEPT_raw}"
export SWEEP_OUT_DIR="${SWEEP_OUT_DIR:-${SWEEP_PHASE5_ROOT}/radiustest}"

TASK_ID="${SGE_TASK_ID:-NA}"
JOB_ID_SAFE="${JOB_ID:-NOJOB}"
RUN_DATE="$(date '+%Y-%m-%d %H:%M:%S')"
LOG_FILE="${LOG_DIR}/sweep_task_JOB${JOB_ID_SAFE}_TASK${TASK_ID}.log"

exec >"$LOG_FILE" 2>&1

echo "================================================="
echo "Run time: $RUN_DATE"
echo "Host: $(hostname)"
echo "JOB_ID: ${JOB_ID:-NA}"
echo "SGE_TASK_ID: ${SGE_TASK_ID:-NA}"
echo "NSLOTS: ${NSLOTS:-NA}"
echo "CODE_DIR: $CODE_DIR"
echo "MATLAB_SCRIPT: $MATLAB_SCRIPT"
echo "SWEEP_PHASE5_ROOT: $SWEEP_PHASE5_ROOT"
echo "SWEEP_GT_ROOT: $SWEEP_GT_ROOT"
echo "SWEEP_OUT_DIR: $SWEEP_OUT_DIR"
echo "Log: $LOG_FILE"
echo "================================================="

module purge
module load matlab

if [[ ! -f "$CODE_DIR/$MATLAB_SCRIPT" ]]; then
    echo "[ERROR] MATLAB script not found: $CODE_DIR/$MATLAB_SCRIPT"
    exit 1
fi
if [[ ! -d "$SWEEP_PHASE5_ROOT" ]]; then
    echo "[ERROR] SWEEP_PHASE5_ROOT not found: $SWEEP_PHASE5_ROOT"
    exit 1
fi
if [[ ! -d "$SWEEP_GT_ROOT" ]]; then
    echo "[ERROR] SWEEP_GT_ROOT not found: $SWEEP_GT_ROOT"
    exit 1
fi
mkdir -p "$SWEEP_OUT_DIR"

matlab -nodisplay -nodesktop -r "cd('$CODE_DIR'); try, run('$MATLAB_SCRIPT'); catch ME, disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"

echo "[DONE] sweep_radius_metrics_online.m finished successfully."
