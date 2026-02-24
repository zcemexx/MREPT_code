#!/bin/bash -l
#$ -S /bin/bash
#$ -N eptRecon
#$ -l h_rt=12:59:00
#$ -l mem=5G
#$ -l tmpfs=16G
#$ -pe smp 8
#$ -wd /myriadfs/home/zcemexx/Scratch
#$ -o /myriadfs/home/zcemexx/Scratch/logs/
#$ -j y
#$ -m abe
#$ -M zcemexx@ucl.ac.uk

set -euo pipefail

CODE_DIR="${CODE_DIR:-/home/zcemexx/projects/MREPT_code/matlab/code}"
MATLAB_SCRIPT="${MATLAB_SCRIPT:-recon.m}"
LOG_DIR="${LOG_DIR:-/home/zcemexx/Scratch/logs}"
mkdir -p "$LOG_DIR"

PHASE5_ROOT="${PHASE5_ROOT:-/home/zcemexx/Scratch/outputs/phase5}"
RADIUS_PRED_ROOT="${RADIUS_PRED_ROOT:-/home/zcemexx/Scratch/outputs/phase5}"
RECON_INPLACE_IO="${RECON_INPLACE_IO:-true}"
RECON_OUT_ROOT="${RECON_OUT_ROOT:-/home/zcemexx/Scratch/outputs/phase5_sigma_recon}"

TASK_ID="${SGE_TASK_ID:-NA}"
JOB_ID_SAFE="${JOB_ID:-NOJOB}"
RUN_DATE="$(date '+%Y-%m-%d %H:%M:%S')"
LOG_FILE="${LOG_DIR}/recon_JOB${JOB_ID_SAFE}_TASK${TASK_ID}.log"

exec >"$LOG_FILE" 2>&1

echo "================================================="
echo "Run time: $RUN_DATE"
echo "Host: $(hostname)"
echo "JOB_ID: ${JOB_ID:-NA}"
echo "SGE_TASK_ID: ${SGE_TASK_ID:-NA}"
echo "NSLOTS: ${NSLOTS:-NA}"
echo "CODE_DIR: $CODE_DIR"
echo "MATLAB_SCRIPT: $MATLAB_SCRIPT"
echo "PHASE5_ROOT: $PHASE5_ROOT"
echo "RADIUS_PRED_ROOT: $RADIUS_PRED_ROOT"
echo "RECON_INPLACE_IO: $RECON_INPLACE_IO"
echo "RECON_OUT_ROOT: $RECON_OUT_ROOT"
echo "Log: $LOG_FILE"
echo "================================================="

module purge
module load matlab

if [[ ! -f "$CODE_DIR/$MATLAB_SCRIPT" ]]; then
    echo "[ERROR] MATLAB script not found: $CODE_DIR/$MATLAB_SCRIPT"
    exit 1
fi
if [[ ! -d "$PHASE5_ROOT" ]]; then
    echo "[ERROR] PHASE5_ROOT not found: $PHASE5_ROOT"
    exit 1
fi
if [[ ! -d "$RADIUS_PRED_ROOT" ]]; then
    echo "[ERROR] RADIUS_PRED_ROOT not found: $RADIUS_PRED_ROOT"
    exit 1
fi

export PHASE5_ROOT
export RADIUS_PRED_ROOT
export RECON_INPLACE_IO
export RECON_OUT_ROOT

matlab -nodisplay -nodesktop -r "cd('$CODE_DIR'); try, run('$MATLAB_SCRIPT'); catch ME, disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"

echo "[DONE] ${MATLAB_SCRIPT} finished successfully."
