#!/bin/bash -l
#$ -S /bin/bash
#$ -N fixcond17
#$ -l h_rt=12:59:00
#$ -l mem=8G
#$ -l tmpfs=8G
#$ -pe smp 8
#$ -wd /home/zcemexx/Scratch
#$ -o /home/zcemexx/Scratch/logs/
#$ -j y
#$ -m abe
#$ -M zcemexx@ucl.ac.uk

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CODE_DIR="${CODE_DIR:-/home/zcemexx/projects/MREPT_code/matlab/code}"
LOG_DIR="${LOG_DIR:-/home/zcemexx/Scratch/logs}"
PHASE5_ROOT="${PHASE5_ROOT:-/home/zcemexx/Scratch/outputs/phase5}"

mkdir -p "$LOG_DIR"

TASK_ID="${SGE_TASK_ID:-${FIXCOND_TASK_ID:-}}"
JOB_ID_SAFE="${JOB_ID:-NOJOB}"
RUN_DATE="$(date '+%Y-%m-%d %H:%M:%S')"
LOG_FILE="${LOG_DIR}/fixcond17_JOB${JOB_ID_SAFE}_TASK${TASK_ID:-NA}.log"
exec >"$LOG_FILE" 2>&1

if [[ ! -d "$PHASE5_ROOT" ]]; then
    echo "[ERROR] PHASE5_ROOT not found: $PHASE5_ROOT"
    exit 1
fi

TMP_BASE="${TMPDIR:-/tmp}"
if [[ ! -d "$TMP_BASE" ]]; then
    TMP_BASE="/tmp"
fi
TASK_LIST="${TMP_BASE}/fixcond17_task_list_${JOB_ID_SAFE}_$$.txt"
: > "$TASK_LIST"
trap 'rm -f "$TASK_LIST"' EXIT

while IFS= read -r snr_path; do
    snr_tag="$(basename "$snr_path")"
    mat_path="$snr_path/noisy_phase_${snr_tag}.mat"
    if [[ -f "$mat_path" ]]; then
        printf '%s\n' "$snr_path" >> "$TASK_LIST"
    fi
done < <(find "$PHASE5_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'SNR*' | sort -V)

TASK_COUNT="$(wc -l < "$TASK_LIST" | tr -d ' ')"
if [[ "$TASK_COUNT" -eq 0 ]]; then
    echo "[ERROR] No runnable tasks found under: $PHASE5_ROOT"
    exit 1
fi

echo "================================================="
echo "Run time: $RUN_DATE"
echo "Host: $(hostname)"
echo "JOB_ID: ${JOB_ID:-NA}"
echo "SGE_TASK_ID: ${SGE_TASK_ID:-NA}"
echo "NSLOTS: ${NSLOTS:-NA}"
echo "CODE_DIR: $CODE_DIR"
echo "PHASE5_ROOT: $PHASE5_ROOT"
echo "Task count: $TASK_COUNT"
echo "Log: $LOG_FILE"
echo "================================================="

if [[ -z "$TASK_ID" ]]; then
    echo "[INFO] No SGE_TASK_ID/FIXCOND_TASK_ID provided."
    echo "[INFO] Submit as array, e.g.:"
    echo "qsub -t 1-${TASK_COUNT} $SCRIPT_DIR/fixcond17.sh"
    exit 0
fi

if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid TASK_ID: $TASK_ID"
    exit 1
fi
if (( TASK_ID < 1 || TASK_ID > TASK_COUNT )); then
    echo "[ERROR] TASK_ID $TASK_ID out of range 1..$TASK_COUNT"
    exit 1
fi

line="$(sed -n "${TASK_ID}p" "$TASK_LIST")"
case_name="$(basename "$(dirname "$line")")"
snr_tag="$(basename "$line")"
echo "[RUN ] TASK $TASK_ID/$TASK_COUNT -> $case_name $snr_tag"
echo "      input : $line/noisy_phase_${snr_tag}.mat"
echo "      output: $line/${case_name}_${snr_tag}_fixcond.nii.gz"

module purge
module load matlab/full/r2023a/9.14

export PHASE5_ROOT
export SGE_TASK_ID="$TASK_ID"

matlab -nodisplay -nodesktop -r "cd('$CODE_DIR'); addpath(pwd); addpath(fullfile(fileparts(pwd),'functions')); addpath(genpath(fullfile(fileparts(pwd),'toolboxes'))); rehash; try, recon_fixcond17; catch ME, disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"

echo "[DONE] Task $TASK_ID finished successfully."
