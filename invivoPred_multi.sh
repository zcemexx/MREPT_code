#!/bin/bash -l
#$ -S /bin/bash
#$ -N invPredR
#$ -l h_rt=11:59:00
#$ -l mem=8G
#$ -l tmpfs=8G
#$ -pe smp 16
#$ -wd /home/zcemexx/Scratch
#$ -o /home/zcemexx/Scratch/logs/
#$ -j y
#$ -m abe
#$ -M zcemexx@ucl.ac.uk

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/home/zcemexx/Scratch/data/invivo/3T_in-vivo_3DGRE_Head_Neck_repeatibility}"
CODE_DIR="${CODE_DIR:-/home/zcemexx/projects/MREPT_code/matlab/code}"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
LOG_DIR="${LOG_DIR:-/home/zcemexx/Scratch/logs}"

# Space-separated list of case stems, for example:
# CASES="M1_SNR030 M2_SNR030 M5_SNR030"
CASES="${CASES:-}"
CASES_FILE="${CASES_FILE:-}"

ESTIMATE_NOISE="${ESTIMATE_NOISE:-true}"
B0_T="${B0_T:-3}"
VOXEL_MM="${VOXEL_MM:-0.85938,0.85938,1.5}"
RADIUS_MIN="${RADIUS_MIN:-1}"
RADIUS_MAX="${RADIUS_MAX:-30}"

TASK_ID="${SGE_TASK_ID:-${INVIVO_TASK_ID:-}}"
JOB_ID_SAFE="${JOB_ID:-NOJOB}"
RUN_DATE="$(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/invivoPred_JOB${JOB_ID_SAFE}_TASK${TASK_ID:-NA}.log"
exec >"$LOG_FILE" 2>&1

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "[ERROR] DATA_ROOT not found: $DATA_ROOT"
    exit 1
fi

TMP_BASE="${TMPDIR:-/tmp}"
if [[ ! -d "$TMP_BASE" ]]; then
    TMP_BASE="/tmp"
fi

TASK_LIST="${TMP_BASE}/invivo_pred_task_list_${JOB_ID_SAFE}_$$.txt"
: > "$TASK_LIST"
trap 'rm -f "$TASK_LIST"' EXIT

append_case() {
    local case_id="$1"
    local radius_nii=""

    if [[ -z "$case_id" ]]; then
        return 0
    fi

    for cand in \
        "$DATA_ROOT/${case_id}.nii.gz" \
        "$DATA_ROOT/${case_id}.nii"
    do
        if [[ -f "$cand" ]]; then
            radius_nii="$cand"
            break
        fi
    done

    if [[ -z "$radius_nii" ]]; then
        echo "[WARN] Skip case without radius map: $case_id"
        return 0
    fi

    printf '%s|%s|%s\n' \
        "$case_id" \
        "$radius_nii" \
        "$DATA_ROOT/cond_pred_${case_id}.nii.gz" >> "$TASK_LIST"
}

if [[ -n "$CASES_FILE" ]]; then
    if [[ ! -f "$CASES_FILE" ]]; then
        echo "[ERROR] CASES_FILE not found: $CASES_FILE"
        exit 1
    fi
    while IFS= read -r line; do
        case_id="${line%%#*}"
        case_id="$(printf '%s' "$case_id" | xargs)"
        append_case "$case_id"
    done < "$CASES_FILE"
elif [[ -n "$CASES" ]]; then
    for case_id in $CASES; do
        append_case "$case_id"
    done
else
    echo "[ERROR] Provide CASES or CASES_FILE."
    echo "        Example:"
    echo "        qsub -v CASES='M1_SNR030 M2_SNR030 M5_SNR030' $SCRIPT_DIR/invivoPred_multi.sh"
    exit 1
fi

TASK_COUNT="$(wc -l < "$TASK_LIST" | tr -d ' ')"
if [[ "$TASK_COUNT" -eq 0 ]]; then
    echo "[ERROR] No runnable cases found."
    echo "        DATA_ROOT=$DATA_ROOT"
    echo "        CASES=${CASES:-<unset>}"
    echo "        CASES_FILE=${CASES_FILE:-<unset>}"
    exit 1
fi

echo "================================================="
echo "Run time: $RUN_DATE"
echo "Host: $(hostname)"
echo "JOB_ID: ${JOB_ID:-NA}"
echo "SGE_TASK_ID: ${SGE_TASK_ID:-NA}"
echo "NSLOTS: ${NSLOTS:-NA}"
echo "DATA_ROOT: $DATA_ROOT"
echo "CODE_DIR: $CODE_DIR"
echo "MATLAB_BIN: $MATLAB_BIN"
echo "Task count: $TASK_COUNT"
echo "Log: $LOG_FILE"
echo "================================================="
sed -n '1,120p' "$TASK_LIST"

if [[ -z "$TASK_ID" ]]; then
    echo "[INFO] No SGE_TASK_ID/INVIVO_TASK_ID provided."
    echo "[INFO] Submit as array, for example:"
    echo "qsub -v CASES='${CASES:-M1_SNR030 M2_SNR030}' -t 1-${TASK_COUNT} $SCRIPT_DIR/invivoPred_multi.sh"
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
IFS='|' read -r case_id radius_nii out_nii <<< "$line"

echo "[RUN ] TASK $TASK_ID/$TASK_COUNT -> $case_id"
echo "      radius : $radius_nii"
echo "      out    : $out_nii"

module purge
module load matlab/full/r2023a/9.14

DATA_ROOT="$DATA_ROOT" \
CODE_DIR="$CODE_DIR" \
MATLAB_BIN="$MATLAB_BIN" \
RADIUS_NII="$radius_nii" \
OUT_NII="$out_nii" \
ESTIMATE_NOISE="$ESTIMATE_NOISE" \
B0_T="$B0_T" \
VOXEL_MM="$VOXEL_MM" \
RADIUS_MIN="$RADIUS_MIN" \
RADIUS_MAX="$RADIUS_MAX" \
bash "$SCRIPT_DIR/invivo.sh"

echo "[DONE] Task $TASK_ID finished successfully."
