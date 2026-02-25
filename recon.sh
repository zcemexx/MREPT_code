#!/bin/bash -l
#$ -S /bin/bash
#$ -N eptRecon
#$ -l h_rt=15:59:00
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
LOG_DIR="${LOG_DIR:-/home/zcemexx/Scratch/logs}"
mkdir -p "$LOG_DIR"

PHASE5_ROOT="${PHASE5_ROOT:-/home/zcemexx/Scratch/outputs/phase5}"
RADIUS_PRED_ROOT="${RADIUS_PRED_ROOT:-/home/zcemexx/Scratch/outputs/phase5}"
RECON_INPLACE_IO="${RECON_INPLACE_IO:-true}"
RECON_OUT_ROOT="${RECON_OUT_ROOT:-/home/zcemexx/Scratch/outputs/phase5_sigma_recon}"

TASK_ID="${SGE_TASK_ID:-${RECON_TASK_ID:-}}"
JOB_ID_SAFE="${JOB_ID:-NOJOB}"
RUN_DATE="$(date '+%Y-%m-%d %H:%M:%S')"
LOG_FILE="${LOG_DIR}/recon_JOB${JOB_ID_SAFE}_TASK${TASK_ID:-NA}.log"

exec >"$LOG_FILE" 2>&1

find_radius_file() {
    local snr_path="$1"
    local pred_root="$2"
    local case_name="$3"
    local snr_tag="$4"

    local candidates=(
        "$snr_path/${case_name}_${snr_tag}.nii.gz"
        "$snr_path/${case_name}_${snr_tag}.nii"
        "$snr_path/radiusmap.nii.gz"
        "$snr_path/radiusmap.nii"
        "$pred_root/${case_name}_${snr_tag}.nii.gz"
        "$pred_root/${case_name}_${snr_tag}.nii"
        "$pred_root/${case_name}/${case_name}_${snr_tag}.nii.gz"
        "$pred_root/${case_name}/${case_name}_${snr_tag}.nii"
    )

    local p
    for p in "${candidates[@]}"; do
        if [[ -f "$p" ]]; then
            printf '%s\n' "$p"
            return 0
        fi
    done

    p="$(compgen -G "$pred_root/${case_name}_${snr_tag}*.nii.gz" | head -n 1 || true)"
    if [[ -n "$p" ]]; then
        printf '%s\n' "$p"
        return 0
    fi

    p="$(compgen -G "$pred_root/${case_name}_${snr_tag}*.nii" | head -n 1 || true)"
    if [[ -n "$p" ]]; then
        printf '%s\n' "$p"
        return 0
    fi

    return 1
}

TASK_LIST="${TMPDIR:-/tmp}/recon_task_list_${JOB_ID_SAFE}_$$.txt"
trap 'rm -f "$TASK_LIST"' EXIT

if [[ ! -d "$PHASE5_ROOT" ]]; then
    echo "[ERROR] PHASE5_ROOT not found: $PHASE5_ROOT"
    exit 1
fi
if [[ ! -d "$RADIUS_PRED_ROOT" ]]; then
    echo "[ERROR] RADIUS_PRED_ROOT not found: $RADIUS_PRED_ROOT"
    exit 1
fi

while IFS= read -r snr_path; do
    case_path="$(dirname "$snr_path")"
    case_name="$(basename "$case_path")"
    snr_tag="$(basename "$snr_path")"

    mat_path="$snr_path/noisy_phase_${snr_tag}.mat"
    if [[ ! -f "$mat_path" ]]; then
        continue
    fi

    radius_path="$(find_radius_file "$snr_path" "$RADIUS_PRED_ROOT" "$case_name" "$snr_tag" || true)"
    if [[ -z "$radius_path" ]]; then
        continue
    fi

    if [[ "${RECON_INPLACE_IO,,}" == "true" || "${RECON_INPLACE_IO,,}" == "1" || "${RECON_INPLACE_IO,,}" == "yes" || "${RECON_INPLACE_IO,,}" == "on" ]]; then
        out_dir="$snr_path"
    else
        out_dir="$RECON_OUT_ROOT/$case_name/$snr_tag"
        mkdir -p "$out_dir"
    fi

    out_nii="$out_dir/${case_name}_${snr_tag}_sigma_recon.nii.gz"
    printf '%s|%s|%s|%s|%s\n' "$case_name" "$snr_tag" "$mat_path" "$radius_path" "$out_nii" >> "$TASK_LIST"
done < <(find "$PHASE5_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'SNR*' | sort)

TASK_COUNT="$(wc -l < "$TASK_LIST" | tr -d ' ')"

if [[ "$TASK_COUNT" -eq 0 ]]; then
    echo "[ERROR] No runnable recon tasks found under: $PHASE5_ROOT"
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
echo "RADIUS_PRED_ROOT: $RADIUS_PRED_ROOT"
echo "RECON_INPLACE_IO: $RECON_INPLACE_IO"
echo "RECON_OUT_ROOT: $RECON_OUT_ROOT"
echo "Task count: $TASK_COUNT"
echo "Log: $LOG_FILE"
echo "================================================="

if [[ -z "$TASK_ID" ]]; then
    echo "[INFO] No SGE_TASK_ID/RECON_TASK_ID provided."
    echo "[INFO] Submit as array, e.g.: qsub -t 1-${TASK_COUNT} /home/zcemexx/projects/MREPT_code/recon.sh"
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
IFS='|' read -r case_name snr_tag mat_path radius_path out_nii <<< "$line"

echo "[RUN ] TASK $TASK_ID/$TASK_COUNT -> $case_name $snr_tag"
echo "      mat    : $mat_path"
echo "      radius : $radius_path"
echo "      out    : $out_nii"

module purge
module load matlab

export PHASE5_ROOT
export RADIUS_PRED_ROOT
export RECON_INPLACE_IO
export RECON_OUT_ROOT
export INPUT_DATA="$mat_path"
export RADIUS_NII="$radius_path"
export OUT_NII="$out_nii"

matlab -nodisplay -nodesktop -r "cd('$CODE_DIR'); try, reconstruct_conductivity_from_radiusmap(); catch ME, disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"

echo "[DONE] Task $TASK_ID finished successfully."
