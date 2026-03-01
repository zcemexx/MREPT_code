#!/usr/bin/env bash
set -euo pipefail

echo "=== nnUNet local predict start ==="
echo "time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "host: $(hostname)"

# Optional environment setup (override at runtime if needed)
NNUNET_VENV="${NNUNET_VENV:-/myriadfs/home/zcemexx/venvs/nnunet39/bin/activate}"
if [[ -f "$NNUNET_VENV" ]]; then
  # shellcheck disable=SC1090
  source "$NNUNET_VENV"
fi

export nnUNet_raw="${nnUNet_raw:-/myriadfs/home/zcemexx/Scratch/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/myriadfs/home/zcemexx/Scratch/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/myriadfs/home/zcemexx/Scratch/nnUNet_results}"

DATASET_ID="${DATASET_ID:-1}"
CONFIG="${CONFIG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainerMRCT_mae}"
PLANS="${PLANS:-nnResUNetPlans}"
FOLDS="${FOLDS:-0}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
REC_MODE="${REC_MODE:-gaussian}"
STEP_SIZE="${STEP_SIZE:-0.3}"
DEVICE="${DEVICE:-cuda}"
NPP="${NPP:-2}"
NPS="${NPS:-2}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# Set these two at runtime:
# INPUT_DIR=/path/to/imagesTs OUTPUT_DIR=/path/to/preds bash predict_local.sh
INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
: "${INPUT_DIR:?Set INPUT_DIR to folder with *_0000.nii.gz (and *_0001 if multi-channel)}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for predictions}"

echo "python: $(which python || true)"
python -V || true
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
echo "DATASET_ID=$DATASET_ID CONFIG=$CONFIG TRAINER=$TRAINER PLANS=$PLANS"
echo "FOLDS=$FOLDS CHECKPOINT=$CHECKPOINT REC_MODE=$REC_MODE STEP_SIZE=$STEP_SIZE"
echo "INPUT_DIR=$INPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"

test -d "$nnUNet_preprocessed"
test -d "$nnUNet_results"
test -d "$INPUT_DIR"
test -f "$nnUNet_preprocessed/Dataset001_EPT/${PLANS}.json"
mkdir -p "$OUTPUT_DIR"

nvidia-smi || true

# shellcheck disable=SC2086
nnUNetv2_predict \
  -d "$DATASET_ID" \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  -c "$CONFIG" \
  -tr "$TRAINER" \
  -p "$PLANS" \
  -f ${FOLDS} \
  -chk "$CHECKPOINT" \
  -step_size "$STEP_SIZE" \
  -npp "$NPP" \
  -nps "$NPS" \
  -device "$DEVICE" \
  --rec "$REC_MODE" \
  $EXTRA_FLAGS

echo "=== nnUNet local predict finished ==="
