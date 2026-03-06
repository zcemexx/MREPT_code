#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"

echo "=== nnUNet local predict start ==="
echo "time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "host: $(hostname)"
echo "project: $PROJECT_DIR"

# Optional environment setup (override at runtime if needed)
NNUNET_VENV="${NNUNET_VENV:-}"
if [[ -z "$NNUNET_VENV" ]]; then
  for candidate in \
    "$PROJECT_DIR/.venv/bin/activate" \
    "$PROJECT_DIR/nnunet/.venv/bin/activate"
  do
    if [[ -f "$candidate" ]]; then
      NNUNET_VENV="$candidate"
      break
    fi
  done
fi
if [[ -f "$NNUNET_VENV" ]]; then
  # shellcheck disable=SC1090
  source "$NNUNET_VENV"
fi

export nnUNet_raw="${nnUNet_raw:-$PROJECT_DIR/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_DIR/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_DIR/nnUNet_results}"

DATASET_ID="${DATASET_ID:-1}"
CONFIG="${CONFIG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainerMRCT_mae_grad_regfix}"
PLANS="${PLANS:-nnResUNetPlans}"
FOLDS="${FOLDS:-0 1 2 3 4}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
REC_MODE="${REC_MODE:-gaussian}"
STEP_SIZE="${STEP_SIZE:-0.3}"
DEVICE="${DEVICE:-auto}"
NPP="${NPP:-2}"
NPS="${NPS:-2}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

MODEL_IDENTIFIER="${TRAINER}__${PLANS}__${CONFIG}"
MODEL_DIR="${MODEL_DIR:-}"
if [[ -z "$MODEL_DIR" ]]; then
  if [[ -d "$nnUNet_results/$MODEL_IDENTIFIER" ]]; then
    MODEL_DIR="$nnUNet_results/$MODEL_IDENTIFIER"
  else
    MODEL_DIR="$(find "$nnUNet_results" -maxdepth 3 -type d -name "$MODEL_IDENTIFIER" | head -n 1 || true)"
  fi
fi

# Set these two at runtime:
# INPUT_DIR=/path/to/imagesTs OUTPUT_DIR=/path/to/preds bash predict_local.sh
INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
: "${INPUT_DIR:?Set INPUT_DIR to folder with *_0000.nii.gz (and *_0001 if multi-channel)}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for predictions}"
: "${MODEL_DIR:?Set MODEL_DIR or ensure a matching model exists in nnUNet_results}"

if ! command -v nnUNetv2_predict_from_modelfolder >/dev/null 2>&1; then
  if [[ -d "$PROJECT_DIR/nnunet/nnunetv2" ]]; then
    export PYTHONPATH="$PROJECT_DIR/nnunet${PYTHONPATH:+:$PYTHONPATH}"
  fi
fi

if command -v nnUNetv2_predict_from_modelfolder >/dev/null 2>&1; then
  PREDICT_CMD=(nnUNetv2_predict_from_modelfolder)
else
  if python -c "import nnunetv2" >/dev/null 2>&1; then
    PREDICT_CMD=(python -c "from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder as _main; _main()")
  else
    echo "[ERROR] nnUNetv2_predict_from_modelfolder not found and nnunetv2 cannot be imported."
    echo "        Install env (pip install -e $PROJECT_DIR/nnunet) or set NNUNET_VENV."
    exit 127
  fi
fi

if [[ "$DEVICE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

echo "python: $(which python || true)"
python -V || true
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
echo "DATASET_ID=$DATASET_ID CONFIG=$CONFIG TRAINER=$TRAINER PLANS=$PLANS"
echo "FOLDS=$FOLDS CHECKPOINT=$CHECKPOINT REC_MODE=$REC_MODE STEP_SIZE=$STEP_SIZE"
echo "MODEL_DIR=$MODEL_DIR"
echo "INPUT_DIR=$INPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PREDICT_CMD=${PREDICT_CMD[*]}"

test -d "$nnUNet_results"
test -d "$INPUT_DIR"
test -d "$MODEL_DIR"
test -f "$MODEL_DIR/plans.json"
test -f "$MODEL_DIR/dataset.json"
mkdir -p "$OUTPUT_DIR"

nvidia-smi || true

# shellcheck disable=SC2086
"${PREDICT_CMD[@]}" \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  -m "$MODEL_DIR" \
  -f ${FOLDS} \
  -chk "$CHECKPOINT" \
  -step_size "$STEP_SIZE" \
  -npp "$NPP" \
  -nps "$NPS" \
  -device "$DEVICE" \
  --rec "$REC_MODE" \
  $EXTRA_FLAGS

echo "=== nnUNet local predict finished ==="

#how to run:
#cd /home/linux1917366562/MREPT_code
#INPUT_DIR=/home/linux1917366562/MREPT_code/data/M6 \
#OUTPUT_DIR=/home/linux1917366562/MREPT_code/preds/5f \
#bash predict_local.sh
