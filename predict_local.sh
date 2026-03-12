#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Local nnUNet prediction entrypoint with optional NIfTI geometry preflight.
#
# High-level flow:
# 1) Resolve environment / model / I/O paths.
# 2) Optionally run geometry preflight (report/fix) before inference.
# 3) Execute nnUNet prediction command.
#
# This script avoids shell-glob pitfalls by delegating NIfTI file discovery to
# Python preflight tooling (`scripts/normalize_nifti_identity.py`).
#
# How to use:
# 1) Basic prediction with geometry check (default preflight mode is "check"):
#    INPUT_DIR=/path/to/imagesTs \
#    OUTPUT_DIR=/path/to/preds \
#    MODEL_DIR=/path/to/model_dir \
#    bash predict_local.sh
#
# 2) Auto-fix only undefined geometry before prediction:
#    INPUT_DIR=/path/to/imagesTs \
#    OUTPUT_DIR=/path/to/preds \
#    MODEL_DIR=/path/to/model_dir \
#    NIFTI_GEOM_PREFLIGHT=fix-undefined \
#    bash predict_local.sh
#
# 3) Fill missing q/s form before prediction:
#    INPUT_DIR=/path/to/imagesTs \
#    OUTPUT_DIR=/path/to/preds \
#    MODEL_DIR=/path/to/model_dir \
#    NIFTI_GEOM_PREFLIGHT=fix-missing-form \
#    bash predict_local.sh
#
# 4) Force full rewrite with explicit matrix policy:
#    INPUT_DIR=/path/to/imagesTs \
#    OUTPUT_DIR=/path/to/preds \
#    MODEL_DIR=/path/to/model_dir \
#    NIFTI_GEOM_PREFLIGHT=fix-all \
#    NIFTI_GEOM_SFORM_MATRIX=identity \
#    bash predict_local.sh
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"

echo "=== nnUNet local predict start ==="
echo "time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "host: $(hostname)"
echo "project: $PROJECT_DIR"

# Optional environment setup (can be overridden via NNUNET_VENV).
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
TRAINER="${TRAINER:-nnUNetTrainerMRCT_mae_regfix}"
PLANS="${PLANS:-nnResUNetPlans}"
FOLDS="${FOLDS:-0 1 2 3 4}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
REC_MODE="${REC_MODE:-gaussian}"
STEP_SIZE="${STEP_SIZE:-0.3}"
DEVICE="${DEVICE:-auto}"
NPP="${NPP:-2}"
NPS="${NPS:-2}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# Geometry preflight configuration.
#
# Notes:
# - Pattern is passed as a raw string to Python (not expanded in Bash).
# - Empty optional values are intentionally omitted from CLI construction.
# - off: skip geometry checks/fixes.
# - check: inspect files and fail if undefined geometry is detected.
# - fix-undefined: rewrite only qform=0 && sform=0 files.
# - fix-missing-form: fill whichever form is missing (or both when both are missing).
# - fix-all: rewrite all matched files.
NIFTI_GEOM_PREFLIGHT="${NIFTI_GEOM_PREFLIGHT:-check}"
NIFTI_GEOM_PATTERN="${NIFTI_GEOM_PATTERN:-*.nii.gz}"
NIFTI_GEOM_FAIL_ON_UNDEFINED="${NIFTI_GEOM_FAIL_ON_UNDEFINED:-true}"
NIFTI_GEOM_FAIL_ON_NO_MATCH="${NIFTI_GEOM_FAIL_ON_NO_MATCH:-true}"
NIFTI_GEOM_HANDEDNESS_POLICY="${NIFTI_GEOM_HANDEDNESS_POLICY:-unify}"
NIFTI_GEOM_QFORM_CODE="${NIFTI_GEOM_QFORM_CODE:-}"
NIFTI_GEOM_SFORM_CODE="${NIFTI_GEOM_SFORM_CODE:-}"
NIFTI_GEOM_SFORM_MATRIX="${NIFTI_GEOM_SFORM_MATRIX:-}"
NIFTI_GEOM_PIXDIM_SPATIAL="${NIFTI_GEOM_PIXDIM_SPATIAL:-}"
NIFTI_GEOM_PIXDIM="${NIFTI_GEOM_PIXDIM:-}"
NIFTI_GEOM_XYZT_UNITS="${NIFTI_GEOM_XYZT_UNITS:-}"
NIFTI_GEOM_CAL_MIN="${NIFTI_GEOM_CAL_MIN:-}"
NIFTI_GEOM_CAL_MAX="${NIFTI_GEOM_CAL_MAX:-}"
NIFTI_GEOM_DESCRIP="${NIFTI_GEOM_DESCRIP:-}"
NIFTI_GEOM_DRY_RUN="${NIFTI_GEOM_DRY_RUN:-false}"

MODEL_IDENTIFIER="${TRAINER}__${PLANS}__${CONFIG}"
MODEL_DIR="${MODEL_DIR:-}"
if [[ -z "$MODEL_DIR" ]]; then
  if [[ -d "$nnUNet_results/$MODEL_IDENTIFIER" ]]; then
    MODEL_DIR="$nnUNet_results/$MODEL_IDENTIFIER"
  else
    MODEL_DIR="$(find "$nnUNet_results" -maxdepth 3 -type d -name "$MODEL_IDENTIFIER" | head -n 1 || true)"
  fi
fi

# Required runtime inputs:
# INPUT_DIR=/path/to/imagesTs OUTPUT_DIR=/path/to/preds bash predict_local.sh
INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
: "${INPUT_DIR:?Set INPUT_DIR to folder with *_0000.nii.gz (and *_0001 if multi-channel)}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for predictions}"
: "${MODEL_DIR:?Set MODEL_DIR or ensure a matching model exists in nnUNet_results}"

if ! command -v nnUNetv2_predict_from_modelfolder >/dev/null 2>&1; then
  if [[ -d "$PROJECT_DIR/nnunet/nnunetv2" ]]; then
    # Fallback for editable-source setups where CLI is not on PATH.
    export PYTHONPATH="$PROJECT_DIR/nnunet${PYTHONPATH:+:$PYTHONPATH}"
  fi
fi

if command -v nnUNetv2_predict_from_modelfolder >/dev/null 2>&1; then
  # Preferred path: installed nnUNet CLI entrypoint.
  PREDICT_CMD=(nnUNetv2_predict_from_modelfolder)
else
  if python -c "import nnunetv2" >/dev/null 2>&1; then
    # Source fallback: call entrypoint function via python -c.
    PREDICT_CMD=(python -c "from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder as _main; _main()")
  else
    echo "[ERROR] nnUNetv2_predict_from_modelfolder not found and nnunetv2 cannot be imported."
    echo "        Install env (pip install -e $PROJECT_DIR/nnunet) or set NNUNET_VENV."
    exit 127
  fi
fi

if [[ "$DEVICE" == "auto" ]]; then
  # Auto-select GPU only when CUDA tooling is available.
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

is_true() {
  # Accept common truthy strings for environment flag parsing.
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

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
echo "NIFTI_GEOM_PREFLIGHT=$NIFTI_GEOM_PREFLIGHT"
echo "NIFTI_GEOM_PATTERN=$NIFTI_GEOM_PATTERN"
echo "NIFTI_GEOM_FAIL_ON_UNDEFINED=$NIFTI_GEOM_FAIL_ON_UNDEFINED"
echo "NIFTI_GEOM_FAIL_ON_NO_MATCH=$NIFTI_GEOM_FAIL_ON_NO_MATCH"
echo "NIFTI_GEOM_HANDEDNESS_POLICY=$NIFTI_GEOM_HANDEDNESS_POLICY"
echo "NIFTI_GEOM_QFORM_CODE=${NIFTI_GEOM_QFORM_CODE:-<unset>}"
echo "NIFTI_GEOM_SFORM_CODE=${NIFTI_GEOM_SFORM_CODE:-<unset>}"
echo "NIFTI_GEOM_SFORM_MATRIX=${NIFTI_GEOM_SFORM_MATRIX:-<unset>}"
echo "NIFTI_GEOM_PIXDIM_SPATIAL=${NIFTI_GEOM_PIXDIM_SPATIAL:-<unset>}"
echo "NIFTI_GEOM_PIXDIM=${NIFTI_GEOM_PIXDIM:-<unset>}"
echo "NIFTI_GEOM_XYZT_UNITS=${NIFTI_GEOM_XYZT_UNITS:-<unset>}"
echo "NIFTI_GEOM_CAL_MIN=${NIFTI_GEOM_CAL_MIN:-<unset>}"
echo "NIFTI_GEOM_CAL_MAX=${NIFTI_GEOM_CAL_MAX:-<unset>}"
echo "NIFTI_GEOM_DESCRIP=${NIFTI_GEOM_DESCRIP:-<unset>}"
echo "NIFTI_GEOM_DRY_RUN=$NIFTI_GEOM_DRY_RUN"

test -d "$nnUNet_results"
test -d "$INPUT_DIR"
test -d "$MODEL_DIR"
test -f "$MODEL_DIR/plans.json"
test -f "$MODEL_DIR/dataset.json"
mkdir -p "$OUTPUT_DIR"

GEOM_FIX_SCRIPT="$PROJECT_DIR/scripts/normalize_nifti_identity.py"
if [[ ! -f "$GEOM_FIX_SCRIPT" ]]; then
  echo "[ERROR] Missing geometry preflight script: $GEOM_FIX_SCRIPT"
  exit 2
fi

append_preflight_arg() {
  # Append only non-empty optional arguments to avoid passing empty-string values.
  local flag="$1"
  local value="${2:-}"
  if [[ -n "$value" ]]; then
    PREFLIGHT_ARGS+=("$flag" "$value")
  fi
}

build_preflight_args() {
  # Build Python preflight args as an array to preserve exact argument boundaries.
  local action="$1"
  PREFLIGHT_ARGS=(--input "$INPUT_DIR" --pattern "$NIFTI_GEOM_PATTERN" --action "$action")

  if [[ "$action" != "report" ]]; then
    # Fix actions operate in-place for pre-predict normalization workflow.
    PREFLIGHT_ARGS+=(--inplace)
  fi

  if is_true "$NIFTI_GEOM_FAIL_ON_NO_MATCH"; then
    PREFLIGHT_ARGS+=(--fail-on-no-match)
  fi

  if [[ "$action" == "report" ]] && is_true "$NIFTI_GEOM_FAIL_ON_UNDEFINED"; then
    # Undefined geometry can block pipeline before expensive inference starts.
    PREFLIGHT_ARGS+=(--fail-on-undefined)
  fi

  # Always pass handedness policy so preflight behavior is explicit in logs.
  PREFLIGHT_ARGS+=(--handedness-policy "$NIFTI_GEOM_HANDEDNESS_POLICY")

  append_preflight_arg --sform-matrix "$NIFTI_GEOM_SFORM_MATRIX"
  append_preflight_arg --qform-code "$NIFTI_GEOM_QFORM_CODE"
  append_preflight_arg --sform-code "$NIFTI_GEOM_SFORM_CODE"
  append_preflight_arg --pixdim-spatial "$NIFTI_GEOM_PIXDIM_SPATIAL"
  append_preflight_arg --pixdim "$NIFTI_GEOM_PIXDIM"
  append_preflight_arg --xyzt-units "$NIFTI_GEOM_XYZT_UNITS"
  append_preflight_arg --cal-min "$NIFTI_GEOM_CAL_MIN"
  append_preflight_arg --cal-max "$NIFTI_GEOM_CAL_MAX"
  append_preflight_arg --descrip "$NIFTI_GEOM_DESCRIP"

  if is_true "$NIFTI_GEOM_DRY_RUN"; then
    PREFLIGHT_ARGS+=(--dry-run)
  fi
}

# Run geometry preflight before prediction to avoid affine/metadata ambiguity.
case "$(printf '%s' "$NIFTI_GEOM_PREFLIGHT" | tr '[:upper:]' '[:lower:]')" in
  off)
    echo "[INFO] Geometry preflight disabled."
    ;;
  check)
    build_preflight_args report
    python "$GEOM_FIX_SCRIPT" "${PREFLIGHT_ARGS[@]}"
    ;;
  fix-undefined)
    build_preflight_args fix-undefined
    python "$GEOM_FIX_SCRIPT" "${PREFLIGHT_ARGS[@]}"
    ;;
  fix-missing-form)
    build_preflight_args fix-missing-form
    python "$GEOM_FIX_SCRIPT" "${PREFLIGHT_ARGS[@]}"
    ;;
  fix-all)
    build_preflight_args fix-all
    python "$GEOM_FIX_SCRIPT" "${PREFLIGHT_ARGS[@]}"
    ;;
  *)
    echo "[ERROR] Invalid NIFTI_GEOM_PREFLIGHT: $NIFTI_GEOM_PREFLIGHT"
    echo "        Allowed values: off | check | fix-undefined | fix-missing-form | fix-all"
    exit 2
    ;;
esac

nvidia-smi || true

# shellcheck disable=SC2086
# EXTRA_FLAGS is intentionally unquoted to allow caller-provided multi-flag payload.
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
#source /home/linux1917366562/nnunet_env/bin/activate
#cd /home/linux1917366562/MREPT_code
#INPUT_DIR=/home/linux1917366562/MREPT_code/data/M6 \
#OUTPUT_DIR=/home/linux1917366562/MREPT_code/preds/5f \
#NIFTI_GEOM_PREFLIGHT=fix-undefined \
#bash predict_local.sh
