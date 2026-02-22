#!/bin/bash -l
#$ -S /bin/bash
#$ -N nnUPredict
#$ -l h_rt=00:00:19
#$ -l mem=16G
#$ -l tmpfs=40G
#$ -pe smp 8
#$ -l gpu=1
#$ -wd /myriadfs/home/zcemexx/Scratch
#$ -o /myriadfs/home/zcemexx/Scratch/logs/
#$ -j y

set -euo pipefail

echo "=== nnUNet predict job start ==="
echo "time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "host: $(hostname)"
echo "job: ${JOB_ID:-NA} task: ${SGE_TASK_ID:-NA}"

module purge
module load python/3.9.10
source /myriadfs/home/zcemexx/venvs/nnunet39/bin/activate

export nnUNet_raw=/myriadfs/home/zcemexx/Scratch/nnUNet_raw
export nnUNet_preprocessed=/myriadfs/home/zcemexx/Scratch/nnUNet_preprocessed
export nnUNet_results=/myriadfs/home/zcemexx/Scratch/nnUNet_results

DATASET_ID="${DATASET_ID:-1}"
CONFIG="${CONFIG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainerMRCT_mae}"
PLANS="${PLANS:-nnResUNetPlans}"
FOLDS="${FOLDS:-0}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
REC_MODE="${REC_MODE:-center_mean}"
STEP_SIZE="${STEP_SIZE:-0.3}"
DEVICE="${DEVICE:-cuda}"
NPP="${NPP:-3}"
NPS="${NPS:-3}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# Input/output can be overridden by qsub -v INPUT_DIR=...,OUTPUT_DIR=...
INPUT_DIR="${INPUT_DIR:-/myriadfs/home/zcemexx/Scratch/exp/exp001_EPT/imagesTs}"
OUTPUT_DIR="${OUTPUT_DIR:-/myriadfs/home/zcemexx/Scratch/preds/fold0_center_test}"

# keep safety checks
: "${INPUT_DIR:?INPUT_DIR is empty}"
: "${OUTPUT_DIR:?OUTPUT_DIR is empty}"

echo "python: $(which python)"
python -V
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

echo "=== nnUNet predict job finished ==="
