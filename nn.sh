#!/bin/bash -l
#$ -S /bin/bash
#$ -N nnUTrain
#$ -l h_rt=48:00:00
#$ -l mem=16G
#$ -l tmpfs=40G
#$ -pe smp 8
#$ -l gpu=1
#$ -t 1-5
#$ -tc 2
#$ -wd /myriadfs/home/zcemexx/Scratch
#$ -o /myriadfs/home/zcemexx/Scratch/logs/
#$ -j y

set -euo pipefail

echo "=== nnUNet train job start ==="
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
FOLD="${FOLD:-$((SGE_TASK_ID - 1))}"
TRAINER="${TRAINER:-nnUNetTrainerMRCT_mae}"
PLANS="${PLANS:-nnResUNetPlans}"

echo "python: $(which python)"
python -V
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
echo "DATASET_ID=$DATASET_ID CONFIG=$CONFIG FOLD=$FOLD TRAINER=$TRAINER PLANS=$PLANS"

test -d "$nnUNet_raw"
test -d "$nnUNet_preprocessed"
test -d "$nnUNet_results"
test -f "$nnUNet_preprocessed/Dataset001_EPT/nnUNetPlans.json"

# Keep this for quick troubleshooting of GPU allocation on node
nvidia-smi || true

nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD" -tr "$TRAINER" -p "$PLANS"

echo "=== nnUNet train job finished ==="
