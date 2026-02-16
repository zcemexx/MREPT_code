#!/bin/bash -l
#$ -S /bin/bash
#$ -N nnUNetSmoke
#$ -l h_rt=00:20:00
#$ -l mem=4G
#$ -l tmpfs=4G
#$ -pe smp 2
#$ -wd /myriadfs/home/zcemexx/Scratch
#$ -o /myriadfs/home/zcemexx/Scratch/logs/
#$ -j y
#$ -m abe
#$ -M zcemexx@ucl.ac.uk

set -euo pipefail

echo "=== nnUNet smoke test start ==="
echo "time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "host: $(hostname)"
echo "job: ${JOB_ID:-NA} task: ${SGE_TASK_ID:-NA}"

module purge
module load python/3.9.10
source /myriadfs/home/zcemexx/venvs/nnunet39/bin/activate

export nnUNet_raw=/myriadfs/home/zcemexx/Scratch/nnUNet_raw
export nnUNet_preprocessed=/myriadfs/home/zcemexx/Scratch/nnUNet_preprocessed
export nnUNet_results=/myriadfs/home/zcemexx/Scratch/nnUNet_results

echo "python: $(which python)"
python -V
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

test -d "$nnUNet_raw"
test -d "$nnUNet_preprocessed"
test -d "$nnUNet_results"
test -d "$nnUNet_raw/Dataset001_EPT/imagesTr"
test -d "$nnUNet_raw/Dataset001_EPT/labelsTr"
test -f "$nnUNet_raw/Dataset001_EPT/dataset.json"

python -c "from nnunetv2.run.run_training import run_training_entry; print('nnUNet train entry OK')"
nnUNetv2_plan_and_preprocess -h | head -n 12
nnUNetv2_train -h | head -n 12

# Optional (slower) integrity-only check on compute node:
# nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity --no_pp

echo "=== nnUNet smoke test passed ==="
