#!/bin/bash -l
#$ -S /bin/bash
#$ -N invFixR17
#$ -l h_rt=12:59:00
#$ -l mem=5G
#$ -l tmpfs=8G
#$ -pe smp 8
#$ -wd /home/zcemexx/Scratch
#$ -o /home/zcemexx/Scratch/logs/
#$ -j y
#$ -m abe
#$ -M zcemexx@ucl.ac.uk

set -euo pipefail
module purge
module load matlab/full/r2023a/9.14

DATA_ROOT=/home/zcemexx/Scratch/data/invivo/3T_in-vivo_3DGRE_Head_Neck_repeatibility \
CODE_DIR=/home/zcemexx/projects/MREPT_code/matlab/code \
MATLAB_BIN=matlab \
FIXED_RADIUS=17 \
ESTIMATE_NOISE=true \
OUT_NII=/home/zcemexx/Scratch/data/invivo/3T_in-vivo_3DGRE_Head_Neck_repeatibility/cond_fixr17.nii.gz \
bash /home/zcemexx/projects/MREPT_code/invivo.sh
