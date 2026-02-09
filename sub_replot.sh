#!/bin/bash -l
#$ -S /bin/bash
#$ -N ReplotGlobal
#$ -l h_rt=1:00:00
#$ -l mem=4G
#$ -l tmpfs=2G
#$ -wd /myriadfs/home/zcemexx/Scratch
#$ -o /myriadfs/home/zcemexx/Scratch/logs/
#$ -e /myriadfs/home/zcemexx/Scratch/logs/
#$ -m bea
#$ -M zcemexx@ucl.ac.uk

module unload compilers mpi
module load matlab

CODE_DIR="/myriadfs/home/zcemexx/projects/MREPT_code/matlab/code"

echo "================================================="
echo "Replot job started on host: $(hostname)"
echo "================================================="

matlab -nodisplay -nodesktop -r "cd('$CODE_DIR'); try, run('replot_global_metric_scales.m'); catch e, disp(getReport(e,'extended')); exit(1); end, exit(0);"
