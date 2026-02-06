#!/bin/bash
#SBATCH --job-name=MatlabBatch
#SBATCH --output=/myriadfs/home/zcemexx/Scratch/logs/job_%A_%a.out  # 日志存 Scratch
#SBATCH --error=/myriadfs/home/zcemexx/Scratch/logs/job_%A_%a.err
#SBATCH --time=10:00:00               # 预计运行时间
#SBATCH --mem=8G                      # 内存
#SBATCH --cpus-per-task=8
#SBATCH --array=84                 # 【注意】运行前请将其修改为 data 文件夹下的实际文件数量！
#SBATCH --workdir=/myriadfs/home/zcemexx/Scratch  # 工作目录

# 1. 建立日志目录 (确保可写)
mkdir -p /myriadfs/home/zcemexx/Scratch/logs

# 2. 加载模块
module purge
module load matlab  # 推荐加上具体版本号

# 3. 设置绝对代码路径
CODE_DIR="/myriadfs/home/zcemexx/projects/MREPT_code/code"

# 4. 运行 MATLAB
echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"

# 进入代码目录 -> 运行 aminos_batch.m -> 退出
matlab -nodisplay -nodesktop -r "cd('$CODE_DIR'); try, run('labelsin1.m'); catch e, disp(e.message); exit(1); end, exit(0);"

echo "Task $SLURM_ARRAY_TASK_ID finished."