#!/bin/bash
#SBATCH --job-name=MREPT_Run
#SBATCH --output=logs/out_%j.log     # 日志存放在项目 logs 文件夹
#SBATCH --error=logs/err_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # 根据你的计算需求调整
#SBATCH --mem=16G                    # 内存需求
#SBATCH --time=04:00:00              # 预计运行 4 小时
#SBATCH --workdir=/scratch/zcemexx/MREPT_job  # 强制要求：在 Scratch 运行

# ## python 版本示例
# # 1. 确保必要的目录存在
# mkdir -p $HOME/projects/MREPT_code/logs
# SCRATCH_PATH="/scratch/zcemexx/MREPT_job"
# mkdir -p $SCRATCH_PATH

# # 2. 加载 Myriad 上的软件环境 (例如你之前用的 git)
# module purge
# module load python  # 或者是 matlab/R2023a 等，取决于你的代码

# # 3. 运行你的核心程序
# # 注意：代码路径在 $HOME，但运行产生的大数据要写进 $SCRATCH_PATH
# python ~/projects/MREPT_code/code/main.py --output_dir $SCRATCH_PATH

## matlab 版本示例
# 1. 环境准备
module purge
module load matlab  # 请根据 Myriad 实际可用的版本调整

# 2. 运行 MATLAB
# -nodisplay: 不启动图形界面
# -nosplash: 不显示启动画面
# -nodesktop: 禁用桌面环境
# -r: 运行指定的命令或脚本名（注意不要加 .m 后缀）
matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('~/projects/MREPT_code')); labelv1_4; exit;"