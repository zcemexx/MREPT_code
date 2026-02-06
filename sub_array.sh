#!/bin/bash -l
#$ -S /bin/bash
#$ -N MatlabBatch              # 作业名称
#$ -l h_rt=9:59:58             # 运行时间 (时:分:秒)
#$ -l mem=8G                   # 内存申请
#$ -pe smp 8                   # 申请 4 个 CPU 核心
#$ -l tmpfs=10G                # 临时空间
#$ -t 1-84                   # 【关键】任务阵列 (Array Task) 1 到 5602
#$ -wd /myriadfs/home/zcemexx/Scratch  # 工作目录 (Working Directory)
#$ -o /myriadfs/home/zcemexx/Scratch/logs/  # 日志目录 (自动命名)
#$ -e /myriadfs/home/zcemexx/Scratch/logs/  # 错误日志目录 (自动命名)

# SGE 必须显式加载环境
module unload compilers mpi
module load matlab

# 设置代码路径
CODE_DIR="/myriadfs/home/zcemexx/projects/MREPT_code/code"

echo "Starting SGE Task ID: $SGE_TASK_ID"

# 运行 MATLAB
# 注意：SGE 的任务 ID 变量是 $SGE_TASK_ID (Slurm 是 SLURM_ARRAY_TASK_ID)
# 我们需要把 SGE_TASK_ID 传给 MATLAB 或者让 MATLAB 自己读 (你的代码能读环境变量吗？)
# 鉴于你的 MATLAB 代码是读文件列表的，我们需要一种机制把 ID 传进去。
# 最简单的方法是：让 MATLAB 知道它是第几个任务。

matlab -nodisplay -nodesktop -r "cd('$CODE_DIR'); try, run('labelsin1.m'); catch e, disp(e.message); exit(1); end, exit(0);"

echo "Task $SGE_TASK_ID finished."