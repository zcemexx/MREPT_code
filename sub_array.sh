#!/bin/bash -l
#$ -S /bin/bash
#$ -N MatlabBatch             # 作业名称
#$ -l h_rt=11:59:58            # 运行时间
#$ -l mem=5G                  # 内存
#$ -l tmpfs=10G               # 临时存储
#$ -pe smp 8                  # 8核并行
#$ -t 1-84                    # 根据你的文件数量设置 (1-84)
#$ -wd /myriadfs/home/zcemexx/Scratch  # 工作目录
#$ -o /myriadfs/home/zcemexx/Scratch/logs/
#$ -j y                       # 合并 stdout/stderr
#$ -m abe
#$ -M zcemexx@ucl.ac.uk

# 1. 加载环境
module unload compilers mpi
module load matlab

# 2. 定义关键路径变量
# CODE_DIR: 代码所在位置
CODE_DIR="/myriadfs/home/zcemexx/projects/MREPT_code/matlab/code"

# SCRATCH_OUT: MATLAB 输出结果的根目录
SCRATCH_OUT="/myriadfs/home/zcemexx/Scratch/nnUNet_raw"
LOG_DIR="/myriadfs/home/zcemexx/Scratch/logs"
mkdir -p "$LOG_DIR"

# 每个 task 一个明确日志文件名: case number + JOB_ID + TASK_ID
TASK_ID="${SGE_TASK_ID:-0}"
CASE_NAME="M${TASK_ID}"
JOB_ID_SAFE="${JOB_ID:-NOJOB}"
RUN_DATE="$(date +'%Y-%m-%d %H:%M:%S')"
LOG_FILE="${LOG_DIR}/${CASE_NAME}_JOB${JOB_ID_SAFE}_TASK${TASK_ID}.log"

# 将本脚本的 stdout/stderr 合并并写入同一日志文件
exec >"$LOG_FILE" 2>&1

echo "================================================="
echo "🕒 Run time: $RUN_DATE"
echo "🚀 Task ID: $SGE_TASK_ID started on host: $(hostname)"
echo "📂 Output will be generated in Scratch: $SCRATCH_OUT"
echo "📝 Log file: $LOG_FILE"
echo "================================================="

# 3. 运行 MATLAB 计算
# 移除所有备份逻辑，只保留计算。只有计算成功才退出 0，否则退出 1
matlab -nodisplay -nodesktop -r "cd('$CODE_DIR'); try, run('labelsin1.m'); catch e, disp(e.message); exit(1); end, exit(0);"
