#!/bin/bash

# 1. 先调用同步脚本
./syncc.sh

# 2. 提交作业
echo "🚀 正在 Myriad 上提交 Slurm 作业..."
ssh myriad << EOF
    cd ~/projects/MREPT_code
    # 提交作业并打印作业 ID
    qsub submit_job.sh
    echo "------------------------------------"
    squeue -u \$USER | grep -E "JOBID|$(date +'%Y')"
EOF