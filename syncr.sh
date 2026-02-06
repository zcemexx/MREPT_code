#!/bin/bash

echo "🔄 step1: 正在推送本地代码到 GitHub..."
./syncc.sh

# 如果同步失败，直接退出，不提交作业
if [ $? -ne 0 ]; then
    echo "❌ 同步失败，终止作业提交。"
    exit 1
fi

echo "🚀 Step 2: 提交 Myriad 作业..."
REMOTE_HOST="myriad"
REMOTE_REPO_DIR="~/projects/MREPT_code"

# 【关键】使用 bash -l -c 加载 Slurm 环境，否则会报 sbatch command not found
# 假设 submit_array.sh 在根目录。如果在 others 文件夹，请改为 others/submit_array.sh
ssh -T $REMOTE_HOST "bash -l -c 'cd $REMOTE_REPO_DIR && sbatch sub_array.sh'"

if [ $? -eq 0 ]; then
    echo "✅ 作业提交成功！"
    echo "💡 查询命令: ssh myriad squeue -u zcemexx"
else
    echo "❌ 作业提交失败。"
fi