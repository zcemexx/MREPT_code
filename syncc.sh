#!/bin/bash

# --- 1. 配置信息 ---
REMOTE_HOST="myriad"
# 确保这里是你的代码库路径 (存放 .git 的那个文件夹)
REMOTE_REPO_DIR="~/projects/MREPT_code" 

echo "🎨 [1/2] 正在推送本地代码到 GitHub..."
# 提交本地修改
git add .
git commit -m "Auto-sync $(date +'%Y-%m-%d %H:%M')"
git push origin main

echo "🌐 [2/2] 正在连接 Myriad 更新代码并提交作业..."

# 【核心修改】
# 使用 -T 禁止分配伪终端 (消除那个 warning)
# 将所有命令用双引号包起来，用 && 连接
ssh -T $REMOTE_HOST "cd $REMOTE_REPO_DIR && git pull origin main && sbatch submit_array.sh"

# 检查上一条命令是否成功
if [ $? -eq 0 ]; then
    echo "✅ 成功！作业已提交。"
    echo "💡 你可以手动运行 'ssh myriad squeue -u zcemexx' 查看状态。"
else
    echo "❌ 远程执行失败，请手动登录 Myriad 检查。"
fi