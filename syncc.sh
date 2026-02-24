#!/bin/bash

# 配置
LOCAL_REPO_DIR="/Users/apple/Documents/MREPT_code"
REMOTE_HOST="myriad"
REMOTE_REPO_DIR="~/projects/MREPT_code"

echo "🎨 [1/2] Local: Pushing code to GitHub..."
git -C "$LOCAL_REPO_DIR" add .
git -C "$LOCAL_REPO_DIR" commit -m "Auto-sync $(date +'%Y-%m-%d %H:%M')" || true
git -C "$LOCAL_REPO_DIR" push origin main

echo "🌐 [2/2] Remote: Pulling code on Myriad..."
# 使用 bash -l -c 确保加载 Git 环境
ssh -T $REMOTE_HOST "bash -l -c 'cd $REMOTE_REPO_DIR && git pull origin main'"

if [ $? -eq 0 ]; then
    echo "✅ 代码同步完成！"
else
    echo "❌ 同步失败，请检查。"
    exit 1
fi