#!/bin/bash

# --- 配置区 ---
REMOTE_HOST="myriad"
REMOTE_DIR="~/projects/MREPT_code"
BRANCH="main"

echo "🎨 [1/2] 正在推送本地代码到 GitHub..."
git add .
# 如果没有变动，commit 会跳过
git commit -m "Manual sync $(date +'%Y-%m-%d %H:%M')" || echo "没有检测到新代码变动。"
git push origin $BRANCH

echo "🌐 [2/2] 正在通知 Myriad 更新代码..."
ssh $REMOTE_HOST << EOF
    cd $REMOTE_DIR
    git pull origin $BRANCH
    echo "✅ Myriad 代码已原地更新。"
EOF

echo "✨ 同步完成。"