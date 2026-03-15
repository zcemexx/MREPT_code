#!/bin/bash

REMOTE_HOST="myriad"

echo "🎨 [1/2] Local: Pushing code to GitHub..."
git add .

if git diff --cached --quiet; then
    echo "ℹ️  Local: no changes to commit."
else
    git commit -m "Auto-sync $(date +'%Y-%m-%d %H:%M')"
fi

git push origin main || exit 1

echo "🌐 [2/2] Remote: Pulling code on Myriad..."
ssh -T "$REMOTE_HOST" 'bash -l -s' <<'EOF'
set -e

REMOTE_REPO_DIR="$HOME/projects/MREPT_code"
cd "$REMOTE_REPO_DIR"

if ! git diff --quiet || ! git diff --cached --quiet; then
    stash_name="auto-sync-before-pull-$(date +'%Y%m%d-%H%M%S')"
    git stash save -u "$stash_name" >/dev/null
    echo "[remote] Stashed local changes as: $stash_name"
fi

git pull --ff-only origin main
EOF

if [ $? -eq 0 ]; then
    echo "✅ 代码同步完成！"
else
    echo "❌ 同步失败，请检查。"
    exit 1
fi
