#!/bin/bash -l
#$ -S /bin/bash
#$ -N MatlabBatch             # 作业名称
#$ -l h_rt=9:59:58              # 运行时间
#$ -l mem=8G                  # 内存
#$ -l tmpfs=10G               # 临时存储
#$ -pe smp 8                  # 4核并行
#$ -t 1-84                    # 【重要】根据你的文件数量设置 (1-84)
#$ -wd /myriadfs/home/zcemexx/Scratch  # 工作目录
#$ -o /myriadfs/home/zcemexx/Scratch/logs/
#$ -e /myriadfs/home/zcemexx/Scratch/logs/
#$ -m abe
#$ -M zcemexx@ucl.ac.uk

# 1. 加载环境
module unload compilers mpi
module load matlab

# 2. 定义关键路径变量
# CODE_DIR: 代码所在位置
CODE_DIR="/myriadfs/home/zcemexx/projects/MREPT_code/code"

# SCRATCH_OUT: MATLAB 输出结果的根目录 (根据你的 labelsin1.m 设置)
SCRATCH_OUT="/myriadfs/home/zcemexx/Scratch/nnUNet_raw"

# BACKUP_DIR: 想要保存结果的永久目录 (这里设为你的 Home 目录下的 backup 文件夹)
BACKUP_DIR="/myriadfs/home/zcemexx/ACFS/MREPT_Results_Backup"

# 3. 预先创建备份目录 (防止报错)
mkdir -p "$BACKUP_DIR/metrics/figures"
mkdir -p "$BACKUP_DIR/Dataset001_EPT/labelsTr"
mkdir -p "$BACKUP_DIR/Dataset001_EPT/imagesTr"
mkdir -p "$BACKUP_DIR/data/ADEPT_noisy"

echo "================================================="
echo "🚀 Task ID: $SGE_TASK_ID started on host: $(hostname)"
echo "📂 Output will be generated in Scratch first."
echo "================================================="

# 4. 运行 MATLAB 计算
# 注意：我们捕捉 exit code，只有计算成功才备份
matlab -nodisplay -nodesktop -r "cd('$CODE_DIR'); try, run('labelsin1.m'); catch e, disp(e.message); exit(1); end, exit(0);"
MATLAB_EXIT_CODE=$?

# 5. 结果备份逻辑 (仅当 MATLAB 成功运行后执行)
if [ $MATLAB_EXIT_CODE -eq 0 ]; then
    echo "✅ MATLAB Calculation Successful. Starting Data Backup..."

    # 使用 rsync 进行增量同步 (比 cp 更安全，适合多任务同时写)
    # -a: 归档模式 (保留时间戳等)
    # -v: 显示过程
    # --update: 仅在源文件较新时才拷贝 (避免覆盖)
    
    # 5.1 备份 Metrics (.mat 数据)
    rsync -av --update "$SCRATCH_OUT/metrics/" "$BACKUP_DIR/metrics/"
    
    # 5.2 备份生成的 Labels (.nii.gz)
    rsync -av --update "$SCRATCH_OUT/Dataset001_EPT/labelsTr/" "$BACKUP_DIR/Dataset001_EPT/labelsTr/"
    
    # 5.3 (可选) 备份含噪声的数据，如果你需要的话
    rsync -av --update "/myriadfs/home/zcemexx/Scratch/data/ADEPT_noisy/" "$BACKUP_DIR/data/ADEPT_noisy/"

    echo "📦 Backup for Task $SGE_TASK_ID Completed!"
    echo "💾 Saved to: $BACKUP_DIR"
else
    echo "❌ MATLAB Calculation Failed with error code $MATLAB_EXIT_CODE."
    echo "⚠️  Skipping backup step."
    exit 1
fi

echo "Task $SGE_TASK_ID Finished."