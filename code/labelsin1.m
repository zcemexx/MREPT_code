%% ===================== Phase 2: Myriad Job Array Version =====================
% Modification: Adapted for Slurm Job Arrays.
% Each job processes ONE file based on SLURM_ARRAY_TASK_ID.
% Now saves 'task_id' for reproducibility and tracking.

clear; clc;
% close all; % 在集群模式下通常不需要关闭窗口，因为我们不显示它们

% --- 1. 路径配置 (Absolute Paths) ---
addpath('/myriadfs/home/zcemexx/projects/MREPT_code/functions');
addpath(genpath('/myriadfs/home/zcemexx/projects/MREPT_code/toolboxes'));

projectRoot  = '/myriadfs/home/zcemexx/Scratch';
datasetName  = 'Dataset001_EPT';
baseDir      = fullfile(projectRoot, 'nnUNet_raw');
inputDataDir = fullfile(projectRoot, 'data', 'ADEPT_raw');
noisyDataDir = fullfile(projectRoot, 'data', 'ADEPT_noisy');
outputBase   = fullfile(baseDir, datasetName);

imagesTr     = fullfile(outputBase, 'imagesTr');
labelsTr     = fullfile(outputBase, 'labelsTr');
metricsDir   = fullfile(baseDir, 'metrics');
figuresDir   = fullfile(metricsDir, 'figures');

% 确保文件夹存在 (suppress potential warnings in high concurrency)
if ~isfolder(baseDir); mkdir(baseDir); end
if ~isfolder(imagesTr); mkdir(imagesTr); end
if ~isfolder(labelsTr); mkdir(labelsTr); end
if ~isfolder(metricsDir); mkdir(metricsDir); end
if ~isfolder(noisyDataDir); mkdir(noisyDataDir); end
if ~isfolder(figuresDir); mkdir(figuresDir); end

% --- 2. 获取当前任务 ID (Job Array Core) ---
% Myriad 会自动传递 SLURM_ARRAY_TASK_ID 环境变量
task_id_str = getenv('SLURM_ARRAY_TASK_ID');
if isempty(task_id_str)
    task_id = 1;
    fprintf('警告: 未检测到 Slurm 阵列 ID，默认运行第 1 个文件 (本地测试模式)\n');
else
    task_id = str2double(task_id_str);
    fprintf('检测到 Slurm Job Array，正在处理第 %d 个任务\n', task_id);
end

% --- 3. 获取文件列表并定位当前任务的文件 ---
fileList = dir(fullfile(inputDataDir, 'M*.mat'));
nFiles = numel(fileList);

if task_id > nFiles
    fprintf('任务 ID (%d) 超过文件总数 (%d)，退出。\n', task_id, nFiles);
    return;
end

% 直接定位到当前要处理的那一个文件
currentFileStruct = fileList(task_id);
currentFilePath = fullfile(inputDataDir, currentFileStruct.name);
[~, fileName, ~] = fileparts(currentFileStruct.name);
caseName = fileName;

fprintf('正在处理 Case: %s (Task ID: %d)\n', caseName, task_id);

% --- 4. 并行池配置 (针对内部循环加速) ---
poolObj = gcp('nocreate');
if isempty(poolObj)
    nSlots = str2double(getenv('SLURM_CPUS_PER_TASK'));
    if isnan(nSlots), nSlots = 1; end
    if nSlots > 1
        parpool('local', nSlots);
        fprintf('并行池已启动，核心数: %d\n', nSlots);
    end
end

% --- 5. 参数配置 ---
doSmoothing   = false;
dorepair      = false;
doSaveMetrics = true;
doPlotting    = true;
doSubVolume   = false;
subVolSize    = 32;
do_filter     = true;
estimatenoise = true;
snr_range_linear = [10, 150];
snr_range_log    = log10(snr_range_linear);
radius_list      = 1:30;
nR               = numel(radius_list);

% ===================== 处理逻辑 (Single File) =====================

% --- 5.1 加载数据 ---
data = load(currentFilePath);

if isfield(data, 'Transceive_phase'), phi0_raw = single(data.Transceive_phase); else, error('Missing Transceive_phase'); end
if isfield(data, 'Conductivity_GT'), sigma_gt = single(data.Conductivity_GT); else, error('Missing Conductivity_GT'); end
tissueMask = uint8(data.Segmentation);

if isfield(data, 'B1plus_mag') && isfield(data, 'B1minus_mag')
    magnitude_raw = single(data.B1plus_mag .* data.B1minus_mag);
elseif isfield(data, 'B1plus_mag')
    magnitude_raw = single(data.B1plus_mag);
else
    magnitude_raw = single(data.T1w);
end

% 子卷提取
if doSubVolume
    sz = size(phi0_raw); cp = round(sz/2);
    idxX = cp(1)-subVolSize/2+1 : cp(1)+subVolSize/2;
    idxY = cp(2)-subVolSize/2+1 : cp(2)+subVolSize/2;
    idxZ = cp(3)-subVolSize/2+1 : cp(3)+subVolSize/2;

    phi0_raw = phi0_raw(idxX, idxY, idxZ);
    sigma_gt = sigma_gt(idxX, idxY, idxZ);
    tissueMask = tissueMask(idxX, idxY, idxZ);
    magnitude_raw = magnitude_raw(idxX, idxY, idxZ);
end

[nx, ny, nz] = size(phi0_raw);
mask = (tissueMask > 0);

% --- 5.2 加噪 (Reproducible Noise) ---
rng(task_id); % 【关键】用 Task ID 固定随机数种子，保证可复现性
this_snr_log = snr_range_log(1) + (snr_range_log(2)-snr_range_log(1)) * rand();
this_snr     = 10^this_snr_log;

complex_signal = magnitude_raw .* exp(1i * phi0_raw);
noisy_complex = gen_noise_complex(complex_signal, this_snr, mask);

phi0_noisy      = angle(noisy_complex);
magnitude_noisy = abs(noisy_complex);

% 保存 Noisy Data (包含 task_id)
nFileName = strrep(fileName, 'M', 'N');
if ~startsWith(nFileName, 'N'); nFileName = ['N_', fileName]; end
% 【修改】保存 task_id 到数据文件，以便后续知道这数据是用哪个种子生成的
save(fullfile(noisyDataDir, [nFileName, '.mat']), 'phi0_noisy', 'magnitude_noisy', 'tissueMask', 'this_snr', 'task_id');

% --- 5.3 穷举搜索 (并行版本) ---
err_stack  = inf(nx, ny, nz, nR, 'single');
cond_stack = zeros(nx, ny, nz, nR, 'single');
Parameters.B0 = 3; Parameters.VoxelSize = [1 1 1];

% 确保在使用 parfor 之前，切片变量已经初始化
parfor ir = 1:nR
    r = radius_list(ir);
    params_r = Parameters;
    params_r.kDiffSize = [2*r+1, 2*r+1, 2*r+1];

    % 执行重建
    [cond_r, ~] = conductivityMapping(phi0_noisy, mask, params_r, ...
        'magnitude', magnitude_noisy, 'segmentation', tissueMask, 'estimatenoise', estimatenoise);

    if do_filter; cond_r(cond_r < 0 | cond_r > 10) = NaN; end

    % 将结果存入 stack (MATLAB 会自动处理这种切片变量的并行写入)
    cond_stack(:,:,:,ir) = cond_r;

    diff_val = abs(cond_r - sigma_gt);
    bad_pixels = ~mask | isnan(cond_r);

    if doSmoothing
        % 注意：masked_smooth_error 必须是全自包含的函数，不能依赖非全局变量
        temp_err = diff_val;
        temp_err(bad_pixels) = 0;
        err_stack(:,:,:,ir) = masked_smooth_error(temp_err, tissueMask);
    else
        temp_err = diff_val;
        temp_err(bad_pixels) = inf;
        err_stack(:,:,:,ir) = temp_err;
    end
end

% --- 5.4 生成标签 ---
[min_err, best_idx_voxel] = min(err_stack, [], 4);
radiusMap_raw = zeros(size(phi0_raw), 'single');
radiusMap_raw(mask) = single(radius_list(best_idx_voxel(mask)));

% 最优重建
[gridX, gridY, gridZ] = ndgrid(1:nx, 1:ny, 1:nz);
linear_idx_optimal = sub2ind([nx, ny, nz, nR], gridX, gridY, gridZ, best_idx_voxel);
cond_optimal = cond_stack(linear_idx_optimal);
clear cond_stack;

if dorepair
    radiusMap_refined = refine_labels_mad(radiusMap_raw, tissueMask, radius_list);
else
    radiusMap_refined = radiusMap_raw;
end
label_final = uint8(round(radiusMap_refined));

% --- 5.5 保存 NIfTI ---
save_nii_gz(phi0_noisy,  fullfile(imagesTr, [caseName, '_0000.nii.gz']));
save_nii_gz(tissueMask,  fullfile(imagesTr, [caseName, '_0001.nii.gz']));
save_nii_gz(label_final, fullfile(labelsTr, [caseName, '.nii.gz']));

% --- 5.6 Metrics & Log (Save Individual File) ---
mae_val = mean(min_err(mask));
diff_sq = (cond_optimal - sigma_gt).^2;
rmse_val = sqrt(mean(diff_sq(mask)));

data_range = max(sigma_gt(mask)) - min(sigma_gt(mask));
if isempty(data_range) || data_range == 0; data_range = 1; end
vol_rec_ssim = cond_optimal; vol_rec_ssim(~mask) = 0;
vol_gt_ssim  = sigma_gt;     vol_gt_ssim(~mask) = 0;
[ssim_val, ssim_map] = ssim(vol_rec_ssim, vol_gt_ssim, 'DynamicRange', data_range);

fprintf('Metrics -> MAE: %.4f, RMSE: %.4f, SSIM: %.4f\n', mae_val, rmse_val, ssim_val);

% 保存单独的 Metric 文件
metric_struct = struct();
metric_struct.CaseID = caseName;
metric_struct.TaskID = task_id;  % 【修改】保存 task_id 到指标文件，方便 Log 追踪
metric_struct.SNR = this_snr;
metric_struct.MAE = mae_val;
metric_struct.RMSE = rmse_val;
metric_struct.SSIM = ssim_val;

% 文件名包含 CaseName，避免冲突
save(fullfile(metricsDir, sprintf('metric_%s.mat', caseName)), 'metric_struct');

% --- 5.7 可视化 ---
if doPlotting
    map_radius = single(label_final);
    map_mae    = abs(cond_optimal - sigma_gt); map_mae(~mask) = 0;
    map_rmse   = map_mae.^2;
    map_ssim   = ssim_map; map_ssim(~mask) = 0;
    center = round([nx, ny, nz] ./ 2);
    title_str = sprintf('SNR: %.1f | MAE: %.3f', this_snr, mae_val);

    plot_metric_view(map_radius, map_mae, map_rmse, map_ssim, center, 3, 'Axial', title_str, caseName, this_snr, figuresDir);
    plot_metric_view(map_radius, map_mae, map_rmse, map_ssim, center, 1, 'Sagittal', title_str, caseName, this_snr, figuresDir);
    plot_metric_view(map_radius, map_mae, map_rmse, map_ssim, center, 2, 'Coronal', title_str, caseName, this_snr, figuresDir);

    % Compare Plot
    h4 = figure('Visible', 'off'); set(h4, 'Position', [100, 100, 1000, 400]);
    subplot(1,3,1); imagesc(sigma_gt(:,:,center(3))); axis image off; colormap(gca, 'jet'); colorbar; title('Ground Truth'); caxis([0 2.5]);
    subplot(1,3,2); imagesc(cond_optimal(:,:,center(3))); axis image off; colormap(gca, 'jet'); colorbar; title('Optimal Recon'); caxis([0 2.5]);
    subplot(1,3,3); imagesc(map_mae(:,:,center(3))); axis image off; colormap(gca, 'hot'); colorbar; title('Abs Error'); caxis([0 1]);
    sgtitle(['Compare - ', title_str]);
    fNameComp = sprintf('%s_SNR%03d_Compare.png', caseName, round(this_snr));
    saveas(h4, fullfile(figuresDir, fNameComp));
    close(h4);
end

fprintf('任务 %d (Case: %s) 完成。\n', task_id, caseName);