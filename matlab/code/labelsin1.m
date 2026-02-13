%% ===================== Phase 2: SGE Job Array Version (Myriad Fixed) =====================
% Modification: Fixed Indexing Logic & Removed ACFS Backup
% Logic: Each job processes ONE file based on SGE_TASK_ID explicitly.

clear; clc;
% close all; % 集群模式下保持注释
warning('off', 'backtrace'); % 精简警告输出，避免长调用栈刷屏

% --- 1. 路径配置 ---
addpath('/myriadfs/home/zcemexx/projects/MREPT_code/matlab/functions');
addpath(genpath('/myriadfs/home/zcemexx/projects/MREPT_code/matlab/toolboxes'));

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
plotCacheDir = fullfile(metricsDir, 'plot_cache');

% 确保文件夹存在
if ~isfolder(baseDir); mkdir(baseDir); end
if ~isfolder(imagesTr); mkdir(imagesTr); end
if ~isfolder(labelsTr); mkdir(labelsTr); end
if ~isfolder(metricsDir); mkdir(metricsDir); end
if ~isfolder(noisyDataDir); mkdir(noisyDataDir); end
if ~isfolder(figuresDir); mkdir(figuresDir); end
if ~isfolder(plotCacheDir); mkdir(plotCacheDir); end

% --- 2. 获取当前任务 ID ---
task_id_str = getenv('SGE_TASK_ID');

if isempty(task_id_str) || strcmp(task_id_str, 'undefined')
    task_id = 1;
    fprintf('警告: 未检测到 SGE 阵列 ID，默认运行第 1 个文件 (本地/单机测试模式)\n');
else
    task_id = str2double(task_id_str);
    fprintf('检测到 SGE Job Array，正在处理第 %d 个任务\n', task_id);
end

% --- 3. 【核心修复】直接通过 Task ID 构造文件名 ---
% 旧逻辑使用 dir() 会导致字母顺序排序 (M1, M10, M100...)，导致 Task ID 与文件名不匹配。
% 新逻辑：强制 Task 1 -> M1.mat, Task 84 -> M84.mat

caseName = sprintf('M%d', task_id);       % 例如: M84
currentFileName = [caseName, '.mat'];     % 例如: M84.mat
currentFilePath = fullfile(inputDataDir, currentFileName);

% 检查文件是否存在，防止 Task ID 超出范围或文件缺失
if ~isfile(currentFilePath)
    fprintf('错误：文件 %s 不存在！\n', currentFilePath);
    % 尝试以 3 位数字格式再次检查 (例如 M084.mat)
    caseNameAlt = sprintf('M%03d', task_id);
    currentFileNameAlt = [caseNameAlt, '.mat'];
    currentFilePathAlt = fullfile(inputDataDir, currentFileNameAlt);

    if isfile(currentFilePathAlt)
        fprintf('找到替代文件名: %s\n', currentFileNameAlt);
        currentFilePath = currentFilePathAlt;
        caseName = caseNameAlt;
        currentFileName = currentFileNameAlt;
    else
        error('无法找到对应 Task ID %d 的输入文件。请检查文件名格式 (M1.mat vs M001.mat)。', task_id);
    end
end

fprintf('正在处理 Case: %s (Task ID: %d)\n', caseName, task_id);

% --- 4. 并行池配置 ---
poolObj = gcp('nocreate');
if isempty(poolObj)
    nSlotsStr = getenv('NSLOTS');
    nSlots = str2double(nSlotsStr);

    if isnan(nSlots) || nSlots < 1
        nSlots = 1;
    end

    if nSlots > 1
        % 使用 local profile
        parpool('local', nSlots);
        fprintf('并行池已启动，核心数 (NSLOTS): %d\n', nSlots);
    else
        fprintf('运行在单核模式。\n');
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
quietMappingLog = true; % true: 抑制 conductivityMapping 在 parfor 中的重复输出
snr_range_linear = [10, 150];
snr_range_log    = log10(snr_range_linear);
radius_list      = 1:30;
nR               = numel(radius_list);

% ===================== 处理逻辑 =====================

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

% --- 5.2 加噪 ---
case_num = sscanf(caseName, 'M%d');
if isempty(case_num)
    warning('无法从 caseName=%s 解析 case number，回退使用 task_id=%d 作为随机种子。', caseName, task_id);
    case_num = task_id;
end
rng(case_num); % 固定随机种子（按 case number）
fprintf('随机种子 (case number): %d\n', case_num);
this_snr_log = snr_range_log(1) + (snr_range_log(2)-snr_range_log(1)) * rand();
this_snr     = 10^this_snr_log;

complex_signal = magnitude_raw .* exp(1i * phi0_raw);
noisy_complex = gen_noise_complex(complex_signal, this_snr, mask);

phi0_noisy      = angle(noisy_complex);
magnitude_noisy = abs(noisy_complex);

% 保存 Noisy Data (到 Scratch)
nFileName = strrep(currentFileName, 'M', 'N');
if ~startsWith(nFileName, 'N'); nFileName = ['N_', currentFileName]; end
save(fullfile(noisyDataDir, nFileName), 'phi0_noisy', 'magnitude_noisy', 'tissueMask', 'this_snr', 'task_id');

% --- 5.3 穷举搜索 ---
err_stack  = inf(nx, ny, nz, nR, 'single');
cond_stack = zeros(nx, ny, nz, nR, 'single');
Parameters.B0 = 3; Parameters.VoxelSize = [1 1 1];

parfor ir = 1:nR
    r = radius_list(ir);
    params_r = Parameters;
    params_r.kDiffSize = [2*r+1, 2*r+1, 2*r+1];

    % 注意：确保你的 conductivityMapping 函数能处理 inf/nan
    if quietMappingLog
        warnState = warning;
        warning('off', 'all');
        try
            evalc('[cond_r, ~] = conductivityMapping(phi0_noisy, mask, params_r, ''magnitude'', magnitude_noisy, ''segmentation'', tissueMask, ''estimatenoise'', estimatenoise);');
        catch ME
            warning(warnState);
            rethrow(ME);
        end
        warning(warnState);
    else
        [cond_r, ~] = conductivityMapping(phi0_noisy, mask, params_r, ...
            'magnitude', magnitude_noisy, 'segmentation', tissueMask, 'estimatenoise', estimatenoise);
    end

    if do_filter; cond_r(cond_r < 0 | cond_r > 10) = NaN; end

    cond_stack(:,:,:,ir) = cond_r;

    diff_val = abs(cond_r - sigma_gt);
    bad_pixels = ~mask | isnan(cond_r);

    if doSmoothing
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

% --- 5.5 保存 NIfTI (到 Scratch) ---
save_nii_gz(phi0_noisy,  fullfile(imagesTr, [caseName, '_0000.nii.gz']));
save_nii_gz(tissueMask,  fullfile(imagesTr, [caseName, '_0001.nii.gz']));
save_nii_gz(label_final, fullfile(labelsTr, [caseName, '.nii.gz']));

% --- 5.6 Metrics & Log ---
mae_val = mean(min_err(mask));
diff_sq = (cond_optimal - sigma_gt).^2;
rmse_val = sqrt(mean(diff_sq(mask)));

data_range = max(sigma_gt(mask)) - min(sigma_gt(mask));
if isempty(data_range) || data_range == 0; data_range = 1; end
vol_rec_ssim = cond_optimal; vol_rec_ssim(~mask) = 0;
vol_gt_ssim  = sigma_gt;     vol_gt_ssim(~mask) = 0;
[ssim_val, ssim_map] = ssim(vol_rec_ssim, vol_gt_ssim, 'DynamicRange', data_range);

fprintf('Metrics -> MAE: %.4f, RMSE: %.4f, SSIM: %.4f\n', mae_val, rmse_val, ssim_val);

metric_struct = struct();
metric_struct.CaseID = caseName;
metric_struct.TaskID = task_id;
metric_struct.SNR = this_snr;
metric_struct.MAE = mae_val;
metric_struct.RMSE = rmse_val;
metric_struct.SSIM = ssim_val;

% 保存 Metrics (到 Scratch)
save(fullfile(metricsDir, sprintf('metric_%s.mat', caseName)), 'metric_struct');

% --- 5.7 可视化 ---
if doPlotting
    map_radius = single(label_final);
    map_mae    = abs(cond_optimal - sigma_gt); map_mae(~mask) = 0;
    map_rmse   = map_mae.^2;
    map_ssim   = ssim_map; map_ssim(~mask) = 0;
    if any(mask(:))
        mae_vals  = map_mae(mask);
        rmse_vals = map_rmse(mask);
        ssim_vals = map_ssim(mask);
        metricLimits = [
            min(mae_vals),  max(mae_vals);
            min(rmse_vals), max(rmse_vals);
            min(ssim_vals), max(ssim_vals)
        ];
    else
        metricLimits = [0, 1; 0, 1; 0, 1];
    end

    for ii = 1:3
        if ~all(isfinite(metricLimits(ii,:)))
            metricLimits(ii,:) = [0, 1];
        elseif metricLimits(ii,1) == metricLimits(ii,2)
            metricLimits(ii,2) = metricLimits(ii,1) + 1e-6;
        end
    end

    center = round([nx, ny, nz] ./ 2);
    title_str = sprintf('SNR: %.1f | MAE: %.3f | RMSE: %.3f | SSIM: %.3f', this_snr, mae_val, rmse_val, ssim_val);

    % 保存重绘缓存：用于后处理阶段按全体 case 的全局色轴统一重绘
    viewDims = [3, 1, 2];
    viewNames = {'Axial', 'Sagittal', 'Coronal'};
    plotCache = struct();
    plotCache.caseName = caseName;
    plotCache.snr = this_snr;
    plotCache.title = title_str;
    plotCache.metricLimitsLocal = metricLimits;
    for iv = 1:3
        dimNow = viewDims(iv);
        vStruct = struct();
        vStruct.name = viewNames{iv};
        vStruct.radius = squeeze_slice(map_radius, dimNow, center);
        vStruct.mae = squeeze_slice(map_mae, dimNow, center);
        vStruct.rmse = squeeze_slice(map_rmse, dimNow, center);
        vStruct.ssim = squeeze_slice(map_ssim, dimNow, center);
        plotCache.views(iv) = vStruct;
    end
    save(fullfile(plotCacheDir, sprintf('plotcache_%s.mat', caseName)), 'plotCache');

    % 确保 plot_metric_view 能够接受路径
    plot_metric_view(map_radius, map_mae, map_rmse, map_ssim, center, 3, 'Axial', title_str, caseName, this_snr, figuresDir, metricLimits);
    plot_metric_view(map_radius, map_mae, map_rmse, map_ssim, center, 1, 'Sagittal', title_str, caseName, this_snr, figuresDir, metricLimits);
    plot_metric_view(map_radius, map_mae, map_rmse, map_ssim, center, 2, 'Coronal', title_str, caseName, this_snr, figuresDir, metricLimits);

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
