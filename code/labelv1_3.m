%% ===================== Phase 2: Professional Batch Label Generation (Final) =====================
% 修改说明：
% 1. 完整集成 MAE, RMSE, SSIM 指标计算
% 2. 生成四张图：Axial/Sagittal/Coronal (含指标) + GT对比图
% 3. 自动生成 metadata.csv 汇总表
% 4. 优化了可视化配色和文件名格式 (含 SNR)

clear; clc; close all;
% 确保工具箱在路径中 (假设你有 nifti 工具箱和 EPT 函数)
addpath(genpath('functions'));
addpath(genpath('toolboxes'));

%% 1. 路径与全局配置
% 自动根据集群环境启动并行池
poolObj = gcp('nocreate'); % 尝试获取当前已有的并行池
if isempty(poolObj)
    nSlots = str2double(getenv('NSLOTS')); % 获取 SGE 申请的核心数
    if isnan(nSlots), nSlots = 1; end      % 如果在本地运行而非集群，默认 1
    
    if nSlots > 1
        parpool('local', nSlots);
    end
end
% --- 核心功能开关 ---
doSmoothing   = false;  % true: 使用组织感知平滑误差; false: 原始体素误差
dorepair      = false;  % true: MAD 异常值修复
doSaveMetrics = true;   % true: 保存 .mat 指标文件
doPlotting    = true;   % true: 生成并保存可视化图片 (4张/Case)
doSubVolume   = false;   % [测试用] true: 只处理中心一小块; [正式跑] 设为 false
subVolSize    = 32;     
do_filter     = true;   % [改名] 避免与内置函数冲突，原 isfilter, filter s>10&s<0
estimatenoise = true;   
snr_range_linear = [10, 150];
snr_range_log    = log10(snr_range_linear);

projectRoot = pwd;
datasetName  = 'Dataset001_EPT';
baseDir      = fullfile(projectRoot, 'nnUNet_raw');
inputDataDir = fullfile(projectRoot, 'data', 'ADEPT_raw');      
noisyDataDir = fullfile(projectRoot, 'data', 'ADEPT_noisy');    
outputBase   = fullfile(baseDir, datasetName);

imagesTr     = fullfile(outputBase, 'imagesTr');
labelsTr     = fullfile(outputBase, 'labelsTr');
metricsDir   = fullfile(baseDir, 'metrics'); 
figuresDir   = fullfile(metricsDir, 'figures'); 

% 创建文件夹
if ~isfolder(baseDir); mkdir(baseDir); end
% if ~isfolder(inputDataDir); mkdir(inputDataDir); end
if ~isfolder(imagesTr); mkdir(imagesTr); end
if ~isfolder(labelsTr); mkdir(labelsTr); end
if ~isfolder(metricsDir); mkdir(metricsDir); end
if ~isfolder(noisyDataDir); mkdir(noisyDataDir); end
if doPlotting && ~isfolder(figuresDir); mkdir(figuresDir); end

% 获取文件列表
fileList = dir(fullfile(inputDataDir, 'M*.mat'));
radius_list = 1:30; % 半径范围
nR = numel(radius_list);

% --- 初始化汇总 Log ---
% 列: CaseName, SNR, MAE, RMSE, SSIM
metadata_log = cell(numel(fileList), 5); 
fprintf('找到 %d 个数据集，开始批量处理...\n', numel(fileList));

%% 2. 批量处理循环
for f = 1:numel(fileList)
    currentFile = fullfile(inputDataDir, fileList(f).name);
    [~, fileName, ~] = fileparts(fileList(f).name);
    caseName = fileName; 
    
    fprintf('\n[Case %d/%d] 处理中: %s\n', f, numel(fileList), caseName);
    
    % --- 2.1 加载数据 ---
    data = load(currentFile);
    
    if isfield(data, 'Transceive_phase'), phi0_raw = single(data.Transceive_phase);
    else, error('Missing Transceive_phase'); end
    
    if isfield(data, 'Conductivity_GT'), sigma_gt = single(data.Conductivity_GT);
    else, error('Missing Conductivity_GT'); end
    
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
    
    % --- 2.2 加噪 ---
    this_snr_log = snr_range_log(1) + (snr_range_log(2)-snr_range_log(1)) * rand();
    this_snr     = 10^this_snr_log;
    
    complex_signal = magnitude_raw .* exp(1i * phi0_raw);
    noisy_complex = gen_noise_complex(complex_signal, this_snr, mask);
    
    phi0_noisy      = angle(noisy_complex);
    magnitude_noisy = abs(noisy_complex);
    
    fprintf('   -> SNR: %.2f\n', this_snr);
    
    nFileName = strrep(fileName, 'M', 'N');
    if ~startsWith(nFileName, 'N'); nFileName = ['N_', fileName]; end
    save(fullfile(noisyDataDir, [nFileName, '.mat']), 'phi0_noisy', 'magnitude_noisy', 'tissueMask', 'this_snr');
    
    % --- 2.3 穷举搜索 & 重建 ---
    err_stack  = inf(nx, ny, nz, nR, 'single');
    % 暂存重建结果以计算最终的 RMSE 和 SSIM
    cond_stack = zeros(nx, ny, nz, nR, 'single'); 
    
    Parameters.B0 = 3;
    Parameters.VoxelSize = [1 1 1];
    
    for ir = 1:nR
        r = radius_list(ir);
        params_r = Parameters;
        params_r.kDiffSize = [2*r+1, 2*r+1, 2*r+1];
        
        [cond_r, ~] = conductivityMapping(phi0_noisy, mask, params_r, ...
            'magnitude', magnitude_noisy, 'segmentation', tissueMask, 'estimatenoise', estimatenoise);
        
        if do_filter; cond_r(cond_r < 0 | cond_r > 10) = NaN; end
        
        % 保存重建结果到 stack
        cond_stack(:,:,:,ir) = cond_r;
        
        diff_val = abs(cond_r - sigma_gt);
        bad_pixels = ~mask | isnan(cond_r);
        
        if doSmoothing
            diff_val(bad_pixels) = 0; 
            err_stack(:,:,:,ir) = masked_smooth_error(diff_val, tissueMask);
        else
            diff_val(bad_pixels) = inf;
            err_stack(:,:,:,ir) = diff_val;
        end
    end
    
    % --- 2.4 生成标签 ---
    [min_err, best_idx_voxel] = min(err_stack, [], 4);
    
    radiusMap_raw = zeros(size(phi0_raw), 'single');
    radiusMap_raw(mask) = single(radius_list(best_idx_voxel(mask)));
    
    % --- 合成最优重建图像 (Optimal Reconstruction) ---
    [gridX, gridY, gridZ] = ndgrid(1:nx, 1:ny, 1:nz);
    linear_idx_optimal = sub2ind([nx, ny, nz, nR], gridX, gridY, gridZ, best_idx_voxel);
    cond_optimal = cond_stack(linear_idx_optimal);
    clear cond_stack gridX gridY gridZ linear_idx_optimal; % 释放内存
    
    % --- 2.5 标签精炼 ---
    if dorepair
        fprintf('   -> 修复异常点...\n');
        radiusMap_refined = refine_labels_mad(radiusMap_raw, tissueMask, radius_list);
    else
        radiusMap_refined = radiusMap_raw;
    end
    label_final = uint8(round(radiusMap_refined));
    
    % --- 2.6 保存 NIfTI ---
    save_nii_gz(phi0_noisy,      fullfile(imagesTr, [caseName, '_0000.nii.gz']));
    save_nii_gz(magnitude_noisy, fullfile(imagesTr, [caseName, '_0001.nii.gz']));
    save_nii_gz(tissueMask,      fullfile(imagesTr, [caseName, '_0002.nii.gz']));
    save_nii_gz(label_final,     fullfile(labelsTr, [caseName, '.nii.gz']));
    
    % --- 2.7 计算 Metrics (MAE, RMSE, SSIM) ---
    % 1. MAE (Masked)
    mae_val = mean(min_err(mask)); 
    
    % 2. RMSE (Masked)
    diff_sq = (cond_optimal - sigma_gt).^2;
    rmse_val = sqrt(mean(diff_sq(mask)));
    
    % 3. SSIM (Masked Logic)
    data_range = max(sigma_gt(mask)) - min(sigma_gt(mask));
    if isempty(data_range) || data_range == 0; data_range = 1; end
    
    % 背景置0，避免噪声干扰 SSIM
    vol_rec_ssim = cond_optimal; vol_rec_ssim(~mask) = 0;
    vol_gt_ssim  = sigma_gt;     vol_gt_ssim(~mask) = 0;
    
    [ssim_val, ssim_map] = ssim(vol_rec_ssim, vol_gt_ssim, 'DynamicRange', data_range);
    
    fprintf('   -> Metrics: MAE=%.4f, RMSE=%.4f, SSIM=%.4f\n', mae_val, rmse_val, ssim_val);
    
    % 保存 Metrics Summary
    if doSaveMetrics
        plotData = struct();
        plotData.stats.mae = mae_val;
        plotData.stats.rmse = rmse_val;
        plotData.stats.ssim = ssim_val;
        plotData.params = struct('SNR', this_snr, 'case', caseName);
        save(fullfile(metricsDir, [caseName, '_summary.mat']), 'plotData');
    end
    
    % --- 2.8 可视化 (4张图: Ax, Sag, Cor, Compare) ---
    if doPlotting
        % 准备 Map 数据
        map_radius = single(label_final);
        map_mae    = abs(cond_optimal - sigma_gt); map_mae(~mask) = 0;
        map_rmse   = map_mae.^2; % 这里的 Map 展示平方误差
        map_ssim   = ssim_map; map_ssim(~mask) = 0;
        
        center = round([nx, ny, nz] ./ 2);
        
        % 标题包含具体指标
        title_str = sprintf('SNR: %.1f | MAE: %.3f, RMSE: %.3f, SSIM: %.3f', ...
                            this_snr, mae_val, rmse_val, ssim_val);
                        
        % === 图 1-3: 三视图指标分布 ===
        plot_metric_view(map_radius, map_mae, map_rmse, map_ssim, center, 3, 'Axial', title_str, caseName, this_snr, figuresDir);
        plot_metric_view(map_radius, map_mae, map_rmse, map_ssim, center, 1, 'Sagittal', title_str, caseName, this_snr, figuresDir);
        plot_metric_view(map_radius, map_mae, map_rmse, map_ssim, center, 2, 'Coronal', title_str, caseName, this_snr, figuresDir);
        
        % === 图 4: 物理重建对比 (GT vs Recon) ===
        h4 = figure('Visible', 'off'); set(h4, 'Position', [100, 100, 1000, 400]);
        % 截取中心 Axial 切片
        subplot(1,3,1); imagesc(sigma_gt(:,:,center(3))); axis image off; colormap(gca, 'jet'); colorbar; title('Ground Truth'); caxis([0 2.5]);
        subplot(1,3,2); imagesc(cond_optimal(:,:,center(3))); axis image off; colormap(gca, 'jet'); colorbar; title('Optimal Recon'); caxis([0 2.5]);
        subplot(1,3,3); imagesc(map_mae(:,:,center(3))); axis image off; colormap(gca, 'hot'); colorbar; title('Abs Error'); caxis([0 1]);
        sgtitle(['Compare - ', title_str]);
        % 文件名包含 SNR
        fNameComp = sprintf('%s_SNR%03d_Compare.png', caseName, round(this_snr));
        saveas(h4, fullfile(figuresDir, fNameComp));
        close(h4);
    end

    % --- 填入汇总 Log ---
    metadata_log{f, 1} = caseName;
    metadata_log{f, 2} = this_snr;
    metadata_log{f, 3} = mae_val;
    metadata_log{f, 4} = rmse_val;
    metadata_log{f, 5} = ssim_val;
end

%% [新增] 2.9 保存 CSV 汇总表
logTable = cell2table(metadata_log, 'VariableNames', {'CaseID', 'SNR', 'MAE', 'RMSE', 'SSIM'});
writetable(logTable, fullfile(baseDir, 'dataset_metadata.csv'));
fprintf('Metadata CSV 已保存: %s\n', fullfile(baseDir, 'dataset_metadata.csv'));

%% 3. 生成 dataset.json
generate_dataset_json(outputBase, numel(fileList), snr_range_linear);
fprintf('\n全部完成！\n');
