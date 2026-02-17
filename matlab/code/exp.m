%% ===================== Phase 5: List-Based SGE Job Array =====================
% 目标：
% 1) 非连续 Case 列表驱动映射（17 Cases x 8 SNR = 136 任务）
% 2) 每任务独立写入 Phase5 标准产出，避免并发冲突
% 3) 支持聚合模式生成 all_results_summary.csv

clear; clc;
warning('off', 'backtrace');

% --- 1. 路径配置 ---
addpath('/myriadfs/home/zcemexx/projects/MREPT_code/matlab/functions');
addpath(genpath('/myriadfs/home/zcemexx/projects/MREPT_code/matlab/toolboxes'));

projectRoot  = '/myriadfs/home/zcemexx/Scratch';
datasetName  = 'exp001_EPT';
baseDir      = fullfile(projectRoot, 'exp');
inputDataDir = fullfile(projectRoot, 'ADEPT_raw');
outputBase   = fullfile(baseDir, datasetName);

imagesTr     = fullfile(outputBase, 'imagesTr');
labelsTr     = fullfile(outputBase, 'labelsTr');
metricsDir   = fullfile(baseDir, 'metrics');
figuresDir   = fullfile(metricsDir, 'figures');
plotCacheDir = fullfile(metricsDir, 'plot_cache');
phase5Root   = fullfile(projectRoot, 'outputs', 'phase5');
niiRoot      = fullfile(projectRoot, 'outputs', 'nii');

if ~isfolder(baseDir); mkdir(baseDir); end
if ~isfolder(imagesTr); mkdir(imagesTr); end
if ~isfolder(labelsTr); mkdir(labelsTr); end
if ~isfolder(metricsDir); mkdir(metricsDir); end
if ~isfolder(figuresDir); mkdir(figuresDir); end
if ~isfolder(plotCacheDir); mkdir(plotCacheDir); end
if ~isfolder(phase5Root); mkdir(phase5Root); end
if ~isfolder(niiRoot); mkdir(niiRoot); end

% --- 2. 任务定义（列表驱动） ---
case_files = {'M6.mat', 'M8.mat', 'M12.mat', 'M19.mat', 'M22.mat', ...
    'M24.mat', 'M39.mat', 'M40.mat', 'M41.mat', 'M42.mat', ...
    'M43.mat', 'M50.mat', 'M66.mat', 'M70.mat', 'M75.mat', ...
    'M79.mat', 'M84.mat'};
snr_list = [10, 20, 30, 40, 50, 75, 100, 150];

n_cases = numel(case_files);
n_snr = numel(snr_list);
n_total = n_cases * n_snr;

% --- 3. 聚合模式开关 ---
aggregateMode = is_truthy_env(getenv('MREPT_AGGREGATE'));
if aggregateMode
    outCsv = fullfile(phase5Root, 'all_results_summary.csv');
    aggregate_phase5_results(phase5Root, outCsv);
    fprintf('聚合完成：%s\n', outCsv);
    return;
end

% --- 4. 任务解析与映射 ---
combo_idx = parse_task_id(n_total);
mapping = build_task_mapping(case_files, snr_list, combo_idx);

task_id = combo_idx;
this_case_file = mapping.case_file;
this_snr = mapping.snr;
caseName = mapping.case_name;
case_num = mapping.case_num;
currentFilePath = fullfile(inputDataDir, this_case_file);

fprintf('Task %d/%d: Processing %s at SNR %d\n', task_id, n_total, this_case_file, this_snr);

% --- 5. 并行池配置 ---
poolObj = gcp('nocreate');
if isempty(poolObj)
    nSlotsStr = getenv('NSLOTS');
    nSlots = str2double(nSlotsStr);
    if isnan(nSlots) || nSlots < 1
        nSlots = 1;
    end
    if nSlots > 1
        parpool('local', nSlots);
        fprintf('并行池已启动，核心数 (NSLOTS): %d\n', nSlots);
    else
        fprintf('运行在单核模式。\n');
    end
end

% --- 6. 参数配置 ---
doSmoothing      = false;
dorepair         = false;
doPlotting       = true;
doSubVolume      = false;
subVolSize       = 32;
do_filter        = true;
estimatenoise    = true;
quietMappingLog  = true;
radius_list      = 1:30;
nR               = numel(radius_list);
methodName       = 'Oracle';

% ===================== 主处理逻辑 =====================

% --- 6.1 加载数据（鲁棒） ---
[phi0_raw, sigma_gt, tissueMask, magnitude_raw] = safe_load_case(currentFilePath);

if doSubVolume
    sz = size(phi0_raw);
    cp = round(sz / 2);
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
if ~any(mask(:))
    error('Case %s 的 mask 为空，无法计算有效指标。', caseName);
end

% --- 6.2 固定种子 + 固定 SNR ---
rng(case_num * 1000);
fprintf('随机种子: %d | 固定 SNR: %d\n', case_num * 1000, this_snr);

complex_signal = magnitude_raw .* exp(1i * phi0_raw);
noisy_complex = gen_noise_complex(complex_signal, this_snr, mask);
phi0_noisy = angle(noisy_complex);
magnitude_noisy = abs(noisy_complex);

% --- 6.3 穷举搜索 ---
err_stack  = inf(nx, ny, nz, nR, 'single');
cond_stack = zeros(nx, ny, nz, nR, 'single');
Parameters.B0 = 3;
Parameters.VoxelSize = [1 1 1];

parfor ir = 1:nR
    r = radius_list(ir);
    params_r = Parameters;
    params_r.kDiffSize = [2*r+1, 2*r+1, 2*r+1];

    if quietMappingLog
        warnState = warning;
        warning('off', 'all');
        try
            [cond_r, ~] = conductivityMapping(phi0_noisy, mask, params_r, ...
                'magnitude', magnitude_noisy, 'segmentation', tissueMask, 'estimatenoise', estimatenoise);
        catch ME
            warning(warnState);
            rethrow(ME);
        end
        warning(warnState);
    else
        [cond_r, ~] = conductivityMapping(phi0_noisy, mask, params_r, ...
            'magnitude', magnitude_noisy, 'segmentation', tissueMask, 'estimatenoise', estimatenoise);
    end

    if do_filter
        cond_r(cond_r < 0 | cond_r > 10) = NaN;
    end

    cond_stack(:,:,:,ir) = cond_r;

    diff_val = abs(cond_r - sigma_gt);
    bad_pixels = ~mask | ~isfinite(cond_r);

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

% --- 6.4 生成标签与最优重建 ---
[min_err, best_idx_voxel] = min(err_stack, [], 4);
radiusMap_raw = zeros(size(phi0_raw), 'single');
radiusMap_raw(mask) = single(radius_list(best_idx_voxel(mask)));

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

% --- 6.5 Metrics ---
[res, ssim_map] = compute_metrics(cond_optimal, sigma_gt, tissueMask, mask);
res.Case_ID = caseName;
res.SNR = this_snr;
res.Method = methodName;
res.TaskID = task_id;

fprintf('Metrics -> Mean_MAE: %.4f, RMSE: %.4f, SSIM: %.4f\n', res.Mean_MAE, res.RMSE, res.SSIM);

% --- 6.6 Phase5 输出（每任务独立目录） ---
snrTag = sprintf('SNR%03d', round(this_snr));
caseTag = sprintf('%s_SNR%03d', caseName, round(this_snr));
taskOutDir = fullfile(phase5Root, caseName, snrTag);
if ~isfolder(taskOutDir)
    mkdir(taskOutDir);
end

save(fullfile(taskOutDir, sprintf('noisy_phase_SNR%03d.mat', round(this_snr))), ...
    'phi0_noisy', 'magnitude_noisy', 'tissueMask', 'this_snr', 'task_id', 'case_num');
save(fullfile(taskOutDir, 'pred_kernel_map.mat'), 'radiusMap_refined', 'label_final');
save(fullfile(taskOutDir, 'sigma_reconstructed.mat'), 'cond_optimal');
error_map = abs(cond_optimal - sigma_gt);
error_map(~mask) = 0;
save(fullfile(taskOutDir, 'mae_map.mat'), 'error_map');
save(fullfile(taskOutDir, 'ssim_map.mat'), 'ssim_map');
write_metrics_json(fullfile(taskOutDir, 'metrics.json'), res);

% --- 6.6b NIfTI 输出（便于下游可视化与检查） ---
taskNiiDir = fullfile(niiRoot, caseName, snrTag);
if ~isfolder(taskNiiDir)
    mkdir(taskNiiDir);
end
save_nii_gz(phi0_noisy,  fullfile(imagesTr, [caseTag, '_0000.nii.gz']));
save_nii_gz(tissueMask,  fullfile(imagesTr, [caseTag, '_0001.nii.gz']));
save_nii_gz(label_final,       fullfile(taskNiiDir, 'pred_kernel_map_label_final.nii.gz'));
save_nii_gz(cond_optimal,      fullfile(taskNiiDir, 'sigma_reconstructed_cond_optimal.nii.gz'));
save_nii_gz(error_map,         fullfile(taskNiiDir, 'mae_map_error_map.nii.gz'));
ssim_out = ssim_map;
ssim_out(~mask) = 0;
save_nii_gz(ssim_out,          fullfile(taskNiiDir, 'ssim_map.nii.gz'));

% --- 6.8 可视化 ---
if doPlotting
    map_radius = single(label_final);
    map_mae = error_map;
    map_rmse = map_mae.^2;
    map_ssim = ssim_map;
    map_ssim(~mask) = 0;

    if any(mask(:))
        mae_vals = map_mae(mask & isfinite(map_mae));
        rmse_vals = map_rmse(mask & isfinite(map_rmse));
        ssim_vals = map_ssim(mask & isfinite(map_ssim));

        metricLimits = [
            min_or_default(mae_vals, 0),  max_or_default(mae_vals, 1);
            min_or_default(rmse_vals, 0), max_or_default(rmse_vals, 1);
            min_or_default(ssim_vals, 0), max_or_default(ssim_vals, 1)
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
    title_str = sprintf('SNR: %.1f | MAE: %.3f | RMSE: %.3f | SSIM: %.3f', ...
        this_snr, res.Mean_MAE, res.RMSE, res.SSIM);

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
    save(fullfile(plotCacheDir, sprintf('plotcache_%s_%s.mat', caseName, snrTag)), 'plotCache');

    plot_metric_view(map_radius, map_mae, map_rmse, map_ssim, center, 3, 'Axial', title_str, caseTag, this_snr, figuresDir, metricLimits);
    plot_metric_view(map_radius, map_mae, map_rmse, map_ssim, center, 1, 'Sagittal', title_str, caseTag, this_snr, figuresDir, metricLimits);
    plot_metric_view(map_radius, map_mae, map_rmse, map_ssim, center, 2, 'Coronal', title_str, caseTag, this_snr, figuresDir, metricLimits);

    h4 = figure('Visible', 'off');
    set(h4, 'Position', [100, 100, 1000, 400]);
    subplot(1,3,1); imagesc(sigma_gt(:,:,center(3))); axis image off; colormap(gca, 'jet'); colorbar; clim([0 2.5]); title('Ground Truth');
    subplot(1,3,2); imagesc(cond_optimal(:,:,center(3))); axis image off; colormap(gca, 'jet'); colorbar; clim([0 2.5]); title('Optimal Recon');
    subplot(1,3,3); imagesc(map_mae(:,:,center(3))); axis image off; colormap(gca, 'hot'); colorbar; clim([0 1]); title('Abs Error');
    sgtitle(['Compare - ', title_str]);
    saveas(h4, fullfile(figuresDir, sprintf('%s_Compare.png', caseTag)));
    close(h4);
end

fprintf('任务 %d 完成: %s @ SNR=%d\n', task_id, caseName, this_snr);

%% ===================== Local Functions =====================
function combo_idx = parse_task_id(n_total)
task_id_str = getenv('SGE_TASK_ID');
if isempty(task_id_str) || strcmpi(task_id_str, 'undefined')
    combo_idx = 1;
    fprintf('Warning: SGE_TASK_ID not found. Local debug mode -> ID=1\n');
    return;
end

combo_idx = str2double(task_id_str);
if ~isfinite(combo_idx) || combo_idx ~= floor(combo_idx)
    error('Invalid SGE_TASK_ID=%s. It must be an integer in [1, %d].', task_id_str, n_total);
end
if combo_idx < 1 || combo_idx > n_total
    error('Invalid SGE_TASK_ID=%d. Range must be 1-%d.', combo_idx, n_total);
end
end

function mapping = build_task_mapping(case_files, snr_list, combo_idx)
n_cases = numel(case_files);
n_snr = numel(snr_list);

case_idx = floor((combo_idx - 1) / n_snr) + 1;
snr_idx = mod(combo_idx - 1, n_snr) + 1;

this_case_file = case_files{case_idx};
this_snr = snr_list(snr_idx);
[~, caseName, ext] = fileparts(this_case_file);

if ~strcmpi(ext, '.mat')
    error('Case file must end with .mat, got: %s', this_case_file);
end

case_num = sscanf(this_case_file, 'M%d.mat');
if isempty(case_num)
    error('Unable to parse case number from file name: %s', this_case_file);
end

mapping = struct();
mapping.case_idx = case_idx;
mapping.snr_idx = snr_idx;
mapping.case_file = this_case_file;
mapping.case_name = caseName;
mapping.case_num = case_num;
mapping.snr = this_snr;
end

function [phi0_raw, sigma_gt, tissueMask, magnitude_raw] = safe_load_case(file_path)
if ~isfile(file_path)
    error('Input case file not found: %s', file_path);
end

data = load(file_path);

if ~isfield(data, 'Transceive_phase')
    error('Missing required field: Transceive_phase in %s', file_path);
end
if ~isfield(data, 'Conductivity_GT')
    error('Missing required field: Conductivity_GT in %s', file_path);
end
if ~isfield(data, 'Segmentation')
    error('Missing required field: Segmentation in %s', file_path);
end

phi0_raw = single(data.Transceive_phase);
sigma_gt = single(data.Conductivity_GT);
tissueMask = uint8(data.Segmentation);

if isfield(data, 'B1plus_mag') && isfield(data, 'B1minus_mag')
    magnitude_raw = single(data.B1plus_mag .* data.B1minus_mag);
elseif isfield(data, 'B1plus_mag')
    magnitude_raw = single(data.B1plus_mag);
elseif isfield(data, 'T1w')
    magnitude_raw = single(data.T1w);
else
    error('Missing magnitude source in %s (need B1plus_mag/B1minus_mag or T1w)', file_path);
end

if ~isequal(size(phi0_raw), size(sigma_gt), size(tissueMask), size(magnitude_raw))
    error('Shape mismatch in %s: phase/gt/seg/magnitude must have same size.', file_path);
end
end

function [res, ssim_map] = compute_metrics(cond_optimal, sigma_gt, seg, mask)
valid_mask = mask & isfinite(cond_optimal) & isfinite(sigma_gt);
if ~any(valid_mask(:))
    error('No valid voxels for metrics after finite-value filtering.');
end

error_map = abs(cond_optimal - sigma_gt);
roi_errors = error_map(valid_mask);

data_range = max(sigma_gt(valid_mask)) - min(sigma_gt(valid_mask));
if isempty(data_range) || data_range == 0 || ~isfinite(data_range)
    data_range = 1;
end

vol_rec_ssim = cond_optimal;
vol_gt_ssim = sigma_gt;
vol_rec_ssim(~mask | ~isfinite(vol_rec_ssim)) = 0;
vol_gt_ssim(~mask | ~isfinite(vol_gt_ssim)) = 0;
[ssim_val, ssim_map] = ssim(vol_rec_ssim, vol_gt_ssim, 'DynamicRange', data_range);

res = struct();
res.Mean_MAE = mean(roi_errors);
res.Median_MAE = median(roi_errors);
res.RMSE = sqrt(mean(roi_errors .^ 2));
res.SSIM = ssim_val;

res.P10 = prctile(roi_errors, 10);
res.P25 = prctile(roi_errors, 25);
res.P50 = prctile(roi_errors, 50);
res.P75 = prctile(roi_errors, 75);
res.P90 = prctile(roi_errors, 90);

wm_mask = valid_mask & (seg == 1);
gm_mask = valid_mask & (seg == 2);
csf_mask = valid_mask & (seg == 3);

if any(wm_mask(:)); res.WM_MAE = mean(error_map(wm_mask)); else; res.WM_MAE = NaN; warning('WM label(1) missing.'); end
if any(gm_mask(:)); res.GM_MAE = mean(error_map(gm_mask)); else; res.GM_MAE = NaN; warning('GM label(2) missing.'); end
if any(csf_mask(:)); res.CSF_MAE = mean(error_map(csf_mask)); else; res.CSF_MAE = NaN; warning('CSF label(3) missing.'); end
end

function write_metrics_json(out_path, res)
res.Timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
txt = jsonencode(res, 'PrettyPrint', true);
fid = fopen(out_path, 'w');
if fid == -1
    error('Cannot open metrics json for writing: %s', out_path);
end
cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s', txt);
end

function aggregate_phase5_results(root_dir, out_csv)
json_files = list_metrics_json_files(root_dir);
if isempty(json_files)
    error('No metrics.json found under: %s', root_dir);
end

rows = cell(numel(json_files), 15);
case_nums = inf(numel(json_files), 1);
snr_vals = inf(numel(json_files), 1);

for i = 1:numel(json_files)
    raw = fileread(json_files{i});
    m = jsondecode(raw);

    case_id = get_field_or_default(m, 'Case_ID', 'NA');
    snr = get_field_or_default(m, 'SNR', NaN);
    method = get_field_or_default(m, 'Method', 'Oracle');

    case_num = sscanf(char(case_id), 'M%d');
    if ~isempty(case_num)
        case_nums(i) = case_num;
    end
    if isfinite(snr)
        snr_vals(i) = snr;
    end

    rows{i,1}  = case_id;
    rows{i,2}  = snr;
    rows{i,3}  = method;
    rows{i,4}  = get_field_or_default(m, 'Mean_MAE', NaN);
    rows{i,5}  = get_field_or_default(m, 'Median_MAE', NaN);
    rows{i,6}  = get_field_or_default(m, 'RMSE', NaN);
    rows{i,7}  = get_field_or_default(m, 'SSIM', NaN);
    rows{i,8}  = get_field_or_default(m, 'P10', NaN);
    rows{i,9}  = get_field_or_default(m, 'P90', NaN);
    rows{i,10} = get_field_or_default(m, 'P25', NaN);
    rows{i,11} = get_field_or_default(m, 'P75', NaN);
    rows{i,12} = get_field_or_default(m, 'WM_MAE', NaN);
    rows{i,13} = get_field_or_default(m, 'GM_MAE', NaN);
    rows{i,14} = get_field_or_default(m, 'CSF_MAE', NaN);
    rows{i,15} = json_files{i};
end

[~, ord] = sortrows([case_nums, snr_vals], [1, 2]);
rows = rows(ord, :);

header = {'Case_ID', 'SNR', 'Method', 'Mean_MAE', 'Median_MAE', 'RMSE', 'SSIM', ...
    '10th_Perc', '90th_Perc', 'P25', 'P75', 'WM_MAE', 'GM_MAE', 'CSF_MAE', 'Source_JSON'};

out_cells = [header; rows];
writecell(out_cells, out_csv);
end

function files = list_metrics_json_files(root_dir)
files = {};
entries = dir(root_dir);
for i = 1:numel(entries)
    name = entries(i).name;
    if strcmp(name, '.') || strcmp(name, '..')
        continue;
    end
    full_path = fullfile(root_dir, name);
    if entries(i).isdir
        sub = list_metrics_json_files(full_path);
        files = [files; sub]; %#ok<AGROW>
    else
        if strcmp(name, 'metrics.json')
            files{end+1,1} = full_path; %#ok<AGROW>
        end
    end
end
end

function value = get_field_or_default(s, field_name, default_val)
if isfield(s, field_name)
    value = s.(field_name);
else
    value = default_val;
end
end

function tf = is_truthy_env(val)
if isempty(val)
    tf = false;
    return;
end
val = lower(strtrim(val));
tf = any(strcmp(val, {'1', 'true', 'yes', 'y', 'on'}));
end

function v = min_or_default(vals, default_val)
if isempty(vals)
    v = default_val;
else
    v = min(vals);
end
end

function v = max_or_default(vals, default_val)
if isempty(vals)
    v = default_val;
else
    v = max(vals);
end
end
