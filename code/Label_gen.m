%% ===================== Phase 2: Label generation for P-EPT =====================
% Goal:
%   - Sweep multiple Laplacian kernel radii (radius_list)  [differential family only]
%   - For each radius, reconstruct conductivity using UCL MR-EPT v2.2
%   - Compute global MAE/RMSE vs GT (demo-style figures, with axes & units)
%   - Build voxel-wise optimal radius_map (argmin MAE per voxel)
%   - Produce radius map figures (mid-slice + histogram)
%   - Extract patch-level dataset: phase_batch + radius_batch (for CNN)
%
% Directory layout:
%   project_root/
%     data/
%       3T_Brain_phantom/
%         input/    % 原始数据（建议你手动整理到这里）
%           Transceive_phase_noiseless.nii.gz
%           T1w_noiseless.nii.gz
%           Tissue_Segmentation.nii.gz  (optional)
%           Conductivity_GroundTruth.nii.gz
%         phase2_results/
%           volumes/
%             phase_corrected.nii.gz
%             mask_brain.nii.gz
%             radius_optimal_map.nii.gz
%             mae_map.nii.gz
%             sigma_gt.nii.gz
%             sigma_r<min>.nii.gz
%             sigma_r<max>.nii.gz
%             sigma_r<best>.nii.gz
%           figures/
%             mae_vs_radius_runX.png
%             demo_GT_EPT_Error_runX.png
%             demo_error_hist_runX.png
%             radius_map_slice_runX.png
%             radius_hist_runX.png
%           metrics/
%             global_metrics_runX.mat
%             global_metrics_runX.csv
%           patches/
%             phase_radius_patches_runX.mat
%
% Author: jared
% Date:   2025/12/02 (mod: differential-only sweep, auto-mask, noise option, 64^3 patches)

clear; clc; close all;

%% ===================== Dependencies and paths =====================
addpath(genpath('functions'));
addpath(genpath('toolboxes'));

%% ===================== Dataset & path configuration =====================
projectRoot = pwd;
datasetName = '3T_Brain_phantom';

datasetDir = fullfile(projectRoot, 'data', datasetName);
inputDir   = fullfile(datasetDir, 'input');  % 推荐原始 NIfTI 放这里

% 兼容旧结构：如果 input/ 不存在，就直接用 datasetDir 做输入目录
if exist(inputDir, 'dir')
    baseDir = inputDir;
else
    baseDir = datasetDir;
end

% Phase 2 输出根目录
phase2Root = fullfile(datasetDir, 'phase2_results');
volDir     = fullfile(phase2Root, 'volumes');
figDir     = fullfile(phase2Root, 'figures');
metricDir  = fullfile(phase2Root, 'metrics');
patchDir   = fullfile(phase2Root, 'patches');

% 创建输出目录
if ~exist(phase2Root, 'dir'), mkdir(phase2Root); end
if ~exist(volDir, 'dir'),     mkdir(volDir);     end
if ~exist(figDir, 'dir'),     mkdir(figDir);     end
if ~exist(metricDir, 'dir'),  mkdir(metricDir);  end
if ~exist(patchDir, 'dir'),   mkdir(patchDir);   end

%% ===================== Load data =====================
% 1) Transceive phase (phi0): measured or simulated B1+ transceive phase
phi0_nii = nii_tool('load', fullfile(baseDir, 'Transceive_phase_noiseless.nii.gz'));
phi0 = phi0_nii.img;                      % transceive phase ϕ0
b1pulse_phase_full = phi0 / 2;            % approximate transmit (B1+) phase ≈ transceive/2

% 2) Ground truth conductivity (needed for auto segmentation if no seg is present)
gt_nii        = nii_tool('load', fullfile(baseDir, 'Conductivity_GroundTruth.nii.gz'));
sigma_gt_full = gt_nii.img;

% 3) Tissue segmentation & mask
segPath = fullfile(baseDir, 'Tissue_Segmentation.nii.gz');
if exist(segPath, 'file')
    seg_nii  = nii_tool('load', segPath);
    seg_full = seg_nii.img;
else
    fprintf('[Auto-seg] Tissue_Segmentation.nii.gz not found. Generating from sigma GT...\n');
    sigma = sigma_gt_full;

    % ----- define conductivity ranges (from adept paper fig2c)
    % ) -----
    wm_mask  = sigma >= 0.27 & sigma <= 0.41;
    gm_mask  = sigma >= 0.49 & sigma <= 0.69;
    csf_mask = sigma >= 1.95 & sigma <= 2.33;

    seg = zeros(size(sigma), 'uint8');
    seg(wm_mask)  = 1;   % WM
    seg(gm_mask)  = 2;   % GM
    seg(csf_mask) = 3;   % CSF

    nii_seg      = gt_nii;
    nii_seg.img  = seg;
    autoSegPath  = fullfile(baseDir, 'Tissue_Segmentation_auto.nii.gz');
    nii_tool('save', nii_seg, autoSegPath);

    fprintf('  Saved auto tissue segmentation:\n    %s\n', autoSegPath);

    seg_nii  = nii_seg;
    seg_full = seg;
end

mask_full = seg_full > 0;             % non-zero voxels as brain mask

% 4) Magnitude image (use T1w_noiseless as default)
mag_nii  = nii_tool('load', fullfile(baseDir, 'T1w_noiseless.nii.gz'));
mag_full = mag_nii.img;

%% ============ (Optional) add phase noise ==========================
addNoise  = false;      % set true to add noise to B1+ phase
noiseStd  = 0.05;       % [radians] std of zero-mean Gaussian noise

if addNoise
    fprintf('[Noise] Adding Gaussian phase noise: std = %.4f rad\n', noiseStd);
    noise = noiseStd * randn(size(b1pulse_phase_full));
    noise(~mask_full) = 0;
    b1pulse_phase_full = b1pulse_phase_full + noise;
end

%% ============ Sub-volume (for 64^3 patch 不再使用，建议整脑) ============
doSubVolume = false;   % ⚠️ 64^3 patch 推荐用整脑体积

if doSubVolume
    sz = size(b1pulse_phase_full);
    cx = round(sz(1)/2);
    cy = round(sz(2)/2);
    cz = round(sz(3)/2);
    rad = 20;  % 确保支持 64^3 patch，给大一点 margin

    xRange = (cx-rad+1):(cx+rad);
    yRange = (cy-rad+1):(cy+rad);
    zRange = (cz-rad+1):(cz+rad);

    b1pulse_phase = b1pulse_phase_full(xRange, yRange, zRange);
    segmentation  = seg_full(xRange, yRange, zRange);
    mask          = mask_full(xRange, yRange, zRange);
    sigma_gt      = sigma_gt_full(xRange, yRange, zRange);
    magnitude     = mag_full(xRange, yRange, zRange);

    % also crop NIfTI headers for saving sub-volume maps
    phi0_nii.img = phi0_nii.img(xRange, yRange, zRange);
    gt_nii.img   = sigma_gt;
    seg_nii.img  = segmentation;
else
    b1pulse_phase = b1pulse_phase_full;
    segmentation  = seg_full;
    mask          = mask_full;
    sigma_gt      = sigma_gt_full;
    magnitude     = mag_full;
end

[nx, ny, nz] = size(b1pulse_phase);

%% ===================== Basic scan / model parameters =====================
parameters.B0        = 3;        % [T] 3T scanner
parameters.VoxelSize = [1 1 1];  % [mm] isotropic voxels

% differential kernel will be swept, integral kernel fixed
parameters.kDiffSize     = [3 3 3];   % baseline template (will be overwritten per radius)
parameters.kIntegralSize = [7 7 7];   % fixed integral kernel (no sweeping)

%% ===================== Radius list & run ID (for metrics/patches naming) =====================
radius_list = 1:15;        % radii to sweep (for differential family)
nR          = numel(radius_list);

% run_id = 1, 2, 3, ...  (按已有 metrics 文件自动 +1)
existingMetrics = dir(fullfile(metricDir, 'global_metrics_run*.mat'));
runIdx = 1;
if ~isempty(existingMetrics)
    nums = [];
    for k = 1:numel(existingMetrics)
        tok = regexp(existingMetrics(k).name, 'global_metrics_run(\d+)\.mat', 'tokens');
        if ~isempty(tok)
            nums(end+1) = str2double(tok{1}{1}); %#ok<AGROW>
        end
    end
    if ~isempty(nums)
        runIdx = max(nums) + 1;
    end
end
run_id  = runIdx;
run_str = sprintf('%d', run_id);

fprintf('\n[Phase 2] Run ID: %s\n', run_str);
fprintf('  volumes: %s\n', volDir);
fprintf('  figures: %s\n', figDir);
fprintf('  metrics: %s\n', metricDir);
fprintf('  patches: %s\n', patchDir);

%% ===================== Radius sweep: reconstruct cond for each radius =====================
cond_multi      = cell(nR,1);
mae_per_radius  = zeros(nR,1);
rmse_per_radius = zeros(nR,1);

% Valid mask: brain & finite GT
validMask = mask & isfinite(sigma_gt);

fprintf('\n================= Radius sweep (global MAE vs radius, differential only) =================\n');

for ir = 1:nR
    r     = radius_list(ir);
    kSize = 2*r + 1;

    params_r = parameters;
    % sweep only differential kernel family
    params_r.kDiffSize = [kSize kSize kSize];
    % kIntegralSize remains fixed

    fprintf('  >> radius = %d (kDiffSize = [%d %d %d], kIntegralSize = [%d %d %d] FIXED)...\n', ...
        r, params_r.kDiffSize(1), params_r.kDiffSize(2), params_r.kDiffSize(3), ...
        params_r.kIntegralSize(1), params_r.kIntegralSize(2), params_r.kIntegralSize(3));

    cond_r = conductivityMapping( ...
        b1pulse_phase, ...
        mask, ...
        params_r, ...
        'magnitude', magnitude, ...
        'segmentation', segmentation);

    cond_multi{ir} = cond_r;

    % Global error vs GT
    if ~isequal(size(cond_r), size(sigma_gt))
        error('Size mismatch: cond_r vs sigma_gt in radius sweep.');
    end

    diff_r = cond_r - sigma_gt;
    diff_r(~validMask) = NaN;
    abs_r  = abs(diff_r);

    mae_per_radius(ir)  = mean(abs_r(validMask), 'omitnan');
    rmse_per_radius(ir) = sqrt(mean(diff_r(validMask).^2, 'omitnan'));
end

fprintf('\n[Global MAE / RMSE vs radius]\n');
for ir = 1:nR
    fprintf('  radius=%d: MAE=%.4f, RMSE=%.4f\n', ...
        radius_list(ir), mae_per_radius(ir), rmse_per_radius(ir));
end

%% ===================== Demo-style figures (using global-best radius) =====================
[~, best_global_idx] = min(mae_per_radius);
best_radius = radius_list(best_global_idx);
cond_best   = cond_multi{best_global_idx};

fprintf('\n[Demo] Using global-best radius = %d for demo-style figures.\n', best_radius);

% Error volumes for best radius
err_best      = cond_best - sigma_gt;
abs_best      = abs(err_best);
err_vals_best = err_best(validMask);
abs_vals_best = abs_best(validMask);

mae_best  = mean(abs_vals_best, 'omitnan');
rmse_best = sqrt(mean(err_vals_best.^2, 'omitnan'));

fprintf('[Demo] Best radius MAE=%.4f, RMSE=%.4f\n', mae_best, rmse_best);

% 1) GT / EPT / Error mid-slice
midZ = round(nz / 2);
gt_slice   = squeeze(sigma_gt(:,:,midZ))';
ept_slice  = squeeze(cond_best(:,:,midZ))';
err_slice  = squeeze(err_best(:,:,midZ))';

figure;
tiledlayout(1,3);

% GT
nexttile;
imagesc(gt_slice);
axis image;
xlabel('x [mm]');
ylabel('y [mm]');
cb1 = colorbar;
ylabel(cb1, '\sigma_{GT} [S/m]');
title('GT conductivity');
set(gca,'YDir','normal');

% EPT
nexttile;
imagesc(ept_slice);
axis image;
xlabel('x [mm]');
ylabel('y [mm]');
cb2 = colorbar;
ylabel(cb2, '\sigma_{EPT} [S/m]');
title(sprintf('EPT conductivity (radius = %d)', best_radius));
set(gca,'YDir','normal');

% Error
nexttile;
imagesc(err_slice);
axis image;
xlabel('x [mm]');
ylabel('y [mm]');
cb3 = colorbar;
ylabel(cb3, '\sigma_{EPT} - \sigma_{GT} [S/m]');
title('Error (EPT - GT)');
set(gca,'YDir','normal');

demoFig_fname = fullfile(figDir, sprintf('demo_GT_EPT_Error_run%s.png', run_str));
saveas(gcf, demoFig_fname);

% 2) Error histogram
figure;
histogram(err_vals_best, 50);
xlabel('Error \sigma_{EPT} - \sigma_{GT} [S/m]');
ylabel('Voxel count [#]');
title(sprintf('Error histogram (radius = %d)', best_radius));
grid on;

demoHist_fname = fullfile(figDir, sprintf('demo_error_hist_run%s.png', run_str));
saveas(gcf, demoHist_fname);

% 3) Global MAE vs radius 曲线
figure;
plot(radius_list, mae_per_radius, '-o');
xlabel('Kernel radius r [voxels] (differential family)');
ylabel('Global MAE [S/m]');
title(sprintf('Global MAE vs radius (run %s)', run_str));
grid on;

maeRadiusFig = fullfile(figDir, sprintf('mae_vs_radius_run%s.png', run_str));
saveas(gcf, maeRadiusFig);

%% ===================== Save global metrics (metrics/) =====================
metrics_fname_mat  = fullfile(metricDir, sprintf('global_metrics_run%s.mat', run_str));
metrics_fname_csv  = fullfile(metricDir, sprintf('global_metrics_run%s.csv', run_str));

radiusSweep = struct();
radiusSweep.radius_list     = radius_list;
radiusSweep.mae_per_radius  = mae_per_radius;
radiusSweep.rmse_per_radius = rmse_per_radius;
radiusSweep.best_radius     = best_radius;
radiusSweep.best_mae        = mae_best;
radiusSweep.best_rmse       = rmse_best;
radiusSweep.doSubVolume     = doSubVolume;

if doSubVolume
    radiusSweep.xRange = xRange;
    radiusSweep.yRange = yRange;
    radiusSweep.zRange = zRange;
end

save(metrics_fname_mat, 'radiusSweep');

T = table(radius_list(:), mae_per_radius(:), rmse_per_radius(:), ...
    'VariableNames', {'radius', 'MAE', 'RMSE'});
writetable(T, metrics_fname_csv);

%% ===================== Phase 2 core: voxel-wise optimal radius_map & mae_map =====================
fprintf('\n================= Phase 2: voxel-wise optimal radius_map =================\n');

% 用 inf 而不是 NaN，方便直接取 min，同时生成 mae_map
err_stack = inf(nx, ny, nz, nR, 'single');  % |σ_r - σ_GT|

for ir = 1:nR
    c_r = cond_multi{ir};
    diff_r = c_r - sigma_gt;
    diff_r(~validMask) = inf;
    err_stack(:,:,:,ir) = abs(single(diff_r));
end

[min_err, best_idx_voxel] = min(err_stack, [], 4);  % argmin over radius dimension

radius_map = zeros(nx, ny, nz, 'single');
radius_map(validMask) = single(radius_list(best_idx_voxel(validMask)));

mae_map = zeros(nx, ny, nz, 'single');
mae_map(validMask) = min_err(validMask);

% ============ Save NIfTI volumes (volumes/) ============

% 1) optimal radius map
radius_nii     = phi0_nii;
radius_nii.img = radius_map;
radius_fname   = fullfile(volDir, 'radius_optimal_map.nii.gz');
nii_tool('save', radius_nii, radius_fname);
fprintf('Saved optimal radius map NIfTI:\n  %s\n', radius_fname);

% 2) mae map
mae_nii     = phi0_nii;
mae_nii.img = mae_map;
mae_fname   = fullfile(volDir, 'mae_map.nii.gz');
nii_tool('save', mae_nii, mae_fname);
fprintf('Saved MAE map NIfTI:\n  %s\n', mae_fname);

% 3) phase_corrected (for CNN input)
phase_nii      = phi0_nii;
phase_nii.img  = b1pulse_phase;
phase_fname    = fullfile(volDir, 'phase_corrected.nii.gz');
nii_tool('save', phase_nii, phase_fname);
fprintf('Saved phase_corrected NIfTI:\n  %s\n', phase_fname);

% 4) mask_brain
mask_nii      = seg_nii;
mask_nii.img  = uint8(mask);
mask_fname    = fullfile(volDir, 'mask_brain.nii.gz');
nii_tool('save', mask_nii, mask_fname);
fprintf('Saved mask_brain NIfTI:\n  %s\n', mask_fname);

% 5) sigma_gt
sigma_gt_nii     = gt_nii;
sigma_gt_nii.img = sigma_gt;
sigma_gt_fname   = fullfile(volDir, 'sigma_gt.nii.gz');
nii_tool('save', sigma_gt_nii, sigma_gt_fname);
fprintf('Saved sigma_gt NIfTI:\n  %s\n', sigma_gt_fname);

% 6) sigma_r(min) / sigma_r(max) / sigma_r(best)
min_radius = radius_list(1);
max_radius = radius_list(end);

cond_min = cond_multi{1};
cond_max = cond_multi{end};

sigma_nii = gt_nii;   % reuse GT header

sigma_nii.img = cond_min;
sigma_min_fname = fullfile(volDir, sprintf('sigma_r%d_min.nii.gz', min_radius));
nii_tool('save', sigma_nii, sigma_min_fname);

sigma_nii.img = cond_max;
sigma_max_fname = fullfile(volDir, sprintf('sigma_r%d_max.nii.gz', max_radius));
nii_tool('save', sigma_nii, sigma_max_fname);

sigma_nii.img = cond_best;
sigma_best_fname = fullfile(volDir, sprintf('sigma_r%d_best.nii.gz', best_radius));
nii_tool('save', sigma_nii, sigma_best_fname);

fprintf('Saved sigma_r(min/max/best) NIfTIs:\n');
fprintf('  %s\n', sigma_min_fname);
fprintf('  %s\n', sigma_max_fname);
fprintf('  %s\n', sigma_best_fname);

%% ===================== Tissue-wise radius statistics =====================
seg_in_mask   = segmentation(validMask);
rvals_in_mask = radius_map(validMask);

t_ids = unique(seg_in_mask);
t_ids(t_ids == 0) = [];

% Dummy tissue names; adjust according to phantom labels
tissueNames = containers.Map('KeyType','double','ValueType','char');
tissueNames(1) = 'WM';
tissueNames(2) = 'GM';
tissueNames(3) = 'CSF';
tissueNames(4) = 'Tissue4';

tissueRadiusStats = struct();

fprintf('\n[Tissue-wise optimal radius statistics]\n');
for k = 1:numel(t_ids)
    tid   = t_ids(k);
    tmask = validMask & (segmentation == tid);

    if ~any(tmask(:))
        continue;
    end

    r_t = radius_map(tmask);

    if isKey(tissueNames, tid)
        tname = tissueNames(tid);
    else
        tname = sprintf('Tissue_%d', tid);
    end

    tissueRadiusStats(k).id    = tid;
    tissueRadiusStats(k).name  = tname;
    tissueRadiusStats(k).meanR = mean(r_t(:), 'omitnan');
    tissueRadiusStats(k).stdR  = std(r_t(:), 0, 'omitnan');
    tissueRadiusStats(k).minR  = min(r_t(:));
    tissueRadiusStats(k).maxR  = max(r_t(:));
    tissueRadiusStats(k).Nvox  = numel(r_t);

    fprintf('  [%s] id=%d: meanR = %.2f, stdR = %.2f, min=%g, max=%g, N=%d\n', ...
        tname, tid, tissueRadiusStats(k).meanR, tissueRadiusStats(k).stdR, ...
        tissueRadiusStats(k).minR, tissueRadiusStats(k).maxR, tissueRadiusStats(k).Nvox);
end

%% ===================== Phase 2 figures: radius map slice + histogram =====================
midZ = round(nz / 2);
r_slice = squeeze(radius_map(:,:,midZ))';

figure;
imagesc(r_slice);
axis image;
xlabel('x [mm]');
ylabel('y [mm]');
cbR = colorbar;
ylabel(cbR, 'Optimal radius r [voxels]');
title(sprintf('Optimal kernel radius map (mid-slice, run %s)', run_str));
set(gca,'YDir','normal');

radiusSliceFig = fullfile(figDir, sprintf('radius_map_slice_run%s.png', run_str));
saveas(gcf, radiusSliceFig);

figure;
histogram(rvals_in_mask, numel(radius_list));
xlabel('Optimal radius r [voxels]');
ylabel('Voxel count [#]');
title(sprintf('Histogram of optimal radius (run %s)', run_str));
grid on;

radiusHistFig = fullfile(figDir, sprintf('radius_hist_run%s.png', run_str));
saveas(gcf, radiusHistFig);

%% ===================== Build patch-level dataset: 64^3 phase_batch + radius_batch =====================
fprintf('\n================= Phase 2: Building patch-level dataset (64^3) =================\n');

patchSize   = [64 64 64];    % ★ 训练 CNN 的 64^3 patch
patchStride = [8 8 8];       % 可以根据显存/样本量调节

halfSize = floor(patchSize / 2); % [hx hy hz]

if any([nx ny nz] < patchSize)
    error('Volume size [%d %d %d] is smaller than patch size [%d %d %d].', ...
        nx, ny, nz, patchSize(1), patchSize(2), patchSize(3));
end

xRangePatch = (1+halfSize(1)) : patchStride(1) : (nx-halfSize(1));
yRangePatch = (1+halfSize(2)) : patchStride(2) : (ny-halfSize(2));
zRangePatch = (1+halfSize(3)) : patchStride(3) : (nz-halfSize(3));

centers = [];
for ix = xRangePatch
    for iy = yRangePatch
        for iz = zRangePatch
            if ~validMask(ix,iy,iz)
                continue;
            end
            centers(end+1,:) = [ix, iy, iz]; %#ok<AGROW>
        end
    end
end

Npatch = size(centers,1);
fprintf('  Number of valid patch centers = %d\n', Npatch);

phase_batch  = zeros(Npatch, 1, patchSize(1), patchSize(2), patchSize(3), 'single');
radius_batch = zeros(Npatch, 1, 'int32');

for pi = 1:Npatch
    cx = centers(pi,1);
    cy = centers(pi,2);
    cz = centers(pi,3);

    xIdx = (cx-halfSize(1)) : (cx+halfSize(1));
    yIdx = (cy-halfSize(2)) : (cy+halfSize(2));
    zIdx = (cz-halfSize(3)) : (cz+halfSize(3));

    patch_phase = b1pulse_phase(xIdx, yIdx, zIdx);

    phase_batch(pi,1,:,:,:) = single(patch_phase);
    radius_batch(pi)        = int32(radius_map(cx,cy,cz));
end

patchDataset_fname = fullfile(patchDir, sprintf('phase_radius_patches_run%s.mat', run_str));

patchMeta = struct();
patchMeta.patchSize    = patchSize;
patchMeta.patchStride  = patchStride;
patchMeta.radius_list  = radius_list;
patchMeta.centers      = centers;
patchMeta.validMaskN   = nnz(validMask);
patchMeta.doSubVolume  = doSubVolume;
if doSubVolume
    patchMeta.xRange = xRange;
    patchMeta.yRange = yRange;
    patchMeta.zRange = zRange;
end

save(patchDataset_fname, ...
     'phase_batch', 'radius_batch', 'patchMeta', ...
     'tissueRadiusStats', 'radius_list', 'mae_per_radius', 'rmse_per_radius', '-v7.3');

fprintf('\nSaved Phase 2 outputs (run %s):\n', run_str);
fprintf('  METRICS: %s\n', metrics_fname_mat);
fprintf('           %s\n', metrics_fname_csv);
fprintf('  VOLUMES:\n');
fprintf('    %s\n', phase_fname);
fprintf('    %s\n', mask_fname);
fprintf('    %s\n', radius_fname);
fprintf('    %s\n', mae_fname);
fprintf('    %s\n', sigma_gt_fname);
fprintf('    %s\n', sigma_min_fname);
fprintf('    %s\n', sigma_max_fname);
fprintf('    %s\n', sigma_best_fname);
fprintf('  PATCHES: %s\n', patchDataset_fname);
fprintf('  FIGS:\n');
fprintf('    %s\n', demoFig_fname);
fprintf('    %s\n', demoHist_fname);
fprintf('    %s\n', maeRadiusFig);
fprintf('    %s\n', radiusSliceFig);
fprintf('    %s\n', radiusHistFig);
fprintf('\n================= Phase 2 finished. =================\n\n');
