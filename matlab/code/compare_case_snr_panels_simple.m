%% Compare conductivity panel from direct input paths (1x3)
% Fill the paths below and run this script directly in MATLAB.

clear; clc;
warning('off', 'backtrace');

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab', 'functions'));
addpath(genpath(fullfile(repo_root, 'matlab', 'toolboxes')));

% -------- User config (edit these paths directly) --------
GT_MAT_PATH = '/Users/apple/Documents/deeplc/ADEPT_Dataset/Healthy/M12.mat';
OPTIMAL_MAT_PATH = '/Users/apple/Documents/mresult/optimal/sigma_reconstructed.mat';
PREDICTED_NII_PATH = '/Users/apple/Documents/mresult/recon/M12_SNR150_sigma_recon.nii.gz';
OUT_PNG = '/Users/apple/Documents/mresult/optimal/compare_panels_simple/M12_SNR150_conductivity_1x3.png';

PANEL_TITLE = 'M12 SNR010 | Conductivity (1x3)';
AXIAL_INDEX_MODE = 'center'; % center | custom
AXIAL_INDEX = 0;             % used when AXIAL_INDEX_MODE='custom'

CLIM_MODE = 'percentile';    % percentile | minmax
PCT_LOW = 1;
PCT_HIGH = 99;

if PCT_LOW < 0; PCT_LOW = 0; end
if PCT_HIGH > 100; PCT_HIGH = 100; end
if PCT_LOW >= PCT_HIGH
    PCT_LOW = 1;
    PCT_HIGH = 99;
end

if ~isfile(GT_MAT_PATH)
    error('GT_MAT_PATH not found: %s', GT_MAT_PATH);
end
if ~isfile(OPTIMAL_MAT_PATH)
    error('OPTIMAL_MAT_PATH not found: %s', OPTIMAL_MAT_PATH);
end
if ~isfile(PREDICTED_NII_PATH)
    error('PREDICTED_NII_PATH not found: %s', PREDICTED_NII_PATH);
end

out_dir = fileparts(OUT_PNG);
if ~isempty(out_dir) && ~isfolder(out_dir)
    mkdir(out_dir);
end

fprintf('GT_MAT_PATH: %s\n', GT_MAT_PATH);
fprintf('OPTIMAL_MAT_PATH: %s\n', OPTIMAL_MAT_PATH);
fprintf('PREDICTED_NII_PATH: %s\n', PREDICTED_NII_PATH);
fprintf('OUT_PNG: %s\n', OUT_PNG);

[ok, reason, payload] = load_and_prepare(GT_MAT_PATH, OPTIMAL_MAT_PATH, PREDICTED_NII_PATH, AXIAL_INDEX_MODE, AXIAL_INDEX);
if ~ok
    error('Failed to load data (%s)\ngt_path: %s\noptimal_path: %s\npredicted_path: %s', ...
        reason, GT_MAT_PATH, OPTIMAL_MAT_PATH, PREDICTED_NII_PATH);
end

clim_cond = compute_clim(payload.cond_vals, CLIM_MODE, PCT_LOW, PCT_HIGH);
render_conductivity_panel_1x3(PANEL_TITLE, payload, clim_cond, OUT_PNG);

fprintf('Saved: %s\n', OUT_PNG);

%% -------- Local helpers --------
function [ok, reason, out] = load_and_prepare(gt_path, optimal_path, predicted_path, axial_mode, axial_index)
out = struct();
ok = false;
reason = '';

try
    gt_data = load(gt_path);
catch
    reason = 'load_gt_failed';
    return;
end
if ~isfield(gt_data, 'Conductivity_GT')
    reason = 'missing_field_Conductivity_GT';
    return;
end

gt = single(gt_data.Conductivity_GT);
if isfield(gt_data, 'Segmentation')
    mask = gt_data.Segmentation > 0;
else
    mask = isfinite(gt);
end

try
    optimal_data = load(optimal_path);
catch
    reason = 'load_optimal_failed';
    return;
end
if ~isfield(optimal_data, 'cond_optimal')
    reason = 'missing_field_cond_optimal';
    return;
end
opt = single(optimal_data.cond_optimal);

try
    pred = single(load_nii_any(predicted_path));
catch
    reason = 'load_predicted_failed';
    return;
end

if ~isequal(size(gt), size(opt), size(pred))
    reason = 'size_mismatch';
    return;
end
if ~any(mask(:))
    reason = 'empty_mask';
    return;
end

nz = size(gt, 3);
if strcmpi(axial_mode, 'custom') && axial_index >= 1
    z = min(max(1, axial_index), nz);
else
    z = max(1, min(nz, round(nz / 2)));
end

gt_slice = squeeze(gt(:,:,z));
opt_slice = squeeze(opt(:,:,z));
pred_slice = squeeze(pred(:,:,z));
mask_slice = squeeze(mask(:,:,z));

valid_opt = mask & isfinite(gt) & isfinite(opt);
valid_pred = mask & isfinite(gt) & isfinite(pred);
if ~any(valid_opt(:)) || ~any(valid_pred(:))
    reason = 'no_valid_voxels';
    return;
end

metrics_opt = calc_metrics(opt, gt, mask);
metrics_pred = calc_metrics(pred, gt, mask);

cond_vals = [gt_slice(mask_slice & isfinite(gt_slice)); ...
    opt_slice(mask_slice & isfinite(opt_slice)); ...
    pred_slice(mask_slice & isfinite(pred_slice))];

out.gt_slice = gt_slice;
out.opt_slice = opt_slice;
out.pred_slice = pred_slice;
out.metrics_opt = metrics_opt;
out.metrics_pred = metrics_pred;
out.cond_vals = cond_vals;

ok = true;
reason = 'ok';
end

function m = calc_metrics(pred, gt, mask)
valid = mask & isfinite(pred) & isfinite(gt);
err = pred - gt;
abs_err = abs(err(valid));

m = struct('MAE', NaN, 'RMSE', NaN, 'SSIM', NaN);
if isempty(abs_err)
    return;
end

m.MAE = mean(abs_err);
m.RMSE = sqrt(mean((err(valid)).^2));

data_range = max(gt(valid)) - min(gt(valid));
if ~isfinite(data_range) || data_range <= 0
    data_range = 1;
end

pred_ssim = pred;
gt_ssim = gt;
pred_ssim(~valid) = 0;
gt_ssim(~valid) = 0;

try
    m.SSIM = ssim(pred_ssim, gt_ssim, 'DynamicRange', data_range);
catch
    m.SSIM = NaN;
end
end

function climv = compute_clim(vals, mode, p_lo, p_hi)
vals = vals(isfinite(vals));
if isempty(vals)
    climv = [0, 1];
    return;
end

if strcmpi(mode, 'minmax')
    lo = min(vals);
    hi = max(vals);
else
    lo = prctile(vals, p_lo);
    hi = prctile(vals, p_hi);
end

if ~isfinite(lo) || ~isfinite(hi)
    climv = [0, 1];
    return;
end
if lo == hi
    hi = lo + 1e-6;
end
climv = [lo, hi];
end

function render_conductivity_panel_1x3(panel_title, payload, climv, out_png)
h = figure('Visible', 'off', 'Position', [120, 120, 1380, 420]);
tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile; draw_tile(ax1, payload.gt_slice, climv); title(ax1, 'gt');
ax2 = nexttile; draw_tile(ax2, payload.opt_slice, climv); title(ax2, 'optimal');
ax3 = nexttile; draw_tile(ax3, payload.pred_slice, climv); title(ax3, 'predicted');

text(ax2, 0.02, 0.98, metric_text(payload.metrics_opt), 'Units', 'normalized', ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
    'Color', 'w', 'FontSize', 10, 'BackgroundColor', [0 0 0]);
text(ax3, 0.02, 0.98, metric_text(payload.metrics_pred), 'Units', 'normalized', ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
    'Color', 'w', 'FontSize', 10, 'BackgroundColor', [0 0 0]);

sgtitle(tl, panel_title);
exportgraphics(h, out_png, 'Resolution', 220);
close(h);
end

function draw_tile(ax, img, climv)
imagesc(ax, img);
axis(ax, 'image');
axis(ax, 'off');
colormap(ax, 'jet');
clim(ax, climv);
end

function txt = metric_text(m)
txt = sprintf('MAE %.4f | RMSE %.4f | SSIM %.4f', m.MAE, m.RMSE, m.SSIM);
end

function img = load_nii_any(path_in)
if exist('niftiread', 'file') == 2
    img = niftiread(path_in);
elseif exist('nii_tool', 'file') == 2
    n = nii_tool('load', path_in);
    img = n.img;
else
    error('No NIfTI reader available. Need niftiread or nii_tool.');
end
img = single(img);
end
