%% Compare conductivity panels per case (8 SNR rows)
% Figure 1: Conductivity, 8x3 -> gt | optimal | predicted
% Figure 2: MAE map,      8x2 -> optimal | predicted
% Figure 3: RMSE map,     8x2 -> optimal | predicted
% Each figure uses one global clim across all SNR rows.

clear; clc;
warning('off', 'backtrace');

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab', 'functions'));
addpath(genpath(fullfile(repo_root, 'matlab', 'toolboxes')));

% -------- Config (override by environment variables) --------
RECON_ROOT = getenv_default('RECON_ROOT', '/home/zcemexx/Scratch/outputs/phase5'); %recon.m, constructed from predicted map
PHASE5_ROOT = getenv_default('PHASE5_ROOT', '/home/zcemexx/Scratch/outputs/phase5');%exp.m, constructed from optimal map
GT_ROOT = getenv_default('GT_ROOT', '~/Scratch/data/ADEPT_raw');
OUT_DIR = getenv_default('COMPARE_OUT_ROOT', fullfile(PHASE5_ROOT, 'compare_panels'));

RUN_MODE = lower(getenv_default('COMPARE_RUN_MODE', 'batch')); % batch | single
CASE_FILTER = getenv_default('COMPARE_CASE', ''); % e.g., M6
AXIAL_INDEX_MODE = lower(getenv_default('COMPARE_AXIAL_INDEX_MODE', 'center')); % center | custom
AXIAL_INDEX = parse_positive_int(getenv_default('COMPARE_AXIAL_INDEX', '0'), 0);

CLIM_MODE = lower(getenv_default('COMPARE_CLIM_MODE', 'percentile')); % percentile | minmax
PCT_LOW = str2double(getenv_default('COMPARE_PCT_LOW', '1'));
PCT_HIGH = str2double(getenv_default('COMPARE_PCT_HIGH', '99'));
if ~isfinite(PCT_LOW); PCT_LOW = 1; end
if ~isfinite(PCT_HIGH); PCT_HIGH = 99; end
if PCT_LOW < 0; PCT_LOW = 0; end
if PCT_HIGH > 100; PCT_HIGH = 100; end
if PCT_LOW >= PCT_HIGH
    PCT_LOW = 1;
    PCT_HIGH = 99;
end

SNR_LIST = parse_int_list(getenv_default('COMPARE_SNR_LIST', '10,20,30,40,50,75,100,150'));
if isempty(SNR_LIST)
    SNR_LIST = [10 20 30 40 50 75 100 150];
end
SNR_LIST = unique(SNR_LIST, 'sorted');

if ~isfolder(RECON_ROOT); error('RECON_ROOT not found: %s', RECON_ROOT); end
if ~isfolder(PHASE5_ROOT); error('PHASE5_ROOT not found: %s', PHASE5_ROOT); end
if ~isfolder(GT_ROOT); error('GT_ROOT not found: %s', GT_ROOT); end
if ~isfolder(OUT_DIR); mkdir(OUT_DIR); end
fig_root = fullfile(OUT_DIR, 'figures');
if ~isfolder(fig_root); mkdir(fig_root); end

fprintf('RECON_ROOT: %s\n', RECON_ROOT);
fprintf('PHASE5_ROOT: %s\n', PHASE5_ROOT);
fprintf('GT_ROOT: %s\n', GT_ROOT);
fprintf('OUT_DIR: %s\n', OUT_DIR);

recon_records = discover_recon_records(RECON_ROOT);
if isempty(recon_records)
    error('No recon output found under RECON_ROOT with pattern <Case>_<SNR>_sigma_recon.nii(.gz).');
end

case_list = unique({recon_records.case_name});
case_list = sort_case_names(case_list);
if strcmp(RUN_MODE, 'single')
    if isempty(CASE_FILTER)
        error('COMPARE_CASE is required when COMPARE_RUN_MODE=single');
    end
    case_list = case_list(strcmp(case_list, CASE_FILTER));
end
if isempty(case_list)
    error('No cases selected after filtering.');
end

metrics_header = {'Case','SNR','MAE_optimal','RMSE_optimal','SSIM_optimal', ...
    'MAE_predicted','RMSE_predicted','SSIM_predicted', ...
    'gt_path','optimal_path','predicted_path','status'};
metrics_rows = {};

missing_header = {'Case','SNR','reason','gt_path','optimal_path','predicted_path'};
missing_rows = {};

for ic = 1:numel(case_list)
    case_name = case_list{ic};
    fprintf('\n[CASE] %s\n', case_name);

    case_fig_dir = fullfile(fig_root, case_name);
    if ~isfolder(case_fig_dir)
        mkdir(case_fig_dir);
    end

    entries = repmat(init_entry(), [numel(SNR_LIST), 1]);
    cond_pool = [];
    mae_pool = [];
    rmse_pool = [];

    for is = 1:numel(SNR_LIST)
        snr_val = SNR_LIST(is);
        snr_tag = sprintf('SNR%03d', snr_val);

        gt_path = fullfile(GT_ROOT, sprintf('%s.mat', case_name));
        optimal_path = fullfile(PHASE5_ROOT, case_name, snr_tag, 'sigma_reconstructed.mat');
        predicted_path = find_recon_path(recon_records, case_name, snr_tag);

        entries(is).case_name = case_name;
        entries(is).snr_val = snr_val;
        entries(is).snr_tag = snr_tag;
        entries(is).gt_path = gt_path;
        entries(is).optimal_path = optimal_path;
        entries(is).predicted_path = predicted_path;

        [ok, reason, payload] = load_and_prepare(gt_path, optimal_path, predicted_path, AXIAL_INDEX_MODE, AXIAL_INDEX);
        if ~ok
            entries(is).status = reason;
            missing_rows(end+1, :) = {case_name, snr_tag, reason, gt_path, optimal_path, predicted_path}; %#ok<AGROW>
            metrics_rows(end+1, :) = {case_name, snr_tag, NaN, NaN, NaN, NaN, NaN, NaN, gt_path, optimal_path, predicted_path, reason}; %#ok<AGROW>
            continue;
        end

        entries(is).available = true;
        entries(is).status = 'ok';
        entries(is).gt_slice = payload.gt_slice;
        entries(is).opt_slice = payload.opt_slice;
        entries(is).pred_slice = payload.pred_slice;
        entries(is).mae_opt_slice = payload.mae_opt_slice;
        entries(is).mae_pred_slice = payload.mae_pred_slice;
        entries(is).rmse_opt_slice = payload.rmse_opt_slice;
        entries(is).rmse_pred_slice = payload.rmse_pred_slice;
        entries(is).metrics_opt = payload.metrics_opt;
        entries(is).metrics_pred = payload.metrics_pred;

        cond_pool = [cond_pool; payload.cond_vals(:)]; %#ok<AGROW>
        mae_pool = [mae_pool; payload.mae_vals(:)]; %#ok<AGROW>
        rmse_pool = [rmse_pool; payload.rmse_vals(:)]; %#ok<AGROW>

        metrics_rows(end+1, :) = {case_name, snr_tag, ...
            payload.metrics_opt.MAE, payload.metrics_opt.RMSE, payload.metrics_opt.SSIM, ...
            payload.metrics_pred.MAE, payload.metrics_pred.RMSE, payload.metrics_pred.SSIM, ...
            gt_path, optimal_path, predicted_path, 'ok'}; %#ok<AGROW>
    end

    clim_cond = compute_clim(cond_pool, CLIM_MODE, PCT_LOW, PCT_HIGH);
    clim_mae = compute_clim(mae_pool, CLIM_MODE, PCT_LOW, PCT_HIGH);
    clim_rmse = compute_clim(rmse_pool, CLIM_MODE, PCT_LOW, PCT_HIGH);

    out_cond = fullfile(case_fig_dir, sprintf('%s_conductivity_8x3.png', case_name));
    out_mae = fullfile(case_fig_dir, sprintf('%s_mae_8x2.png', case_name));
    out_rmse = fullfile(case_fig_dir, sprintf('%s_rmse_8x2.png', case_name));

    render_conductivity_panel(entries, clim_cond, out_cond);
    render_error_panel(entries, clim_mae, out_mae, 'mae');
    render_error_panel(entries, clim_rmse, out_rmse, 'rmse');

    fprintf('  saved: %s\n', out_cond);
    fprintf('  saved: %s\n', out_mae);
    fprintf('  saved: %s\n', out_rmse);
end

writecell([metrics_header; metrics_rows], fullfile(OUT_DIR, 'metrics_summary.csv'));
writecell([missing_header; missing_rows], fullfile(OUT_DIR, 'missing_pairs.csv'));

fprintf('\nDone. metrics_summary.csv and missing_pairs.csv saved in %s\n', OUT_DIR);

%% -------- Local helpers --------
function e = init_entry()
e = struct();
e.case_name = '';
e.snr_val = NaN;
e.snr_tag = '';
e.gt_path = '';
e.optimal_path = '';
e.predicted_path = '';
e.available = false;
e.status = 'missing';
e.gt_slice = [];
e.opt_slice = [];
e.pred_slice = [];
e.mae_opt_slice = [];
e.mae_pred_slice = [];
e.rmse_opt_slice = [];
e.rmse_pred_slice = [];
e.metrics_opt = struct('MAE', NaN, 'RMSE', NaN, 'SSIM', NaN);
e.metrics_pred = struct('MAE', NaN, 'RMSE', NaN, 'SSIM', NaN);
end

function [ok, reason, out] = load_and_prepare(gt_path, optimal_path, predicted_path, axial_mode, axial_index)
out = struct();
ok = false;
reason = '';

if ~isfile(gt_path)
    reason = 'missing_gt';
    return;
end
if ~isfile(optimal_path)
    reason = 'missing_optimal';
    return;
end
if isempty(predicted_path) || ~isfile(predicted_path)
    reason = 'missing_predicted';
    return;
end

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
    seg = gt_data.Segmentation;
    mask = seg > 0;
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
if strcmp(axial_mode, 'custom') && axial_index >= 1
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

mae_opt = abs(opt - gt);
mae_pred = abs(pred - gt);
rmse_opt = sqrt((opt - gt).^2);
rmse_pred = sqrt((pred - gt).^2);

mae_opt_slice = squeeze(mae_opt(:,:,z));
mae_pred_slice = squeeze(mae_pred(:,:,z));
rmse_opt_slice = squeeze(rmse_opt(:,:,z));
rmse_pred_slice = squeeze(rmse_pred(:,:,z));

cond_vals = [gt_slice(mask_slice & isfinite(gt_slice)); ...
    opt_slice(mask_slice & isfinite(opt_slice)); ...
    pred_slice(mask_slice & isfinite(pred_slice))];
mae_vals = [mae_opt_slice(mask_slice & isfinite(mae_opt_slice)); ...
    mae_pred_slice(mask_slice & isfinite(mae_pred_slice))];
rmse_vals = [rmse_opt_slice(mask_slice & isfinite(rmse_opt_slice)); ...
    rmse_pred_slice(mask_slice & isfinite(rmse_pred_slice))];

out.gt_slice = gt_slice;
out.opt_slice = opt_slice;
out.pred_slice = pred_slice;
out.mae_opt_slice = mae_opt_slice;
out.mae_pred_slice = mae_pred_slice;
out.rmse_opt_slice = rmse_opt_slice;
out.rmse_pred_slice = rmse_pred_slice;
out.metrics_opt = metrics_opt;
out.metrics_pred = metrics_pred;
out.cond_vals = cond_vals;
out.mae_vals = mae_vals;
out.rmse_vals = rmse_vals;

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

if strcmp(mode, 'minmax')
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

function render_conductivity_panel(entries, climv, out_png)
n = numel(entries);
h = figure('Visible', 'off', 'Position', [60, 40, 1550, 260*n]);
tl = tiledlayout(n, 3, 'TileSpacing', 'compact', 'Padding', 'loose');

for i = 1:n
    e = entries(i);

    ax1 = nexttile; draw_tile(ax1, e.available, e.gt_slice, climv);
    ax2 = nexttile; draw_tile(ax2, e.available, e.opt_slice, climv);
    ax3 = nexttile; draw_tile(ax3, e.available, e.pred_slice, climv);

    if i == 1
        title(ax1, 'gt');
        title(ax2, 'optimal');
        title(ax3, 'predicted');
    end

    add_snr_label(ax1, e.snr_tag);

    if e.available
        text(ax2, 0.02, 0.98, metric_text(e.metrics_opt), 'Units', 'normalized', ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
            'Color', 'w', 'FontSize', 8, 'BackgroundColor', [0 0 0]);
        text(ax3, 0.02, 0.98, metric_text(e.metrics_pred), 'Units', 'normalized', ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
            'Color', 'w', 'FontSize', 8, 'BackgroundColor', [0 0 0]);
    end
end

cb = colorbar(tl, 'eastoutside');
cb.Label.String = 'Conductivity';
cb.FontSize = 9;

sgtitle(tl, sprintf('%s | Conductivity (global clim)', entries(1).case_name));
exportgraphics(h, out_png, 'Resolution', 220);
close(h);
end

function render_error_panel(entries, climv, out_png, map_type)
n = numel(entries);
h = figure('Visible', 'off', 'Position', [80, 40, 1150, 260*n]);
tl = tiledlayout(n, 2, 'TileSpacing', 'compact', 'Padding', 'loose');

for i = 1:n
    e = entries(i);

    if strcmp(map_type, 'mae')
        map1 = e.mae_opt_slice;
        map2 = e.mae_pred_slice;
        title_text = 'MAE';
    else
        map1 = e.rmse_opt_slice;
        map2 = e.rmse_pred_slice;
        title_text = 'RMSE';
    end

    ax1 = nexttile; draw_tile(ax1, e.available, map1, climv);
    ax2 = nexttile; draw_tile(ax2, e.available, map2, climv);

    if i == 1
        title(ax1, 'optimal');
        title(ax2, 'predicted');
    end

    add_snr_label(ax1, e.snr_tag);

    if e.available
        text(ax1, 0.02, 0.98, metric_text(e.metrics_opt), 'Units', 'normalized', ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
            'Color', 'w', 'FontSize', 8, 'BackgroundColor', [0 0 0]);
        text(ax2, 0.02, 0.98, metric_text(e.metrics_pred), 'Units', 'normalized', ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
            'Color', 'w', 'FontSize', 8, 'BackgroundColor', [0 0 0]);
    end
end

cb = colorbar(tl, 'eastoutside');
cb.Label.String = title_text;
cb.FontSize = 9;

sgtitle(tl, sprintf('%s | %s (global clim)', entries(1).case_name, title_text));
exportgraphics(h, out_png, 'Resolution', 220);
close(h);
end

function add_snr_label(ax, snr_tag)
text(ax, -0.10, 0.5, snr_tag, 'Units', 'normalized', ...
    'Rotation', 90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontWeight', 'bold', 'Color', 'k', 'Clipping', 'off');
end

function draw_tile(ax, available, img, climv)
axes(ax);
if available && ~isempty(img)
    imagesc(img);
    axis image off;
    colormap(ax, 'jet');
    clim(climv);
else
    imagesc(zeros(32,32));
    axis image off;
    colormap(ax, 'gray');
    clim([0 1]);
    text(ax, 0.5, 0.5, 'MISSING', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontWeight', 'bold', 'Color', [0.9 0.1 0.1]);
end
end

function txt = metric_text(m)
txt = sprintf('MAE %.4f | RMSE %.4f | SSIM %.4f', m.MAE, m.RMSE, m.SSIM);
end

function records = discover_recon_records(root_dir)
records = struct('case_name', {}, 'snr_tag', {}, 'snr_val', {}, 'path', {});
all_files = [dir(fullfile(root_dir, '**', '*_sigma_recon.nii.gz')); ...
    dir(fullfile(root_dir, '**', '*_sigma_recon.nii'))];
for i = 1:numel(all_files)
    name = all_files(i).name;
    tok = regexp(name, '^(M\d+)_(SNR\d{3})_sigma_recon\.nii(\.gz)?$', 'tokens', 'once');
    if isempty(tok)
        continue;
    end
    case_name = tok{1};
    snr_tag = tok{2};
    snr_val = str2double(regexprep(snr_tag, '^SNR', ''));
    records(end+1).case_name = case_name; %#ok<AGROW>
    records(end).snr_tag = snr_tag;
    records(end).snr_val = snr_val;
    records(end).path = fullfile(all_files(i).folder, name);
end
end

function p = find_recon_path(records, case_name, snr_tag)
p = '';
if isempty(records)
    return;
end
idx = find(strcmp({records.case_name}, case_name) & strcmp({records.snr_tag}, snr_tag), 1, 'first');
if ~isempty(idx)
    p = records(idx).path;
end
end

function names = sort_case_names(names)
if isempty(names)
    return;
end
nums = nan(numel(names),1);
for i = 1:numel(names)
    v = sscanf(names{i}, 'M%d');
    if ~isempty(v)
        nums(i) = v;
    end
end
[~, idx] = sortrows([isnan(nums), nums]);
names = names(idx);
end

function arr = parse_int_list(s)
arr = [];
if isempty(s)
    return;
end
toks = regexp(strrep(s, ' ', ''), '[,;]', 'split');
vals = nan(1, numel(toks));
for i = 1:numel(toks)
    vals(i) = str2double(toks{i});
end
vals = vals(isfinite(vals));
vals = vals(vals > 0);
arr = round(vals(:))';
end

function val = getenv_default(name, default_val)
val = getenv(name);
if isempty(val)
    val = default_val;
end
end

function n = parse_positive_int(s, default_val)
n = str2double(strtrim(char(s)));
if ~isfinite(n) || n < 1 || mod(n,1) ~= 0
    n = default_val;
end
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
