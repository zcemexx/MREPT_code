%% sweep_radius_metrics_online.m
% -------------------------------------------------------------------------
% Purpose
% -------------------------------------------------------------------------
% This script performs NaN-safe radius sweep evaluation for conductivity
% reconstruction with a strict physical filter, optional parallel radius
% mapping, and an ordered online winner update engine (arena-style). It
% produces two CSV tables per task:
%
% 1) per_radius_metrics.csv
%    - Fixed-kernel benchmark across Radius = 1..40.
%    - Rows are recorded for Global/WM/GM/CSF tissues separately.
%
% 2) cap_curve_metrics.csv
%    - Oracle upper-bound curve across Cap = 1..40, where each cap uses the
%      best voxel-wise radius seen so far (online winner map).
%    - Includes tissue-wise radius distribution counts: R01_Count..R40_Count.
%
% -------------------------------------------------------------------------
% Data Sources and Mask Policy (important)
% -------------------------------------------------------------------------
% GT source: <Case>.mat
%   - Conductivity_GT
%   - Segmentation   (the ONLY source for tissue labels/masks)
%
% Noisy source: noisy_phase_SNRxxx.mat
%   - phi0_noisy
%   - magnitude_noisy
%
% NOTE:
%   Even if noisy_phase_SNRxxx.mat contains tissueMask, this script ignores it
%   by design. All tissue definitions use GT Segmentation only:
%     Global: Segmentation > 0
%     WM:     Segmentation == 1
%     GM:     Segmentation == 2
%     CSF:    Segmentation == 3
%
% -------------------------------------------------------------------------
% Job Array Mapping (aligned with exp.m)
% -------------------------------------------------------------------------
% This script is task-based. One task processes ONE (case, snr) pair.
% Mapping:
%   n_total = numel(case_files) * numel(snr_list)
%   case_idx = floor((task_id - 1) / n_snr) + 1
%   snr_idx  = mod(task_id - 1, n_snr) + 1
%
% Example (default list):
%   task 1   -> M6  + SNR010
%   task 8   -> M6  + SNR150
%   task 9   -> M8  + SNR010
%   ...
%
% task_id source:
%   - Cluster: SGE_TASK_ID
%   - Local debug fallback: task_id = 1
%
% -------------------------------------------------------------------------
% Core Algorithm
% -------------------------------------------------------------------------
% Radius search space: 1..40
% For each radius r:
%   1) Run conductivityMapping(...)
%   2) Apply physical filter: cond_r(cond_r < 0 | cond_r > 10) = NaN
%   3) Compute error map e_r = abs(cond_r - sigma_gt)
%   4) Update online winner maps using:
%        better_mask = (e_r < min_err_map) & ~isnan(e_r)
%      and update min_err_map / best_r_map / best_cond_map on better_mask.
%
% Memory behavior:
%   - Serial mode: O(N) streaming update (no full radius stack).
%   - Parallel mode: caches one conductivity volume per radius before
%     ordered cap-curve update.
%
% -------------------------------------------------------------------------
% Metrics (NaN-safe, fair normalization)
% -------------------------------------------------------------------------
% nMAE must be computed on the same valid voxels for numerator/denominator:
%   valid_mask = tissue_mask & isfinite(pred) & isfinite(gt)
%   mean_err   = mean(abs(pred(valid_mask)-gt(valid_mask)))
%   mean_gt    = mean(gt(valid_mask))
%   nMAE       = mean_err / mean_gt
%
% RMSE:
%   sqrt(mean((pred(valid_mask)-gt(valid_mask)).^2))
%
% Valid_Ratio:
%   N_valid / N_total_tissue
%
% Oracle_Coverage (cap table only):
%   nnz(best_r_map > 0 within tissue) / N_total_tissue
%
% Robustness behavior:
%   - If tissue is empty (N_total=0), metrics are NaN.
%   - If N_valid=0 or mean_gt<=0, nMAE is NaN.
%   - Failures in one case/SNR are logged as status rows; script continues.
%
% -------------------------------------------------------------------------
% CSV schema
% -------------------------------------------------------------------------
% per_radius_metrics.csv
%   Case,SNR,Radius,Tissue,nMAE,RMSE,Valid_Ratio,N_Valid,N_Total,Status
%
% cap_curve_metrics.csv
%   Case,SNR,Cap,Tissue,nMAE,RMSE,Valid_Ratio,Oracle_Coverage,N_Valid,N_Total,
%   R01_Count,...,R40_Count,Status
%
% -------------------------------------------------------------------------
% Workload definitions
% -------------------------------------------------------------------------
% Reuses exp.m default list:
%   case_files = {'M6.mat','M8.mat','M12.mat','M19.mat','M22.mat','M24.mat',
%                 'M39.mat','M40.mat','M41.mat','M42.mat','M43.mat','M50.mat',
%                 'M66.mat','M70.mat','M75.mat','M79.mat','M84.mat'}
%   snr_list   = [10 20 30 40 50 75 100 150]
%
% Environment variable overrides (optional):
%   SWEEP_PHASE5_ROOT
%   SWEEP_GT_ROOT
%   SWEEP_OUT_DIR
%
% -------------------------------------------------------------------------

clear; clc;
warning('off', 'backtrace');

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab', 'functions'));
addpath(genpath(fullfile(repo_root, 'matlab', 'toolboxes')));

PHASE5_ROOT = getenv_default('SWEEP_PHASE5_ROOT', '/myriadfs/home/zcemexx/Scratch/outputs/phase5');
GT_ROOT = getenv_default('SWEEP_GT_ROOT', '/myriadfs/home/zcemexx/Scratch/data/ADEPT_raw');
OUT_DIR = getenv_default('SWEEP_OUT_DIR', fullfile(PHASE5_ROOT, 'radiustest'));

if ~isfolder(PHASE5_ROOT)
    error('PHASE5_ROOT not found: %s', PHASE5_ROOT);
end
if ~isfolder(GT_ROOT)
    error('GT_ROOT not found: %s', GT_ROOT);
end
if ~isfolder(OUT_DIR)
    mkdir(OUT_DIR);
end

fprintf('PHASE5_ROOT: %s\n', PHASE5_ROOT);
fprintf('GT_ROOT: %s\n', GT_ROOT);
fprintf('OUT_DIR: %s\n', OUT_DIR);

case_files = {'M6.mat', 'M8.mat', 'M12.mat', 'M19.mat', 'M22.mat', ...
    'M24.mat', 'M39.mat', 'M40.mat', 'M41.mat', 'M42.mat', ...
    'M43.mat', 'M50.mat', 'M66.mat', 'M70.mat', 'M75.mat', ...
    'M79.mat', 'M84.mat'};
snr_list = [10, 20, 30, 40, 50, 75, 100, 150];

n_cases = numel(case_files);
n_snr = numel(snr_list);
n_total = n_cases * n_snr;

task_id = parse_task_id(n_total);
mapping = build_task_mapping(case_files, snr_list, task_id);

this_case_file = mapping.case_file;
case_name = mapping.case_name;
snr_val = mapping.snr;
snr_tag = sprintf('SNR%03d', snr_val);

fprintf('Task %d/%d mapped to Case=%s, SNR=%d\n', task_id, n_total, case_name, snr_val);

radius_list = 1:40;
n_radius = numel(radius_list);

% --- Parallel pool config (aligned with exp.m) ---
poolObj = gcp('nocreate');
if isempty(poolObj)
    nSlotsStr = getenv('NSLOTS');
    nSlots = str2double(nSlotsStr);
    if ~isfinite(nSlots) || nSlots < 1
        nSlots = 1;
    end
    if nSlots > 1
        parpool('local', nSlots);
        poolObj = gcp('nocreate');
        fprintf('并行池已启动，核心数 (NSLOTS): %d\n', nSlots);
    else
        fprintf('运行在单核模式。\n');
    end
end

use_parallel = ~isempty(poolObj) && poolObj.NumWorkers > 1;
if use_parallel
    fprintf('Radius sweep 并行模式启用，workers=%d\n', poolObj.NumWorkers);
end

% Tissue definitions from GT Segmentation only.
TISSUE_NAMES = {'Global', 'WM', 'GM', 'CSF'};
TISSUE_CODES = [0, 1, 2, 3]; % 0 means Segmentation > 0

per_header = {'Case','SNR','Radius','Tissue','nMAE','RMSE','Valid_Ratio','N_Valid','N_Total','Status'};
count_headers = arrayfun(@(r) sprintf('R%02d_Count', r), radius_list, 'UniformOutput', false);
cap_header = [{'Case','SNR','Cap','Tissue','nMAE','RMSE','Valid_Ratio','Oracle_Coverage','N_Valid','N_Total'}, count_headers, {'Status'}];

per_rows = cell(0, numel(per_header));
cap_rows = cell(0, numel(cap_header));

Parameters.B0 = 3;
Parameters.VoxelSize = [1 1 1];
do_filter = true;
estimatenoise = true;
quietMappingLog = true;

gt_path = fullfile(GT_ROOT, this_case_file);
noisy_path = fullfile(PHASE5_ROOT, case_name, snr_tag, sprintf('noisy_phase_SNR%03d.mat', snr_val));

fprintf('\n[Task=%d] [Case=%s | SNR=%d] start\n', task_id, case_name, snr_val);

try
    [sigma_gt, seg, phi0_noisy, magnitude_noisy] = load_case_inputs(gt_path, noisy_path);

    if ~isequal(size(sigma_gt), size(seg), size(phi0_noisy), size(magnitude_noisy))
        error('size_mismatch: GT/Seg/noisy_phase/magnitude shapes are inconsistent.');
    end

    mask_global = seg > 0;
    if ~any(mask_global(:))
        error('empty_global_mask: Segmentation>0 is empty.');
    end

    min_err_map = inf(size(sigma_gt), 'single');
    best_r_map = zeros(size(sigma_gt), 'uint8');
    best_cond_map = nan(size(sigma_gt), 'single');

    tissue_masks = cell(1, numel(TISSUE_NAMES));
    for it = 1:numel(TISSUE_NAMES)
        tissue_masks{it} = get_tissue_mask(seg, TISSUE_CODES(it));
    end

    if use_parallel
        cond_cache = cell(1, n_radius);
        parfor ir = 1:n_radius
            r = radius_list(ir);
            params_r = Parameters;
            params_r.kDiffSize = [2*r+1, 2*r+1, 2*r+1];

            cond_r = run_mapping(phi0_noisy, mask_global, params_r, magnitude_noisy, seg, estimatenoise, quietMappingLog);

            if do_filter
                cond_r(cond_r < 0 | cond_r > 10) = NaN;
            end

            cond_cache{ir} = cond_r;
        end
    end

    for ir = 1:n_radius
        r = radius_list(ir);
        if use_parallel
            cond_r = cond_cache{ir};
            cond_cache{ir} = [];
        else
            params_r = Parameters;
            params_r.kDiffSize = [2*r+1, 2*r+1, 2*r+1];

            cond_r = run_mapping(phi0_noisy, mask_global, params_r, magnitude_noisy, seg, estimatenoise, quietMappingLog);

            if do_filter
                cond_r(cond_r < 0 | cond_r > 10) = NaN;
            end
        end

        e_r = abs(cond_r - sigma_gt);
        better_mask = (e_r < min_err_map) & ~isnan(e_r);

        min_err_map(better_mask) = e_r(better_mask);
        best_r_map(better_mask) = uint8(r);
        best_cond_map(better_mask) = cond_r(better_mask);

        % Table A: fixed radius performance.
        for it = 1:numel(TISSUE_NAMES)
            tissue_name = TISSUE_NAMES{it};
            tissue_mask = tissue_masks{it};
            [nmae, rmse, valid_ratio, n_valid, n_total_tissue] = calc_metrics_aligned(cond_r, sigma_gt, tissue_mask);

            per_rows(end+1, :) = {case_name, snr_val, r, tissue_name, nmae, rmse, valid_ratio, n_valid, n_total_tissue, 'ok'}; %#ok<AGROW>
        end

        % Table B: cap curve at Cap=r (online oracle best so far).
        for it = 1:numel(TISSUE_NAMES)
            tissue_name = TISSUE_NAMES{it};
            tissue_mask = tissue_masks{it};

            [nmae, rmse, valid_ratio, n_valid, n_total_tissue] = calc_metrics_aligned(best_cond_map, sigma_gt, tissue_mask);
            oracle_cov = calc_oracle_coverage(best_r_map, tissue_mask);
            counts = radius_counts(best_r_map, tissue_mask, n_radius);

            cap_rows(end+1, :) = [{case_name, snr_val, r, tissue_name, nmae, rmse, valid_ratio, oracle_cov, n_valid, n_total_tissue}, ...
                num2cell(counts), {'ok'}]; %#ok<AGROW>
        end
    end

    fprintf('[Task=%d] [Case=%s | SNR=%d] done\n', task_id, case_name, snr_val);

catch ME
    warn_msg = sprintf('[Task=%d] [Case=%s | SNR=%d] failed: %s', task_id, case_name, snr_val, ME.message);
    warning('%s', warn_msg);
    status_txt = error_status_text(ME);

    % Keep CSV schema stable by appending status rows.
    for it = 1:numel(TISSUE_NAMES)
        tissue_name = TISSUE_NAMES{it};
        per_rows(end+1, :) = {case_name, snr_val, NaN, tissue_name, NaN, NaN, NaN, NaN, NaN, status_txt}; %#ok<AGROW>
        cap_rows(end+1, :) = [{case_name, snr_val, NaN, tissue_name, NaN, NaN, NaN, NaN, NaN, NaN}, ...
            num2cell(zeros(1,n_radius)), {status_txt}]; %#ok<AGROW>
    end
end

task_out_dir = fullfile(OUT_DIR, case_name, snr_tag);
if ~isfolder(task_out_dir)
    mkdir(task_out_dir);
end

per_out = fullfile(task_out_dir, 'per_radius_metrics.csv');
cap_out = fullfile(task_out_dir, 'cap_curve_metrics.csv');
meta_out = fullfile(task_out_dir, 'task_mapping.json');

writecell([per_header; per_rows], per_out);
writecell([cap_header; cap_rows], cap_out);

task_meta = struct();
task_meta.TaskID = task_id;
task_meta.TotalTasks = n_total;
task_meta.Case = case_name;
task_meta.SNR = snr_val;
task_meta.CaseIndex = mapping.case_idx;
task_meta.SNRIndex = mapping.snr_idx;
fid = fopen(meta_out, 'w');
if fid ~= -1
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s', jsonencode(task_meta, 'PrettyPrint', true));
end

fprintf('\nSaved:\n  %s\n  %s\n  %s\n', per_out, cap_out, meta_out);

%% ------------------------- Local functions -------------------------

function task_id = parse_task_id(n_total)
task_id_str = getenv('SGE_TASK_ID');
if isempty(task_id_str) || strcmpi(task_id_str, 'undefined')
    task_id = 1;
    fprintf('Warning: SGE_TASK_ID not found. Local debug mode -> task_id=1\n');
    return;
end

task_id = str2double(task_id_str);
if ~isfinite(task_id) || task_id ~= floor(task_id)
    error('invalid_task_id: SGE_TASK_ID=%s is invalid. It must be an integer in [1,%d].', task_id_str, n_total);
end
if task_id < 1 || task_id > n_total
    error('invalid_task_id_range: SGE_TASK_ID=%d is out of range [1,%d].', task_id, n_total);
end
end

function mapping = build_task_mapping(case_files, snr_list, task_id)
n_snr = numel(snr_list);

case_idx = floor((task_id - 1) / n_snr) + 1;
snr_idx = mod(task_id - 1, n_snr) + 1;

case_file = case_files{case_idx};
snr_val = snr_list(snr_idx);

[~, case_name, ext] = fileparts(case_file);
if ~strcmpi(ext, '.mat')
    error('invalid_case_file: Case file must end with .mat, got %s', case_file);
end

mapping = struct();
mapping.case_idx = case_idx;
mapping.snr_idx = snr_idx;
mapping.case_file = case_file;
mapping.case_name = case_name;
mapping.snr = snr_val;
end

function [sigma_gt, seg, phi0_noisy, magnitude_noisy] = load_case_inputs(gt_path, noisy_path)
if ~isfile(gt_path)
    error('missing_gt: %s', gt_path);
end
if ~isfile(noisy_path)
    error('missing_noisy: %s', noisy_path);
end

gt = load(gt_path);
if ~isfield(gt, 'Conductivity_GT')
    error('missing_field_Conductivity_GT: %s', gt_path);
end
if ~isfield(gt, 'Segmentation')
    error('missing_field_Segmentation: %s', gt_path);
end

noisy = load(noisy_path);
if ~isfield(noisy, 'phi0_noisy')
    error('missing_field_phi0_noisy: %s', noisy_path);
end
if ~isfield(noisy, 'magnitude_noisy')
    error('missing_field_magnitude_noisy: %s', noisy_path);
end

sigma_gt = single(gt.Conductivity_GT);
seg = uint8(gt.Segmentation); % only GT Segmentation is used
phi0_noisy = single(noisy.phi0_noisy);
magnitude_noisy = single(noisy.magnitude_noisy);
end

function cond_r = run_mapping(phi0_noisy, mask_global, params_r, magnitude_noisy, seg, estimatenoise, quietMappingLog)
if quietMappingLog
    warnState = warning;
    warning('off', 'all');
    try
        [cond_r, ~] = conductivityMapping(phi0_noisy, mask_global, params_r, ...
            'magnitude', magnitude_noisy, 'segmentation', seg, 'estimatenoise', estimatenoise);
    catch ME
        warning(warnState);
        rethrow(ME);
    end
    warning(warnState);
else
    [cond_r, ~] = conductivityMapping(phi0_noisy, mask_global, params_r, ...
        'magnitude', magnitude_noisy, 'segmentation', seg, 'estimatenoise', estimatenoise);
end

cond_r = single(cond_r);
end

function tissue_mask = get_tissue_mask(seg, tissue_code)
if tissue_code == 0
    tissue_mask = seg > 0;
else
    tissue_mask = seg == tissue_code;
end
end

function [nmae, rmse, valid_ratio, n_valid, n_total] = calc_metrics_aligned(pred, gt, tissue_mask)
n_total = nnz(tissue_mask);
if n_total == 0
    nmae = NaN;
    rmse = NaN;
    valid_ratio = NaN;
    n_valid = 0;
    return;
end

valid = tissue_mask & isfinite(pred) & isfinite(gt);
n_valid = nnz(valid);
valid_ratio = double(n_valid) / double(n_total);

if n_valid == 0
    nmae = NaN;
    rmse = NaN;
    return;
end

d = pred(valid) - gt(valid);
mean_err = mean(abs(d), 'omitnan');
mean_gt = mean(gt(valid), 'omitnan');

if ~isfinite(mean_gt) || mean_gt <= 0
    nmae = NaN;
else
    nmae = mean_err / mean_gt;
end

rmse = sqrt(mean(d.^2, 'omitnan'));
end

function cov = calc_oracle_coverage(best_r_map, tissue_mask)
n_total = nnz(tissue_mask);
if n_total == 0
    cov = NaN;
    return;
end
cov = double(nnz(tissue_mask & (best_r_map > 0))) / double(n_total);
end

function counts = radius_counts(best_r_map, tissue_mask, max_radius)
counts = zeros(1, max_radius);
if ~any(tissue_mask(:))
    return;
end
rvals = double(best_r_map(tissue_mask));
for rr = 1:max_radius
    counts(rr) = nnz(rvals == rr);
end
end

function txt = error_status_text(ME)
if ~isempty(ME.identifier)
    txt = ME.identifier;
else
    txt = strrep(ME.message, sprintf('\n'), ' ');
end
end

function val = getenv_default(name, default_val)
val = getenv(name);
if isempty(val)
    val = default_val;
end
end
