%% recon_fixcond17.m
% Fixed-kernel conductivity reconstruction from phase5 noisy MAT files.
% - Input:  <PHASE5_ROOT>/M*/SNR*/noisy_phase_SNRxxx.mat
% - Output: <PHASE5_ROOT>/M*/SNR*/M*_SNR*_fixcond.nii.gz
% - Kernel: kDiffSize = [17 17 17] (Laplacian-form path: no kIntegralSize)
%
% Modes:
% - If SGE_TASK_ID is set: run one mapped (case, snr) pair only.
% - Otherwise: run all discovered pairs.

clear; clc;
warning('off', 'backtrace');

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab', 'functions'));
addpath(genpath(fullfile(repo_root, 'matlab', 'toolboxes')));

PHASE5_ROOT = getenv_default('PHASE5_ROOT', '/home/zcemexx/Scratch/outputs/phase5');
if ~isfolder(PHASE5_ROOT)
    error('PHASE5_ROOT not found: %s', PHASE5_ROOT);
end

fprintf('PHASE5_ROOT: %s\n', PHASE5_ROOT);

tasks = collect_tasks(PHASE5_ROOT);
n_total = numel(tasks);
if n_total == 0
    error('No runnable pairs found under: %s', PHASE5_ROOT);
end

task_id = parse_task_id(n_total);
if isempty(task_id)
    run_indices = 1:n_total;
    fprintf('SGE_TASK_ID not set -> run all pairs (%d).\n', n_total);
else
    run_indices = task_id;
    t = tasks(task_id);
    fprintf('SGE_TASK_ID=%d/%d -> Case=%s, SNR=%s\n', ...
        task_id, n_total, t.case_name, t.snr_tag);
end

params = struct();
params.B0 = 3;
params.VoxelSize = [1 1 1];
params.kDiffSize = [17 17 17];
estimate_noise = true;

total_run = 0;
total_ok = 0;
total_fail = 0;

for ii = run_indices
    total_run = total_run + 1;
    task = tasks(ii);

    fprintf('\n[RUN ] (%d/%d) %s %s\n', total_run, numel(run_indices), task.case_name, task.snr_tag);
    fprintf('      mat: %s\n', task.mat_path);
    fprintf('      out: %s\n', task.out_nii);

    try
        [phase_map, mag, seg] = load_inputs_from_mat(task.mat_path);
        mask = seg > 0;

        if ~isequal(size(phase_map), size(mag), size(seg))
            error('size_mismatch: phase/magnitude/segmentation shapes are inconsistent.');
        end
        if ~any(mask(:))
            error('empty_mask: segmentation>0 is empty.');
        end

        cond = conductivityMapping(phase_map, mask, params, ...
            'magnitude', mag, 'segmentation', seg, 'estimatenoise', estimate_noise);
        cond = single(cond);
        cond(cond < 0 | cond > 10) = NaN;

        save_nii_gz(cond, task.out_nii); % overwrite by design
        total_ok = total_ok + 1;
        fprintf('[ OK ] %s\n', task.out_nii);
    catch ME
        total_fail = total_fail + 1;
        fprintf(2, '[FAIL] %s %s: %s\n', task.case_name, task.snr_tag, error_status_text(ME));
    end
end

fprintf('\n===== recon_fixcond17 summary =====\n');
fprintf('Discovered pairs: %d\n', n_total);
fprintf('Executed pairs  : %d\n', numel(run_indices));
fprintf('Succeeded       : %d\n', total_ok);
fprintf('Failed          : %d\n', total_fail);

if total_fail > 0
    error('recon_fixcond17 finished with failures.');
end

%% ------------------------- Local functions -------------------------

function tasks = collect_tasks(phase5_root)
case_dirs = dir(fullfile(phase5_root, 'M*'));
case_dirs = case_dirs([case_dirs.isdir]);
case_dirs = case_dirs(~startsWith({case_dirs.name}, '.'));
case_dirs = sort_dirs_by_number(case_dirs, '^M(\d+)$');

tasks = struct('case_name', {}, 'snr_tag', {}, 'mat_path', {}, 'snr_path', {}, 'out_nii', {});

for ic = 1:numel(case_dirs)
    case_name = case_dirs(ic).name;
    case_path = fullfile(case_dirs(ic).folder, case_name);

    snr_dirs = dir(fullfile(case_path, 'SNR*'));
    snr_dirs = snr_dirs([snr_dirs.isdir]);
    snr_dirs = snr_dirs(~startsWith({snr_dirs.name}, '.'));
    snr_dirs = sort_dirs_by_number(snr_dirs, '^SNR(\d+)$');

    for is = 1:numel(snr_dirs)
        snr_tag = snr_dirs(is).name;
        snr_path = fullfile(snr_dirs(is).folder, snr_tag);

        mat_name = sprintf('noisy_phase_%s.mat', snr_tag);
        mat_path = fullfile(snr_path, mat_name);
        if ~isfile(mat_path)
            continue;
        end

        out_nii = fullfile(snr_path, sprintf('%s_%s_fixcond.nii.gz', case_name, snr_tag));
        tasks(end+1) = struct( ... %#ok<AGROW>
            'case_name', case_name, ...
            'snr_tag', snr_tag, ...
            'mat_path', mat_path, ...
            'snr_path', snr_path, ...
            'out_nii', out_nii);
    end
end
end

function out = sort_dirs_by_number(in_dirs, pattern)
if isempty(in_dirs)
    out = in_dirs;
    return;
end

names = {in_dirs.name};
nums = nan(numel(in_dirs), 1);
for i = 1:numel(in_dirs)
    tok = regexp(names{i}, pattern, 'tokens', 'once');
    if ~isempty(tok)
        nums(i) = str2double(tok{1});
    end
end

valid_idx = find(~isnan(nums));
invalid_idx = find(isnan(nums));

[~, ord_valid] = sort(nums(valid_idx), 'ascend');
valid_idx = valid_idx(ord_valid);

if ~isempty(invalid_idx)
    invalid_names = names(invalid_idx);
    invalid_names = cellfun(@lower, invalid_names, 'UniformOutput', false);
    [~, ord_invalid] = sort(invalid_names);
    invalid_idx = invalid_idx(ord_invalid);
end

out = in_dirs([valid_idx; invalid_idx]);
end

function task_id = parse_task_id(n_total)
task_id = [];
task_id_str = getenv('SGE_TASK_ID');
if isempty(task_id_str) || strcmpi(task_id_str, 'undefined')
    return;
end

task_id = str2double(task_id_str);
if ~isfinite(task_id) || task_id ~= floor(task_id)
    error('invalid_task_id: SGE_TASK_ID=%s must be an integer in [1,%d].', task_id_str, n_total);
end
if task_id < 1 || task_id > n_total
    error('invalid_task_id_range: SGE_TASK_ID=%d out of range [1,%d].', task_id, n_total);
end
end

function [phase_map, mag, seg] = load_inputs_from_mat(mat_path)
if ~isfile(mat_path)
    error('missing_input_mat: %s', mat_path);
end

data = load(mat_path);

phase_map = single(pick_field(data, {'phi0_noisy', 'Transceive_phase'}));
seg = single(pick_field(data, {'seg', 'tissueMask', 'Segmentation'}));
mag = single(pick_magnitude(data));
end

function out = pick_magnitude(data)
if isfield(data, 'magnitude_noisy')
    out = data.magnitude_noisy;
    return;
end
if isfield(data, 'mag')
    out = data.mag;
    return;
end
if isfield(data, 'B1plus_mag') && isfield(data, 'B1minus_mag')
    out = data.B1plus_mag .* data.B1minus_mag;
    return;
end
if isfield(data, 'T1w')
    out = data.T1w;
    return;
end
error('missing_magnitude: expected one of magnitude_noisy, mag, B1plus_mag/B1minus_mag, T1w');
end

function out = pick_field(data, candidates)
for i = 1:numel(candidates)
    name = candidates{i};
    if isfield(data, name)
        out = data.(name);
        return;
    end
end
error('missing_field: none of [%s] found', strjoin(candidates, ', '));
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
