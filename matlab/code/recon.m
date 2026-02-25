%% Auto batch reconstruction from predicted radius maps
% Index logic:
% 1) scan phase5/<Case>/SNRxxx/noisy_phase_SNRxxx.mat
% 2) find predicted radius map (prefer same SNR folder, fallback to prediction folder)
% 3) reconstruct and save output in the same SNR folder by default

clear; clc;
warning('off', 'backtrace');

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab', 'functions'));
addpath(genpath(fullfile(repo_root, 'matlab', 'toolboxes')));

% -------- Config (override with environment variables) --------
phase5_root = getenv_default('PHASE5_ROOT', '/home/zcemexx/Scratch/outputs/phase5');
pred_root = getenv_default('RADIUS_PRED_ROOT', '/home/zcemexx/Scratch/pred/preds_local');
out_root = getenv_default('RECON_OUT_ROOT', fullfile(pred_root, 'sigma_recon'));
recon_inplace_io = parse_bool(getenv_default('RECON_INPLACE_IO', 'true'));

if ~isfolder(phase5_root)
    error('PHASE5_ROOT not found: %s', phase5_root);
end
if ~isfolder(pred_root)
    error('RADIUS_PRED_ROOT not found: %s', pred_root);
end
if ~recon_inplace_io && ~isfolder(out_root)
    mkdir(out_root);
end

fprintf('PHASE5_ROOT: %s\n', phase5_root);
fprintf('RADIUS_PRED_ROOT: %s\n', pred_root);
if recon_inplace_io
    fprintf('RECON_OUT_ROOT: <same as phase5 case/snr folder>\n');
else
    fprintf('RECON_OUT_ROOT: %s\n', out_root);
end

case_dirs = dir(fullfile(phase5_root, 'M*'));
case_dirs = case_dirs([case_dirs.isdir]);
case_dirs = case_dirs(~startsWith({case_dirs.name}, '.'));
case_dirs = sort_struct_by_name(case_dirs);

total_found = 0;
total_ok = 0;
total_skip = 0;
total_fail = 0;

for ic = 1:numel(case_dirs)
    case_name = case_dirs(ic).name; % e.g., M12
    case_path = fullfile(case_dirs(ic).folder, case_name);

    snr_dirs = dir(fullfile(case_path, 'SNR*'));
    snr_dirs = snr_dirs([snr_dirs.isdir]);
    snr_dirs = snr_dirs(~startsWith({snr_dirs.name}, '.'));
    snr_dirs = sort_struct_by_name(snr_dirs);

    for is = 1:numel(snr_dirs)
        snr_tag = snr_dirs(is).name; % e.g., SNR010
        snr_path = fullfile(snr_dirs(is).folder, snr_tag);
        total_found = total_found + 1;

        mat_name = sprintf('noisy_phase_%s.mat', snr_tag);
        mat_path = fullfile(snr_path, mat_name);
        if ~isfile(mat_path)
            fprintf('[SKIP] Missing mat: %s\n', mat_path);
            total_skip = total_skip + 1;
            continue;
        end

        radius_file = find_radius_map(snr_path, pred_root, case_name, snr_tag);
        if isempty(radius_file)
            fprintf('[SKIP] Missing radius map for %s %s (searched in %s and %s)\n', case_name, snr_tag, snr_path, pred_root);
            total_skip = total_skip + 1;
            continue;
        end

        if recon_inplace_io
            out_dir_case = snr_path;
        else
            out_dir_case = fullfile(out_root, case_name, snr_tag);
            if ~isfolder(out_dir_case)
                mkdir(out_dir_case);
            end
        end
        out_nii = fullfile(out_dir_case, sprintf('%s_%s_sigma_recon.nii.gz', case_name, snr_tag));

        fprintf('[RUN ] %s %s\n', case_name, snr_tag);
        fprintf('      mat    : %s\n', mat_path);
        fprintf('      radius : %s\n', radius_file);
        fprintf('      out    : %s\n', out_nii);

        try
            setenv('INPUT_DATA', mat_path);
            setenv('RADIUS_NII', radius_file);
            setenv('OUT_NII', out_nii);
            reconstruct_conductivity_from_radiusmap();
            total_ok = total_ok + 1;
        catch ME
            total_fail = total_fail + 1;
            fprintf(2, '[FAIL] %s %s: %s\n', case_name, snr_tag, ME.message);
        end
    end
end

fprintf('\n===== Recon Summary =====\n');
fprintf('Pairs discovered: %d\n', total_found);
fprintf('Succeeded       : %d\n', total_ok);
fprintf('Skipped         : %d\n', total_skip);
fprintf('Failed          : %d\n', total_fail);

if total_fail > 0
    error('Batch reconstruction finished with failures. See log above.');
end

%% -------- Local helpers --------
function out = getenv_default(name, default_val)
out = getenv(name);
if isempty(out)
    out = default_val;
end
end

function tf = parse_bool(x)
x = lower(strtrim(char(x)));
tf = any(strcmp(x, {'1', 'true', 'yes', 'y', 'on'}));
end

function s = sort_struct_by_name(s)
if isempty(s)
    return;
end
[~, idx] = sort({s.name});
s = s(idx);
end

function radius_path = find_radius_map(local_snr_path, pred_root, case_name, snr_tag)
radius_path = '';

% Only check prediction folder (do not use local SNR radius maps):
%   <pred_root>/<Case>_<SNR>.nii.gz
%   <pred_root>/<Case>_<SNR>.nii
candidates = {
    fullfile(pred_root, sprintf('%s_%s.nii.gz', case_name, snr_tag))
    fullfile(pred_root, sprintf('%s_%s.nii', case_name, snr_tag))
    fullfile(pred_root, case_name, sprintf('%s_%s.nii.gz', case_name, snr_tag))
    fullfile(pred_root, case_name, sprintf('%s_%s.nii', case_name, snr_tag))
    };

for i = 1:numel(candidates)
    if isfile(candidates{i})
        radius_path = candidates{i};
        return;
    end
end

% Fallback wildcard in pred_root top-level
pat1 = fullfile(pred_root, sprintf('%s_%s*.nii.gz', case_name, snr_tag));
hit = dir(pat1);
if ~isempty(hit)
    radius_path = fullfile(hit(1).folder, hit(1).name);
    return;
end

pat2 = fullfile(pred_root, sprintf('%s_%s*.nii', case_name, snr_tag));
hit = dir(pat2);
if ~isempty(hit)
    radius_path = fullfile(hit(1).folder, hit(1).name);
    return;
end

% Fallback wildcard in pred_root/<Case>
pat3 = fullfile(pred_root, case_name, sprintf('%s_%s*.nii.gz', case_name, snr_tag));
hit = dir(pat3);
if ~isempty(hit)
    radius_path = fullfile(hit(1).folder, hit(1).name);
    return;
end

pat4 = fullfile(pred_root, case_name, sprintf('%s_%s*.nii', case_name, snr_tag));
hit = dir(pat4);
if ~isempty(hit)
    radius_path = fullfile(hit(1).folder, hit(1).name);
end
end
