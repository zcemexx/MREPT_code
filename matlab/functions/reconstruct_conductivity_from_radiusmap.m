function reconstruct_conductivity_from_radiusmap()
% Reconstruct conductivity map from a predicted radius map NIfTI.
%
% Required env vars:
%   RADIUS_NII  path to radius map NIfTI (.nii or .nii.gz)
%   OUT_NII     output conductivity NIfTI path (.nii or .nii.gz)
%
% Optional env vars:
%   INPUT_DATA  if provided and ends with .mat, load inputs from this MAT file.
%               For MAT mode, required fields are:
%               - phi0_noisy (or Transceive_phase)
%               - seg/tissueMask/Segmentation
%               - mag/magnitude_noisy (or B1plus_mag/B1minus_mag, or T1w)
%               In MAT mode, PHASE_NII/MASK_NII/MAG_NII/SEG_NII are ignored.
%
%   PHASE_NII   phase NIfTI (for non-MAT mode)
%   MASK_NII    mask NIfTI (for non-MAT mode)
%   MAG_NII     magnitude NIfTI (for Mag/Seg EPT)
%   SEG_NII     segmentation NIfTI (for Seg/Mag+Seg EPT)
%   B0_T        field strength in Tesla, default 3
%   VOXEL_MM    optional voxel size override in mm, format "1,1,1".
%               If unset, infer from NIfTI header (PHASE_NII in NIfTI mode,
%               RADIUS_NII in MAT mode).
%   ESTIMATE_NOISE  true/false, default false
%   RADIUS_MIN  default 1
%   RADIUS_MAX  default 30
%
% Example:
%   RADIUS_NII=/path/radiusmap.nii.gz \
%   INPUT_DATA=/path/noisy_phase_SNR010.mat \
%   OUT_NII=/path/sigma_recon.nii.gz \
%   matlab -nodisplay -nodesktop -r "cd('matlab/code'); reconstruct_conductivity_from_radiusmap; exit"

warning('off', 'backtrace');

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab', 'functions'));
addpath(genpath(fullfile(repo_root, 'matlab', 'toolboxes')));

radius_nii = must_getenv('RADIUS_NII');
out_nii = must_getenv('OUT_NII');

input_data = getenv_default('INPUT_DATA', '');
mag_nii = getenv_default('MAG_NII', '');
seg_nii = getenv_default('SEG_NII', '');
b0_t = str2double(getenv_default('B0_T', '3'));
estimate_noise = parse_bool(getenv_default('ESTIMATE_NOISE', 'false'));
rmin = str2double(getenv_default('RADIUS_MIN', '1'));
rmax = str2double(getenv_default('RADIUS_MAX', '30'));

phase_nii = '';
mask_nii = '';
if ~isempty(input_data) && is_mat_path(input_data)
    voxel_mm = infer_voxel_mm(radius_nii, 'RADIUS_NII');
else
    phase_nii = must_getenv('PHASE_NII');
    mask_nii = must_getenv('MASK_NII');
    voxel_mm = infer_voxel_mm(phase_nii, 'PHASE_NII');
end

fprintf('Loading inputs...\n');
radius_map = single(load_nii_any(radius_nii));
if ~isempty(input_data) && is_mat_path(input_data)
    [phase_map, mask, mag, seg] = load_inputs_from_mat(input_data);
else
    phase_map = single(load_nii_any(phase_nii));
    mask_img = load_nii_any(mask_nii);
    mask = mask_img > 0;

    if ~isempty(mag_nii)
        mag = single(load_nii_any(mag_nii));
    else
        mag = [];
    end

    if ~isempty(seg_nii)
        seg = single(load_nii_any(seg_nii));
    else
        seg = [];
    end
end

if ~isequal(size(radius_map), size(phase_map), size(mask))
    error('Input size mismatch: radius, phase, mask must have identical shape.');
end
if ~isempty(mag) && ~isequal(size(mag), size(mask))
    error('Input size mismatch: magnitude does not match mask.');
end
if ~isempty(seg) && ~isequal(size(seg), size(mask))
    error('Input size mismatch: segmentation does not match mask.');
end

radius_map(~isfinite(radius_map)) = 0;
radius_map = round(radius_map);
radius_map = max(rmin, min(rmax, radius_map));
radius_map(~mask) = 0;

params = struct();
params.B0 = b0_t;
params.VoxelSize = voxel_mm;
fprintf('Using voxel size [mm]: [%g %g %g]\n', voxel_mm(1), voxel_mm(2), voxel_mm(3));
% No integral kernel -> Laplacian-form EPT (same as your training data generation path).

sigma = zeros(size(phase_map), 'single');
unique_r = unique(radius_map(mask));
unique_r = unique_r(unique_r >= rmin & unique_r <= rmax);

fprintf('Reconstructing conductivity for %d radius values...\n', numel(unique_r));
for i = 1:numel(unique_r)
    r = unique_r(i);
    params_r = params;
    params_r.kDiffSize = [2*r+1, 2*r+1, 2*r+1];

    if ~isempty(mag) && ~isempty(seg)
        cond_r = conductivityMapping(phase_map, mask, params_r, ...
            'magnitude', mag, 'segmentation', seg, 'estimatenoise', estimate_noise);
    elseif ~isempty(mag)
        cond_r = conductivityMapping(phase_map, mask, params_r, ...
            'magnitude', mag, 'estimatenoise', estimate_noise);
    elseif ~isempty(seg)
        cond_r = conductivityMapping(phase_map, mask, params_r, ...
            'segmentation', seg);
    else
        cond_r = conductivityMapping(phase_map, mask, params_r);
    end

    vx = (radius_map == r) & mask;
    sigma(vx) = single(cond_r(vx));
    fprintf('  radius=%d done, voxels=%d\n', r, nnz(vx));
end

sigma(~mask) = 0;
save_nii_gz(sigma, out_nii);
fprintf('Saved conductivity map: %s\n', out_nii);
end

function [phase_map, mask, mag, seg] = load_inputs_from_mat(mat_path)
if ~isfile(mat_path)
    error('INPUT_DATA .mat not found: %s', mat_path);
end
data = load(mat_path);

phase_map = pick_field(data, {'phi0_noisy', 'Transceive_phase'});
seg = pick_field(data, {'seg', 'tissueMask', 'Segmentation'});
mag = pick_magnitude(data);

phase_map = single(phase_map);
seg = single(seg);
mag = single(mag);
mask = seg > 0;

if ~isequal(size(phase_map), size(seg), size(mag))
    error('MAT field shape mismatch: phase/seg/mag must have same size in %s', mat_path);
end
end

function out = pick_magnitude(data)
if isfield(data, 'mag')
    out = data.mag;
    return;
end
if isfield(data, 'magnitude_noisy')
    out = data.magnitude_noisy;
    return;
end
if isfield(data, 'B1plus_mag') && isfield(data, 'B1minus_mag')
    out = data.B1plus_mag .* data.B1minus_mag;
    return;
end
if isfield(data, 'B1plus_mag')
    out = data.B1plus_mag;
    return;
end
if isfield(data, 'T1w')
    out = data.T1w;
    return;
end
error('Missing magnitude field in MAT. Need one of: mag, magnitude_noisy, B1plus_mag/B1minus_mag, T1w');
end

function out = pick_field(data, names)
for i = 1:numel(names)
    if isfield(data, names{i})
        out = data.(names{i});
        return;
    end
end
error('Missing required field. Tried: %s', strjoin(names, ', '));
end

function val = must_getenv(name)
val = getenv(name);
if isempty(val)
    error('Missing required env var: %s', name);
end
end

function val = getenv_default(name, default_val)
val = getenv(name);
if isempty(val)
    val = default_val;
end
end

function out = parse_bool(x)
if islogical(x)
    out = x;
    return;
end
x = lower(strtrim(char(x)));
out = any(strcmp(x, {'1', 'true', 'yes', 'y', 'on'}));
end

function v = parse_vec3(s)
tokens = regexp(strrep(s, ' ', ''), '[,;]', 'split');
if numel(tokens) ~= 3
    error('VOXEL_MM must have 3 values, got: %s', s);
end
v = [str2double(tokens{1}), str2double(tokens{2}), str2double(tokens{3})];
if any(~isfinite(v)) || any(v <= 0)
    error('VOXEL_MM values must be positive numbers, got: %s', s);
end
end

function v = infer_voxel_mm(reference_nii, source_name)
override = getenv('VOXEL_MM');
if ~isempty(override)
    v = parse_vec3(override);
    fprintf('VOXEL_MM override from env: [%g %g %g]\n', v(1), v(2), v(3));
    return;
end

v = read_voxel_mm_from_nifti(reference_nii);
fprintf('VOXEL_MM inferred from %s header (%s): [%g %g %g]\n', ...
    source_name, reference_nii, v(1), v(2), v(3));
end

function v = read_voxel_mm_from_nifti(path_in)
if exist('niftiinfo', 'file') == 2
    info = niftiinfo(path_in);
    if isfield(info, 'PixelDimensions') && numel(info.PixelDimensions) >= 3
        v = double(info.PixelDimensions(1:3));
    else
        error('NIfTI header missing PixelDimensions: %s', path_in);
    end
elseif exist('nii_tool', 'file') == 2
    hdr = nii_tool('hdr', path_in);
    if isfield(hdr, 'pixdim') && numel(hdr.pixdim) >= 4
        v = double(hdr.pixdim(2:4));
    elseif isfield(hdr, 'hdr') && isfield(hdr.hdr, 'pixdim') && numel(hdr.hdr.pixdim) >= 4
        v = double(hdr.hdr.pixdim(2:4));
    else
        error('NIfTI header missing pixdim: %s', path_in);
    end
else
    error('No NIfTI header reader available. Need niftiinfo or nii_tool.');
end

if any(~isfinite(v)) || any(v <= 0)
    error('Invalid voxel size read from NIfTI header for %s: [%g %g %g]', ...
        path_in, v(1), v(2), v(3));
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

function tf = is_mat_path(p)
tf = ~isempty(regexp(lower(p), '\.mat$', 'once'));
end
