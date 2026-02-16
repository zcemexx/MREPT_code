function save_nii_gz(img, filename)
ensure_nifti_writers_on_path();

outDir = fileparts(filename);
if ~isempty(outDir) && ~isfolder(outDir)
    mkdir(outDir);
end

filenameLower = lower(filename);
if endsWith(filenameLower, '.nii.gz')
    isGz = true;
    niiPath = filename(1:end-3); % strip ".gz"
elseif endsWith(filenameLower, '.nii')
    isGz = false;
    niiPath = filename;
else
    error('save_nii_gz:InvalidFilename', ...
        'Filename must end with .nii or .nii.gz: %s', filename);
end

imgToWrite = normalize_img_type(img);

% Primary writer: dicm2nii nii_tool
if exist('nii_tool', 'file') == 2
    try
        nii = nii_tool('init', imgToWrite);
        nii.hdr.pixdim(2:4) = [1 1 1];
        nii.img = imgToWrite;
        nii_tool('save', nii, niiPath); % write .nii first
        if isGz
            gzip_and_validate(niiPath, filename);
        end
        return;
    catch primaryErr
        warning('save_nii_gz:PrimaryWriterFailed', ...
            'nii_tool save failed (%s). Trying fallback writers.', primaryErr.message);
    end
end

% Fallback 1: MATLAB builtin niftiwrite
if exist('niftiwrite', 'file') == 2
    try
        niftiwrite(imgToWrite, niiPath, 'Compressed', false);
        if isGz
            gzip_and_validate(niiPath, filename);
        end
        return;
    catch niftiErr
        warning('save_nii_gz:NiftiwriteFailed', ...
            'niftiwrite failed (%s). Trying make_nii/save_nii fallback.', niftiErr.message);
    end
end

% Fallback 2: NIfTI_20140122 make_nii/save_nii
if exist('make_nii', 'file') ~= 2 || exist('save_nii', 'file') ~= 2
    error('save_nii_gz:NoAvailableWriter', ...
        ['Unable to save "%s": no working writer found. Tried nii_tool, niftiwrite, and make_nii/save_nii. ', ...
        'Add toolboxes/NIfTI_20140122 to path or set MREPT_TOOLBOX_ROOT.'], filename);
end

niiFallback = make_nii(single(imgToWrite));
niiFallback.img = imgToWrite;
save_nii(niiFallback, niiPath);

if isGz
    gzip_and_validate(niiPath, filename);
end
end

function imgOut = normalize_img_type(imgIn)
if isinteger(imgIn)
    imgOut = imgIn;
elseif islogical(imgIn)
    imgOut = uint8(imgIn);
else
    imgOut = single(imgIn);
end
end

function ensure_nifti_writers_on_path()
% Best-effort path bootstrap for cluster jobs where toolbox addpath may be incomplete.
if exist('make_nii', 'file') == 2 && exist('save_nii', 'file') == 2
    return;
end

candidateRoots = {};

% 1) Resolve toolbox path relative to this function file.
thisDir = fileparts(mfilename('fullpath'));            % .../matlab/functions
matlabDir = fileparts(thisDir);                        % .../matlab
candidateRoots{end+1} = fullfile(matlabDir, 'toolboxes');

% 2) Optional override from environment, useful for cluster setups.
envRoot = getenv('MREPT_TOOLBOX_ROOT');
if ~isempty(envRoot)
    candidateRoots{end+1} = envRoot;
end

for i = 1:numel(candidateRoots)
    root = candidateRoots{i};
    if ~isfolder(root)
        continue;
    end

    niftiDir = fullfile(root, 'NIfTI_20140122');
    if isfolder(niftiDir)
        addpath(genpath(niftiDir));
    end

    dicmDir = fullfile(root, 'dicm2nii');
    if isfolder(dicmDir)
        addpath(genpath(dicmDir));
    end
end
end

function gzip_and_validate(niiPath, targetFilename)
gzDir = fileparts(niiPath);
if isempty(gzDir)
    gzDir = pwd;
end
gzFiles = gzip(niiPath, gzDir);
if ~strcmp(gzFiles{1}, targetFilename)
    movefile(gzFiles{1}, targetFilename, 'f');
end
delete(niiPath);
assert_gzip_file(targetFilename);
end

function assert_gzip_file(path)
fid = fopen(path, 'r');
if fid == -1
    error('save_nii_gz:OutputMissing', 'Output file not found: %s', path);
end
bytes = fread(fid, 2, 'uint8=>uint8');
fclose(fid);
if numel(bytes) < 2 || bytes(1) ~= uint8(31) || bytes(2) ~= uint8(139)
    error('save_nii_gz:InvalidGzipOutput', ...
        'Expected gzip file but got invalid header for: %s', path);
end
end
