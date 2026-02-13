function save_nii_gz(img, filename)
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

% Primary writer: dicm2nii nii_tool
try
    nii = nii_tool('init', img);
    nii.hdr.pixdim(2:4) = [1 1 1];
    if isinteger(img)
        nii.img = img;
    elseif islogical(img)
        nii.img = uint8(img);
    else
        nii.img = single(img);
    end
    % Write .nii first, then gzip if requested.
    % Direct writes to .nii.gz can be invalid in some environments.
    nii_tool('save', nii, niiPath);
    if isGz
        gzDir = fileparts(niiPath);
        if isempty(gzDir)
            gzDir = pwd;
        end
        gzFiles = gzip(niiPath, gzDir);
        if ~strcmp(gzFiles{1}, filename)
            movefile(gzFiles{1}, filename, 'f');
        end
        delete(niiPath);
        assert_gzip_file(filename);
    end
    return;
catch primaryErr
    warning('save_nii_gz:PrimaryWriterFailed', ...
        'nii_tool save failed (%s). Trying save_nii fallback.', primaryErr.message);
end

% Fallback writer: NIfTI_20140122 make_nii/save_nii (+ gzip)
if exist('make_nii', 'file') ~= 2 || exist('save_nii', 'file') ~= 2
    error('save_nii_gz:NoFallbackWriter', ...
        ['Unable to save "%s": nii_tool failed and make_nii/save_nii are missing. ', ...
        'Add toolboxes/NIfTI_20140122 to path.'], filename);
end

niiFallback = make_nii(single(img));
if isinteger(img)
    niiFallback.img = img;
elseif islogical(img)
    niiFallback.img = uint8(img);
else
    niiFallback.img = single(img);
end

save_nii(niiFallback, niiPath);

if isGz
    gzDir = fileparts(niiPath);
    if isempty(gzDir)
        gzDir = pwd;
    end
    gzFiles = gzip(niiPath, gzDir);
    if ~strcmp(gzFiles{1}, filename)
        movefile(gzFiles{1}, filename, 'f');
    end
    delete(niiPath);
    assert_gzip_file(filename);
end
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
