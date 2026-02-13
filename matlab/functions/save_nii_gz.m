function save_nii_gz(img, filename)
    outDir = fileparts(filename);
    if ~isempty(outDir) && ~isfolder(outDir)
        mkdir(outDir);
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
        nii_tool('save', nii, filename);
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
    end
end
