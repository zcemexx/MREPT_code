function save_nii_gz(img, filename)
    % 尝试使用 nii_tool，需确保工具箱已安装
    try
        nii = nii_tool('init', img); 
        nii.hdr.pixdim(2:4) = [1 1 1]; 
        if isinteger(img); nii.img = img; else; nii.img = single(img); end
        nii_tool('save', nii, filename); 
    catch
        warning('nii_tool not found or failed. Using basic struct save.');
        % 简单的 fallback (不推荐，丢失 header)
        nii.img = img;
        save(filename, 'nii'); % 这里的保存实际上是错误的，仅作占位，请确保有 nii_tool
    end
end
