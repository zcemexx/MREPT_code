% 切片提取
function sl = squeeze_slice(vol, dim, center)
    if dim == 1
        sl = squeeze(vol(center(1), :, :));
    elseif dim == 2
        sl = squeeze(vol(:, center(2), :));
    else
        sl = squeeze(vol(:, :, center(3)));
    end
    sl = rot90(sl); 
end