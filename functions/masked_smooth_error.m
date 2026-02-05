function smoothed_err = masked_smooth_error(diff_map, tissueMask)
    [nx, ny, nz] = size(diff_map);
    smoothed_err = inf(nx, ny, nz, 'single');
    for t = 1:max(tissueMask(:))
        t_mask = (tissueMask == t);
        if ~any(t_mask(:)); continue; end
        current_diff = diff_map;
        current_diff(~t_mask) = 0; 
        win_sum_err = imboxfilt3(current_diff, 3) * 27;
        win_count   = imboxfilt3(single(t_mask), 3) * 27;
        valid_idx = (t_mask & win_count > 0);
        smoothed_err(valid_idx) = win_sum_err(valid_idx) ./ win_count(valid_idx);
    end
end