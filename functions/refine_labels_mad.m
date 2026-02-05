function radiusMap_out = refine_labels_mad(radiusMap_in, tissueMask, radius_list)
    radiusMap_out = radiusMap_in;
    [nx, ny, nz] = size(radiusMap_in);
    outlierMask = false(size(radiusMap_in));
    for t = 1:max(tissueMask(:))
        idx = (tissueMask == t);
        if ~any(idx(:)); continue; end
        vals = radiusMap_in(idx);
        m = median(vals);
        s = mad(vals, 1);
        outlierMask(idx & (abs(radiusMap_in - m) > 3*s)) = true;
    end
    [r_idx, c_idx, s_idx] = ind2sub(size(radiusMap_in), find(outlierMask));
    for i = 1:length(r_idx)
        cx = r_idx(i); cy = c_idx(i); cz = s_idx(i);
        ix = max(cx-1,1):min(cx+1,nx); iy = max(cy-1,1):min(cy+1,ny); iz = max(cz-1,1):min(cz+1,nz);
        winLabels = radiusMap_in(ix, iy, iz); winTissue = tissueMask(ix, iy, iz); winOutlier = outlierMask(ix, iy, iz);
        targetTissue = tissueMask(cx, cy, cz);
        validNeighbors = winLabels( (winTissue == targetTissue) & (~winOutlier) );
        if ~isempty(validNeighbors); radiusMap_out(cx, cy, cz) = mode(validNeighbors); end
    end
end