function [surfIntegral, varargout] = weightedSurfaceIntegral(xGradient, yGradient, zGradient, voxelSize, kernelSize, options)

arguments
    xGradient
    yGradient
    zGradient
    voxelSize % 3d [x,y,z] voxel size in mm
    kernelSize % maximum kernel size (diameter) in x-, y-, and z-directions in voxel

    options.segmentation {mustBeNumericOrLogical} = ones(size(xGradient)) % segmentation or brain mask with non-brain = 0s
    options.magnitude {mustBeNumeric} = ones(size(xGradient))
    options.delta {mustBeNumeric} = ones(size(xGradient))
    options.alpha {mustBeNumeric} = 0.5
    options.verbose {mustBeNumericOrLogical} = false
end
nargoutchk(1,2)

% expected gx gy gz size equal
if ~isequal(size(xGradient), size(yGradient), size(zGradient))
    error('x-, y-, and z-gradient must have same size')
end

% calculate kernel radius for multi-dimensional grid of ouput kernel
kernelRadius = floor(kernelSize/2);

% check kernel dimension
k_dim = nnz(kernelRadius ~= 0);
if k_dim == 3
    k_r = kernelRadius;
elseif k_dim == 2
    k_r = kernelRadius;
    k_r(k_r == 0) = 1; % prevent NaNs in division for 3rd dimension
else
    error('Only 2d or 3d kernel can be accepted')
end

% define Savitky-Golay Laplacian or gradient kernel
% cartisian index
k_x = -kernelRadius(1):kernelRadius(1);
k_y = -kernelRadius(2):kernelRadius(2);
k_z = -kernelRadius(3):kernelRadius(3);
[k_x, k_y, k_z] = ndgrid(k_x.*voxelSize(1), k_y.*voxelSize(2), k_z.*voxelSize(3));

% ellipsoid kernel
ellips = ( k_x.^2/(k_r(1)*voxelSize(1))^2 + ...
    k_y.^2/(k_r(2)*voxelSize(2))^2 + ...
    k_z.^2/(k_r(3)*voxelSize(3))^2 ) <= 1;

% zero-pad inputs
xGradient = padarray(xGradient, kernelRadius, 0, 'both');
yGradient = padarray(yGradient, kernelRadius, 0, 'both');
zGradient = padarray(zGradient, kernelRadius, 0, 'both');
seg = padarray(options.segmentation, kernelRadius, 0, 'both');
mag = padarray(options.magnitude, kernelRadius, 0, 'both');
deltamap = padarray(options.delta, kernelRadius, 0, 'both');

% initialisation
im_sz = size(xGradient);
surfIntegral = zeros(prod(im_sz), 1);

% calculate integrals within mask or each tissue types
segments = unique(seg);
segments = segments(segments ~= 0);
for tissue = 1:numel(segments)
    if options.verbose
        disp(['Normalising magnitude in ' num2str(tissue) ...
            ' of ' num2str(numel(segments)) ' tissue type(s).'])
    end

    tissueType = logical(seg == segments(tissue));
    % robust normalisation within each tissue types
    mag_tissue = mag(tissueType);
    mag_tissue = mag_tissue( (mag_tissue > prctile(mag_tissue(:),2)) & ...
        (mag_tissue < prctile(mag_tissue(:),98)) );
    mag_mean = mean(mag_tissue, 'all', 'omitnan');
    mag_std = std(mag_tissue, 1, 'all', 'omitnan');
    if (mag_std == 0) || isnan(mag_std)
        mag_tissue = mag;
    else
        mag_tissue = ( (mag - mag_mean) ./ mag_std );
    end
    mag_tissue(~tissueType) = Inf;

    % normalise deltamap
    delta_tissue = deltamap;
    if (numel(unique(deltamap(:))) > 2) && (numel(segments) > 1)
        delta_temp = deltamap(tissueType);
        delta_temp = rescale(delta_temp, options.alpha, 2-options.alpha, ...
            'InputMin', prctile(delta_temp(:), 2), ...
            'InputMax', prctile(delta_temp(:), 98));
        delta_tissue(tissueType) = delta_temp;
    end

    % refine mask/segmentation 
    tissueType = refineSegment(double(tissueType));

    % calculate integrals within mask or ROI(s) of each tissue type
    if options.verbose
        disp(['Calculating derivatives in ' num2str(numel(unique(tissueType))-1) ' ROI(s).'])
    end

    idx = find(tissueType);

    % initialise
    nIdx = numel(idx);
    coef_surf = zeros(nIdx,1);

    % calculate surface integral
    parfor n = 1:nIdx
        % local window
        [ix, iy, iz] = ind2sub(im_sz, idx(n));
        win_x = ix-kernelRadius(1):ix+kernelRadius(1);
        win_y = iy-kernelRadius(2):iy+kernelRadius(2);
        win_z = iz-kernelRadius(3):iz+kernelRadius(3);

        % find local gradient
        x_grad = xGradient(win_x, win_y, win_z);
        y_grad = yGradient(win_x, win_y, win_z);
        z_grad = zGradient(win_x, win_y, win_z);

        % find local roi mask
        roi = ( tissueType(win_x, win_y, win_z) == tissueType(ix, iy, iz) );

        % local noise level
        delta = delta_tissue(ix,iy,iz);

        % adaptive kernel shape
        kernel = (abs(mag_tissue(win_x,win_y,win_z) - mag_tissue(ix,iy,iz))) <= (sqrt(2*log(2)) * delta); % fwhm
        kernel = (kernel & ellips & roi);

        integral_nd = 0;
        switch k_dim
            case 3
                % calculate surface of the 3d shape and enclosed volume
                [k_surf, k_vol] = vol2surf(kernel, voxelSize);

                % calculate surface integral kernel
                x_surf = k_surf.x .* voxelSize(2) .* voxelSize(3) ./ k_vol;
                y_surf = k_surf.y .* voxelSize(1) .* voxelSize(3) ./ k_vol;
                z_surf = k_surf.z .* voxelSize(1) .* voxelSize(2) ./ k_vol;

                integral_nd = x_surf .* x_grad + y_surf .* y_grad + z_surf .* z_grad;

            case 2
                % calculate perimeter of the 2d shape and enclosed area
                [k_perim, k_area] = area2perim(kernel, voxelSize);

                % calculate surface integral kernel
                x_perim = k_perim.x .* voxelSize(2) ./ k_area;
                y_perim = k_perim.y .* voxelSize(1) ./ k_area;

                integral_nd = x_perim .* x_grad + y_perim .* y_grad;

        end
        coef_surf(n) = sum(integral_nd(:));
    end

    % assign values
    surfIntegral(idx) = coef_surf;
end

% reshape output and crop zero pad
surfIntegral = reshape(surfIntegral, im_sz);
surfIntegral = surfIntegral(kernelRadius(1)+1:end-kernelRadius(1),...
    kernelRadius(2)+1:end-kernelRadius(2),...
    kernelRadius(3)+1:end-kernelRadius(3));

% find uncertainty map and remove invalid values
[surfIntegral, invalidMap] = fillmissing3d(surfIntegral);

if nargout > 1
    options.invalidMap = invalidMap;
    varargout{1} = options;
end

end

%% Sub-functions
function [k_surf, k_vol] = vol2surf(shape, vol_sz)
% calculate surface (k_surf) and enclosed volume (k_vol) of
% a given 3d volume (shape)

k_surf = struct;
shape = double(shape);

% edge-finding kernel
k_edge = [-1; 0; 1];

% find (inner) boundaries of the shape
k_surf.x = convn(shape, k_edge, 'same') .* shape;
k_surf.y = convn(shape, permute(k_edge, [2 1 3]), 'same') .* shape;
k_surf.z = convn(shape, permute(k_edge, [3 2 1]), 'same') .* shape;

% volume kernel
k_dv = zeros(3,3,3);
k_dv([1 3],2,2) = vol_sz(2)*vol_sz(3);
k_dv(2,[1 3],2) = vol_sz(1)*vol_sz(3);
k_dv(2,2,[1 3]) = vol_sz(1)*vol_sz(2);
k_dv = k_dv ./ sum(k_dv(:));

% volume surrounded by the surface
k_vol = convn(shape, k_dv, 'same') .* shape;
k_vol = sum(k_vol(:)) * prod(vol_sz);

end

function [k_perim, k_area] = area2perim(shape, vol_sz)
% calculate perimiters (k_perim) and enclosed area (k_area) of
% a given 2d shape (x,y), assuming 1d along z-direction

k_perim = struct;
shape = double(shape);

% edge-finding kernel
k_edge = [-1; 0; 1];

% find (inner) boundaries of the shape
k_perim.x = convn(shape, k_edge, 'same') .* shape;
k_perim.y = convn(shape, permute(k_edge, [2 1 3]), 'same') .* shape;

% area kernel
k_da = zeros(3,3);
k_da([1 3],2) = vol_sz(2);
k_da(2,[1 3]) = vol_sz(1);
k_da = k_da ./ sum(k_da(:));

% area/surface surrounded by the perimeter
k_area = convn(shape, k_da, 'same') .* shape;
k_area = sum(k_area(:)) * vol_sz(1) * vol_sz(2);

end
