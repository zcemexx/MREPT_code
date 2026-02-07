function [der2, der1, varargout] = weightedDifferentiation(txPhase, voxelSize, kernelSize, options)
% Function to calculate 1st (gradient) and 2nd (Laplacian) differentiation
% of MR transive phase w/o weighted polynomial fitting, within shaped kernels.

arguments
    txPhase % transmit phase φ0/2 in radius
    voxelSize % 3d [x,y,z] voxel size in mm
    kernelSize % maximum kernel size (diameter) in x-, y-, and z-directions in voxel

    options.segmentation {mustBeNumericOrLogical} = ones(size(txPhase)) % segmentation or brain mask with non-brain = 0s
    options.magnitude {mustBeNumeric} = ones(size(txPhase))
    options.delta {mustBeNumeric} = ones(size(txPhase))
    options.alpha {mustBeNumeric} = 0.5
    options.verbose {mustBeNumericOrLogical} = false
end
nargoutchk(2,3)

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

% define kernel using cartisian grid
k_x = -kernelRadius(1):kernelRadius(1);
k_y = -kernelRadius(2):kernelRadius(2);
k_z = -kernelRadius(3):kernelRadius(3);
[k_x, k_y, k_z] = ndgrid(k_x.*voxelSize(1), k_y.*voxelSize(2), k_z.*voxelSize(3));

% ellipsoid shape within kernel grid
ellips = ( k_x.^2 / (k_r(1)*voxelSize(1))^2 + ...
    k_y.^2 / (k_r(2)*voxelSize(2))^2 + ...
    k_z.^2 / (k_r(3)*voxelSize(3))^2 ) <= 1;

k_x = k_x(:); k_y = k_y(:); k_z = k_z(:); % vectorisation

% define polynomial fitting for kernel coefficients
F = [ones(numel(k_x), 1), k_x.^2, k_y.^2, k_z.^2, ...
    k_x, k_y, k_z, k_x.*k_y, k_y.*k_z, k_z.*k_x];

% zero-pad inputs
txPhase = padarray(txPhase, kernelRadius, 0, 'both');
mag = padarray(options.magnitude, kernelRadius, 0, 'both');
seg = padarray(options.segmentation, kernelRadius, 0, 'both');
deltamap = padarray(options.delta, kernelRadius, 0, 'both');

% initialisation
im_sz = size(txPhase);
der2 = zeros(prod(im_sz), 1);
der1 = struct;
der1.x = zeros(prod(im_sz), 1);
der1.y = zeros(prod(im_sz), 1);
der1.z = zeros(prod(im_sz), 1);

% calculate derivatives within mask or each tissue type
segments = unique(seg);
segments = segments(segments ~= 0);
for tissue = 1:numel(segments)
    if options.verbose
        disp(['Normalising magnitude in ' num2str(tissue) ...
            ' of ' num2str(numel(segments)) ' tissue type(s).'])
    end

    tissueType = (seg == segments(tissue));

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

    % refine mask/segmentation for polynomial fitting
    tissueType = refineSegment(double(tissueType));

    % calculate derivatives within mask or ROI(s) of each tissue type
    if options.verbose
        disp(['Calculating derivatives in ' num2str(numel(unique(tissueType))-1) ' ROI(s).'])
    end

    idx = find(tissueType);

    % initialise
    nIdx = numel(idx);
    coef_sg2 = zeros(nIdx,1);
    coef_sg1_x = zeros(nIdx, 1);
    coef_sg1_y = zeros(nIdx, 1);
    coef_sg1_z = zeros(nIdx, 1);

    parfor n = 1:nIdx
        [ix, iy, iz] = ind2sub(im_sz, idx(n));
        win_x = ix-kernelRadius(1):ix+kernelRadius(1);
        win_y = iy-kernelRadius(2):iy+kernelRadius(2);
        win_z = iz-kernelRadius(3):iz+kernelRadius(3);

        % noise weight
        delta = delta_tissue(ix,iy,iz);

        % additional magnitude weight
        w = exp(-(mag_tissue(win_x,win_y,win_z) - mag_tissue(ix,iy,iz)).^2 / (2 * delta^2));
        w(isnan(w)|isinf(w)) = 0;

        % local φ0
        b = txPhase(win_x,win_y,win_z);

        % find local roi mask
        roi = ( tissueType(win_x, win_y, win_z) == tissueType(ix, iy, iz) );

        % shape kernel
        kernel = (ellips & roi);

        % weight within shape
        w = w(kernel); w = w(:);
        w = repmat(w, [1 size(F, 2)]);

        % local φ0 within shape
        b = b(kernel);
        b = b(:) .* w(:,1);

        % coefficient matrix
        A = F(kernel(:), :) .* w;

        % least-squares polynomial fitting
        x = lsqminnorm(A, b);

        % laplacian
        coef_sg2(n) = (x(2) + x(3) + x(4)) .* 2;

        % gradient
        coef_sg1_x(n) = x(5);
        coef_sg1_y(n) = x(6);
        coef_sg1_z(n) = x(7);
    end

    % assign 2nd and 1st derivatives
    der2(idx) = coef_sg2;
    der1.x(idx) = coef_sg1_x;
    der1.y(idx) = coef_sg1_y;
    der1.z(idx) = coef_sg1_z;

end

% reshape outputs
der2 = reshape(der2, im_sz);
der1.x = reshape(der1.x, im_sz);
der1.y = reshape(der1.y, im_sz);
der1.z = reshape(der1.z, im_sz);

% crop zero pads
der2 = der2(kernelRadius(1)+1:end-kernelRadius(1),...
    kernelRadius(2)+1:end-kernelRadius(2),...
    kernelRadius(3)+1:end-kernelRadius(3));
der1.x = der1.x(kernelRadius(1)+1:end-kernelRadius(1),...
    kernelRadius(2)+1:end-kernelRadius(2),...
    kernelRadius(3)+1:end-kernelRadius(3));
der1.y = der1.y(kernelRadius(1)+1:end-kernelRadius(1),...
    kernelRadius(2)+1:end-kernelRadius(2),...
    kernelRadius(3)+1:end-kernelRadius(3));
der1.z = der1.z(kernelRadius(1)+1:end-kernelRadius(1),...
    kernelRadius(2)+1:end-kernelRadius(2),...
    kernelRadius(3)+1:end-kernelRadius(3));

% find uncertainty map and remove invalid values
invalidMap1 = ismissing(der1.x .* der1.y .* der1.z);
der1.x(invalidMap1) = 0;
der1.y(invalidMap1) = 0;
der1.z(invalidMap1) = 0;
der2 = fillmissing3d(der2);

if nargout > 2
    options.segmentation = seg(kernelRadius(1)+1:end-kernelRadius(1),...
        kernelRadius(2)+1:end-kernelRadius(2),...
        kernelRadius(3)+1:end-kernelRadius(3));
    varargout{1} = options;
end

end


