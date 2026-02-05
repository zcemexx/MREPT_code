function [conductivity, varargout] = conductivityMapping(txPhase, Mask, Parameters, options)
arguments
    txPhase {mustBeNumeric} % 2D or 3D (estimated) TRANSMIT phase (= transceive phase/2), i.e. φ0/2 in [radian] (2D/3D double/single array)
    Mask {mustBeNumericOrLogical} % brain mask (non-brain must be 0), same size as txPhase
    Parameters
    % Parameters must be a struct array with following fields:
    %           .B0 (scalar) {mustBePositive} % external magnetic field strength in [Tesla]
    %           .VoxelSize (1x3 vector) {mustBePositive} % 3d [x,y,z] voxel size (resolution) in [milli-meter]
    %
    % EPT kernel size(s) in [x,y,z] directions must be defined as one of the following fields:
    %           .kDiffSize (1x3 vector) {mustBePositive} % kernel size (diameter) for differentiation in [voxel]
    %           .kDiffRadius (1x3 vector) {mustBePositive} % kernel radius for differentiation in [milli-meter]
    %
    % To use integral-form EPT, size of integration kernel must be additionally defined as one of the following fields:
    %           .kIntegralSize (1x3 vector) {mustBePositive} % maximum kernel size (diameter) in [x,y,z] for surface integral in [voxel]
    %           .kIntegralRadius (1x3 vector) {mustBePositive} % maximum kernel radius in [x,y,z] for surface integral in [milli-meter]
    %
    % For 2D EPT, kernel size(s) must be defined as [x,y,1].

    % Optional inputs
    options.segmentation {mustBeNumericOrLogical} % tissue segmentations (labels), non-brain must be 0, same size as txPhase
    options.magnitude {mustBeNumeric} % magnitude image, same size as txPhase
    options.noise {mustBeNumeric} % phase noisemap, same size as txPhase, or explicit magnitude noise level (scalar). This controls the magnitude weight of polynomial fitting, and the smoothness of the output conductivity map.
    options.estimatenoise {mustBeNumericOrLogical} = false; % estimate noise level from the magnitude image

    % Post-processing filtering option
    options.isfilter = true; % replace unphysiological conductivity (<0 or >10 S/m) by 3D interpolation and output filtered conductivity map(s)

end
tic
nargoutchk(1,3)

isfilter = options.isfilter; options = rmfield(options, 'isfilter');
estimatenoise = options.estimatenoise; options = rmfield(options, 'estimatenoise');

% ================= check required parameters =================
if ~isfield(Parameters, 'B0'); error('Please provide external magnetic field strength B0.'); end
if ~isfield(Parameters, 'VoxelSize'); error('Please provide voxel size (resolution).'); end

if isfield(Parameters, 'kDiffSize')
    if any( mod(Parameters.kDiffSize,2) == 0 )
        warning('Kernel dimension(s) with even size will be increased by 1 voxel.')
        Parameters.kDiffSize = 2 * floor(Parameters.kDiffSize./2) + 1;
    end
    Parameters.kDiffRadius = (Parameters.kDiffSize .* Parameters.VoxelSize) ./ 2;
elseif isfield(Parameters, 'kDiffRadius')
    Parameters.kDiffSize = 2 * floor(Parameters.kDiffRadius ./ Parameters.VoxelSize) + 1;
    Parameters.kDiffRadius = (Parameters.kDiffSize .* Parameters.VoxelSize) ./ 2;
else
    error('Please provide kernel size (diameter in voxel) or kernel radius (in mm) for differentiation.')
end

if isfield(Parameters, 'kIntegralSize') && ~isempty(Parameters.kIntegralSize)
    if any( mod(Parameters.kIntegralSize,2) == 0 )
        warning('Kernel dimension(s) with even size will be increased by 1 voxel.')
        Parameters.kIntegralSize = 2 * floor(Parameters.kIntegralSize./2) + 1;
    end
    Parameters.kIntegralRadius = (Parameters.kIntegralSize .* Parameters.VoxelSize) ./ 2;
    form = 'integral';
elseif isfield(Parameters, 'kIntegralRadius') && ~isempty(Parameters.kIntegralRadius)
    Parameters.kIntegralSize = 2 * floor(Parameters.kIntegralRadius ./ Parameters.VoxelSize) + 1;
    Parameters.kIntegralRadius = (Parameters.kIntegralSize .* Parameters.VoxelSize) ./ 2;
    form = 'integral';
else
    warning('No integration kernel size provided, running Laplacian-form EPT.')
    form = 'laplacian';
end

% ================= check sizes and additional inputs =================
if isequal(size(txPhase), size(Mask))
    txPhase = double(txPhase .* Mask);
else
    error('Phase and mask must have the same size.')
end

if isfield(options, 'magnitude') && isfield(options, 'segmentation')
    if isequal(size(txPhase), size(options.magnitude), size(options.segmentation))
        disp('Magnitude and segmentation provided, using Mag+Seg EPT.')
    else
        error('Phase, magnitude and segmentations must have the same size.')
    end
    options.magnitude = double(options.magnitude) .* double(Mask);
    options.segmentation = double(options.segmentation) .* double(Mask);
elseif isfield(options, 'magnitude')
    if isequal(size(txPhase), size(options.magnitude))
        disp('Magnitude provided, using Mag EPT.')
    else
        error('Phase and magnitude must have the same size.')
    end
    options.magnitude = double(options.magnitude) .* double(Mask);
    options.segmentation = double(Mask);
elseif isfield(options, 'segmentation')
    if isequal(size(txPhase), size(options.segmentation))
        disp('Segmentation provided, using Seg EPT.')
    else
        error('Phase and segmentations must have the same size.')
    end
    options.segmentation = double(options.segmentation) .* double(Mask);
else
    disp('No additional input provided, using ellipsoid EPT.')
    options.segmentation = double(Mask);
end

if isfield(options, 'noise')
    if isscalar(options.noise)
        options.delta = double(options.noise .* Mask);
    elseif isequal(size(txPhase), size(options.noise))
        if isfield(options, 'segmentation')
            warning(['Voxelwise noisemap may not further improve the results when segmentation is provided.', ...
                newline, 'This feature will be removed from future release.'])
        end
        magNoise = double(1 ./ options.noise .* Mask);
        magNoise( isnan(magNoise) | isinf(magNoise) ) = 0;
        magNoise = rescale(magNoise, 0, 1, ...
            'InputMin', 0, 'InputMax', prctile(magNoise(:),99));
        options.delta = magNoise;
        clear magNoise
    else
        error('Noise must ba a scalar or have the same size as phase.')
    end
    options = rmfield(options, 'noise');
elseif isfield(options, 'magnitude') && estimatenoise
    disp('Estimating magnitude noise level from normalised magnitude image.')
    magNoise = double(options.magnitude .* Mask);
    magNoise = rescale(magNoise, 0, 1, ...
        'InputMin', 0, 'InputMax', prctile(magNoise(:),99));
    options.delta = magNoise;
    clear magNoise
end

% ================= set constants =================
gamma = 42.577478518 * 2 * pi; % gyromagnetic ratio of proton in [rad/(s*T)]
miu0 = 4 * pi * 1e-7; % magnetic vacuum permeability in [H/m]

% ================= electrical property tomography =================
settings = namedargs2cell(options);
% calculate conductivity
switch form
    case 'laplacian'
        % estimate 2nd order derivatives
        [der2, ~] = weightedDifferentiation(txPhase, ...
            Parameters.VoxelSize, Parameters.kDiffSize, settings{:});

    case 'integral'
        % estimate 1st- and 2nd-order derivatives
        [der2, der1] = weightedDifferentiation(txPhase, ...
            Parameters.VoxelSize, Parameters.kDiffSize, settings{:});

        % calculate surface integral
        surfaceIntegral = weightedSurfaceIntegral(der1.x, der1.y, der1.z, ...
            Parameters.VoxelSize, Parameters.kIntegralSize, settings{:});
end

if exist('surfaceIntegral', 'var')
    conductivity = surfaceIntegral ./ (gamma * Parameters.B0 * miu0);
else
    conductivity = der2 ./ (gamma * Parameters.B0 * miu0);
end

% find unphysiological conductivities (S/m)
unphysioMap = (conductivity < 0) | (conductivity > 10);
if isfilter
    conductivity = fillmissing3d(conductivity, ...
        'missingMap', unphysioMap, 'interpMask', Mask);
end

t = toc;
% ================= additional outputs =================
if nargout > 1
    % EPT settings
    options.Parameters = Parameters; % effective parameters
    options.unphysioMap = unphysioMap;
    options.isFilter = isfilter;

    if exist('surfaceIntegral', 'var')
        % pass on integral-form EPT intermediate results
        options.der1 = der1;
        options.der2ept.img = der2 ./ (gamma * Parameters.B0 * miu0);

        % find unphysiological conductivities (S/m)
        unphysioMap = (options.der2ept.img < 0) | (options.der2ept.img > 10);
        options.der2ept.unphysioMap = unphysioMap;
        if isfilter
            options.der2ept.img = fillmissing3d(options.der2ept.img, ...
                'missingMap', unphysioMap, 'interpMask', Mask);
        end
    end

    options.runtime = string(duration(seconds(t), 'Format', 'hh:mm:ss'));
    varargout{1} = options;
end

end
