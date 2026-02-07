function [array3d, varargout] = fillmissing3d(array3d, options)
arguments
    array3d
    options.method {mustBeMember(options.method, {'linear', 'nearest'})} = 'linear'
    options.missingMap {mustBeNumericOrLogical} = (isnan(array3d) | isinf(array3d))
    options.interpMask {mustBeNumericOrLogical} = ones(size(array3d))
end

method = options.method;
missingMap = logical(options.missingMap);
interpMask = logical(options.interpMask);

if any(missingMap(:))
    % prepare masks for interpolation
    interpMask = interpMask & ~missingMap;

    % cartesian grid
    imSize = size(array3d);
    [x, y, z] = ndgrid(1:imSize(1),1:imSize(2),1:imSize(3));

    % interpolation
    array3d(missingMap) = griddata(x(interpMask), y(interpMask), z(interpMask), ...
        array3d(interpMask), x(missingMap), y(missingMap), z(missingMap), method);
end

if nargout > 1
    varargout{:} = missingMap;
end
end
