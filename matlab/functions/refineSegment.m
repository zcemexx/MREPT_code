function segmentOut = refineSegment(segmentIn)

% Separate the label map into connected regions
labels = unique(double(segmentIn));
labels(labels == 0) = [];

% initialise
maxLabel = 0;
segmentOut = zeros(size(segmentIn));

for segmentNo = 1:numel(labels)
    segmentRefined = zeros(size(segmentIn));

    segment = (segmentIn == labels(segmentNo));
    connComp = bwconncomp(segment);
    compProp = regionprops3(connComp, 'Volume');

    % segregate large ROIs into smaller components by connectivity
    if any(compProp.Volume >= 10)
        largeROIs = find(compProp.Volume >= 10);
        for n = 1:numel(largeROIs)
            idx = connComp.PixelIdxList{1,largeROIs(n)};
            segmentRefined(idx) = maxLabel + 1;
            maxLabel = max(segmentRefined(:));
        end                              
    end

    % assign small ROIs to the closest segment
    if any(compProp.Volume < 10)
        smallROIs = find(compProp.Volume < 10);

        for roiNo = 1:numel(smallROIs)
            idx = connComp.PixelIdxList{1,smallROIs(roiNo)};
            smallSegment = logical(segment .* 0);
            smallSegment(idx) = true;

            % calculate distance
            D = bwdist(smallSegment);
            D(smallSegment) = NaN; D(segmentRefined == 0) = NaN;
            
            roiLabel = segmentRefined(D == min(D, [], 'all', 'omitnan'));
            segmentRefined(idx) = roiLabel(1);
        end
    end
    segmentOut = segmentOut + segmentRefined;
    
end

end