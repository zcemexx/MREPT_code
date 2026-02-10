%% Replot MAE/RMSE/SSIM figures with global metric scales across all cases
clear; clc;

addpath('/myriadfs/home/zcemexx/projects/MREPT_code/matlab/functions');

projectRoot  = '/myriadfs/home/zcemexx/Scratch';
datasetName  = 'Dataset001_EPT';
baseDir      = fullfile(projectRoot, 'nnUNet_raw');
metricsDir   = fullfile(baseDir, 'metrics');
figuresDir   = fullfile(metricsDir, 'figures');
plotCacheDir = fullfile(metricsDir, 'plot_cache');

if ~isfolder(plotCacheDir)
    error('plot_cache directory not found: %s', plotCacheDir);
end
if ~isfolder(figuresDir)
    mkdir(figuresDir);
end

cacheFiles = dir(fullfile(plotCacheDir, 'plotcache_M*.mat'));
if isempty(cacheFiles)
    error('No plot cache files found in: %s', plotCacheDir);
end

globalLimits = [inf, -inf; inf, -inf; inf, -inf];
cacheList = cell(numel(cacheFiles), 1);

for i = 1:numel(cacheFiles)
    fpath = fullfile(cacheFiles(i).folder, cacheFiles(i).name);
    S = load(fpath, 'plotCache');
    if ~isfield(S, 'plotCache')
        error('Invalid cache file (missing plotCache): %s', fpath);
    end
    C = S.plotCache;
    cacheList{i} = C;

    if isfield(C, 'metricLimitsLocal') && isequal(size(C.metricLimitsLocal), [3, 2])
        lim = double(C.metricLimitsLocal);
    else
        lim = local_metric_limits_from_views(C.views);
    end

    globalLimits(:,1) = min(globalLimits(:,1), lim(:,1));
    globalLimits(:,2) = max(globalLimits(:,2), lim(:,2));
end

for k = 1:3
    if ~all(isfinite(globalLimits(k,:)))
        globalLimits(k,:) = [0, 1];
    elseif globalLimits(k,1) == globalLimits(k,2)
        globalLimits(k,2) = globalLimits(k,1) + 1e-6;
    end
end

save(fullfile(metricsDir, 'global_metric_limits.mat'), 'globalLimits');
fprintf('Global limits saved: %s\n', fullfile(metricsDir, 'global_metric_limits.mat'));
fprintf('MAE  limits: [%.6f, %.6f]\n', globalLimits(1,1), globalLimits(1,2));
fprintf('RMSE limits: [%.6f, %.6f]\n', globalLimits(2,1), globalLimits(2,2));
fprintf('SSIM limits: [%.6f, %.6f]\n', globalLimits(3,1), globalLimits(3,2));

for i = 1:numel(cacheList)
    C = cacheList{i};
    for iv = 1:numel(C.views)
        plot_metric_view_slices(C.views(iv), C.title, C.caseName, C.snr, figuresDir, globalLimits);
    end
end

fprintf('Replot done for %d cases.\n', numel(cacheList));

function lim = local_metric_limits_from_views(views)
    mae_vals = [];
    rmse_vals = [];
    ssim_vals = [];
    for j = 1:numel(views)
        mae_vals = [mae_vals; views(j).mae(:)]; %#ok<AGROW>
        rmse_vals = [rmse_vals; views(j).rmse(:)]; %#ok<AGROW>
        ssim_vals = [ssim_vals; views(j).ssim(:)]; %#ok<AGROW>
    end
    lim = [
        safe_minmax(mae_vals);
        safe_minmax(rmse_vals);
        safe_minmax(ssim_vals)
    ];
end

function mm = safe_minmax(x)
    x = x(isfinite(x));
    if isempty(x)
        mm = [0, 1];
    else
        mm = [double(min(x)), double(max(x))];
    end
end
