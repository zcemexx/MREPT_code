% === Merge Script (Run ONLY after all jobs are finished) ===
metricsDir = '/myriadfs/home/zcemexx/Scratch/nnUNet_raw/metrics';
matFiles = dir(fullfile(metricsDir, 'metric_*.mat'));
logData = {};

for k = 1:numel(matFiles)
    d = load(fullfile(matFiles(k).folder, matFiles(k).name));
    s = d.metric_struct;
    logData(k,:) = {s.CaseID, s.SNR, s.MAE, s.RMSE, s.SSIM};
end

logTable = cell2table(logData, 'VariableNames', {'CaseID', 'SNR', 'MAE', 'RMSE', 'SSIM'});
writetable(logTable, fullfile(metricsDir, '..', 'dataset_metadata.csv'));
fprintf('合并完成，CSV 已生成。\n');

% 在这里调用你的 generate_dataset_json 函数
% generate_dataset_json(...)