% [更新] 通用绘图函数：文件名增加 SNR
function plot_metric_view(rad, mae, rmse, ssim_map, center, dim, viewName, mainTitle, caseName, snr_val, saveDir)
    % dim: 1=Sagittal, 2=Coronal, 3=Axial
    
    slice_rad  = squeeze_slice(rad, dim, center);
    slice_mae  = squeeze_slice(mae, dim, center);
    slice_rmse = squeeze_slice(rmse, dim, center);
    slice_ssim = squeeze_slice(ssim_map, dim, center);
    
    h = figure('Visible', 'off'); 
    set(h, 'Position', [100, 100, 1200, 900]); 
    
    % 1. Optimal Kernel Radius
    subplot(2,2,1); 
    imagesc(slice_rad); axis image off; colormap(gca, 'parula'); colorbar;
    title('Optimal Kernel Radius');
    
    % 2. MAE Map
    subplot(2,2,2); 
    imagesc(slice_mae); axis image off; colormap(gca, 'hot'); colorbar;
    title('MAE Map (Abs Error)');
    
    % 3. RMSE Map
    subplot(2,2,3); 
    imagesc(slice_rmse); axis image off; colormap(gca, 'hot'); colorbar;
    title('RMSE Map (Squared Error)');
    
    % 4. SSIM Map
    subplot(2,2,4); 
    imagesc(slice_ssim); axis image off; colormap(gca, 'gray'); colorbar;
    title('Local SSIM Map'); 
    
    sgtitle({[viewName, ' View'], mainTitle});
    
    % 文件名：Case_SNR_View.png
    fName = sprintf('%s_SNR%03d_%s.png', caseName, round(snr_val), viewName);
    saveas(h, fullfile(saveDir, fName));
    close(h);
end