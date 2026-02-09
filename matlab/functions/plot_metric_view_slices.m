function plot_metric_view_slices(sliceData, mainTitle, caseName, snr_val, saveDir, metricLimits)
    h = figure('Visible', 'off');
    set(h, 'Position', [100, 100, 1200, 900]);

    subplot(2,2,1);
    imagesc(sliceData.radius); axis image off; colormap(gca, 'parula'); colorbar;
    title('Optimal Kernel Radius');

    subplot(2,2,2);
    imagesc(sliceData.mae); axis image off; colormap(gca, 'hot'); colorbar;
    caxis(metricLimits(1,:));
    title('MAE Map (Abs Error)');

    subplot(2,2,3);
    imagesc(sliceData.rmse); axis image off; colormap(gca, 'hot'); colorbar;
    caxis(metricLimits(2,:));
    title('RMSE Map (Squared Error)');

    subplot(2,2,4);
    imagesc(sliceData.ssim); axis image off; colormap(gca, 'gray'); colorbar;
    caxis(metricLimits(3,:));
    title('Local SSIM Map');

    sgtitle({[sliceData.name, ' View'], mainTitle});

    fName = sprintf('%s_SNR%03d_%s.png', caseName, round(snr_val), sliceData.name);
    saveas(h, fullfile(saveDir, fName));
    close(h);
end
