% ... (以下函数保持不变: gen_noise_complex, masked_smooth_error, refine_labels_mad) ...
function noisy_complex = gen_noise_complex(vals, SNR, mask)
    signal_level = mean(abs(vals(mask)), 'all');
    sigma_noise = signal_level / SNR;
    noise_real = randn(size(vals), 'single') * sigma_noise;
    noise_imag = randn(size(vals), 'single') * sigma_noise;
    noisy_complex = vals + (noise_real + 1i * noise_imag);
end