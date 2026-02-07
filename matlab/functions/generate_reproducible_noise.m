function [noisy_complex, snr_used] = generate_reproducible_noise(vals, mask, task_id, snr_range_linear)
% 输入:
%   vals: 原始复数信号
%   mask: 组织掩码
%   task_id: 用于复现的种子 (Slurm Array ID)
%   snr_range_linear: SNR 的随机范围, 例如 [10, 150]

% --- 1. 创建独立的随机数流 ---
% 这样可以确保这个函数内的随机操作不被外部干扰
s = RandStream('mt19937ar', 'Seed', task_id);

% --- 2. 复现 SNR 的选择 ---
% 注意：这里显式地使用了 rand(s, ...)，表示从 s 这个流里取数
snr_range_log = log10(snr_range_linear);
this_snr_log = snr_range_log(1) + (snr_range_log(2) - snr_range_log(1)) * rand(s, 1);
snr_used = 10^this_snr_log;

% --- 3. 计算噪声强度 ---
signal_level = mean(abs(vals(mask)), 'all');
sigma_noise = signal_level / snr_used;

% --- 4. 生成噪声 ---
% 同样显式使用 randn(s, ...)，确保噪声序列是由 task_id 唯一决定的
noise_real = randn(s, size(vals), 'single') * sigma_noise;
noise_imag = randn(s, size(vals), 'single') * sigma_noise;

noisy_complex = vals + (noise_real + 1i * noise_imag);
end