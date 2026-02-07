%% Demo script to run conductivity mapping using UCL MR-EPT v2.2
% This MATLAB package is a major re-write of the original UCL QCM algorithm
% that reconstructs electrical conductivity map from MR transmit/transceive
% (unwrapped) phase acquired from ANY MRI pulse sequences, in theory.
% 
% This UCL MR-EPT (v2.2) package supports different phase-based methods 
% that CONCEPTIONALLY consistent with the initial work of:
% [Karsa A and Shmueli K. 2021. ISMRM. Abstract 3774].
% Nevertheless, this UCL MR-EPT (v2.2) should produce significantly
% improved conductivity maps and greater consistency across various MRI
% pulse sequences, compared with the original QCM implementation, thanks to 
% the following new features (mainly):
%
% - Accelerated reconstruction
% - Improved segmentations & magnitude normalisation for data with low magnitude contrasts
% - Post-reconstruction filtering for unphysiological conductivities (<0 S/m or > 10 S/m)
% 
%
% The package has been tested on MATLAB 2022b, and should be compatible to
% later MATLAB versions. Please report any issue to jierong.luo@ucl.ac.uk
%
% Last update: 
% Jierong Luo 
% 31st-Aug-2025
% University College London
%

%% ===================== Dependencies and paths =====================
addpath(genpath('functions'))
addpath(genpath('toolboxes'))

%% ===================== Load data =====================
dataDir = fullfile(pwd, 'data', '3T_Brain_phantom');

% 1) Transceive phase (phi0): measured or simulated B1+ phase
phi0_nii = nii_tool('load', fullfile(dataDir, 'Transceive_phase_noiseless.nii.gz'));
phi0 = phi0_nii.img;   % 直接用相位（必要时可除以 2）
b1pulse_phase = phi0 / 2;

% 2) Mask：从 segmentation 自动生成基本 mask
seg_nii = nii_tool('load', fullfile(dataDir, 'Tissue_Segmentation.nii.gz'));
segmentation = seg_nii.img;
mask = segmentation > 0;       % 非 0 都作为脑区掩膜

% 3) Magnitude（可以任选 T1w 或 T2w）
mag_nii = nii_tool('load', fullfile(dataDir, 'T1w_noiseless.nii.gz'));
magnitude = mag_nii.img;

% 4) Noisemap（你目前的 demo 是 noiseless，所以先不用）
noisemap = []; % 暂无噪声图

%% ===================== Scan parameters =====================
parameters.B0 = 3; % external magnetic field strength in [Tesla]
parameters.VoxelSize = [1 1 1]; % 3D [x,y,z] voxel size (resolution) in [MILLI-meter]
parameters.kDiffSize = [7 7 7]; % 3D [x,y,z] kernel size (diameter) for differentiation in [VOXEL]
parameters.kIntegralSize = [13 13 13]; % 3D [x,y,z] kernel size (diameter) for integration in [VOXEL]
% Alternatively, provide kernel size(s) in radius [MILLI-meter] in
% following fields:
% parameters.kDiffRadius = dummy_differentiation_kernel_radius;
% parameters.kIntegralRadius = dummy_integration_kernel_radius;

% %% ===================== EPT 2.2 =====================
% % This section provides some examples to call different EPT reconstruction methods 
% 
% % >>>>>>>>>>> Integral-form EPT with ellipsoid kernels
% conductivity = conductivityMapping(b1plus_phase, mask, parameters);
% 
% % >>>>>>>>>>> Integral-form  Mag EPT with explicit magnitude noise level
% conductivity = conductivityMapping(b1plus_phase, mask, parameters, ...
%     'magnitude', magnitude, 'noisemap', 0.5);
% 
% % >>>>>>>>>>> Integral-form  Mag EPT with automatic magnitude weighting
% % using phase noisemap
% [conductivity, options] = conductivityMapping(b1plus_phase, mask, parameters, ...
%     'magnitude', magnitude, 'noisemap', noisemap);
% 
% % >>>>>>>>>>> Integral-form  Mag EPT with automatic magnitude weighting
% % using estimated noise level from the data
% [conductivity, options] = conductivityMapping(b1plus_phase, mask, parameters, ...
%     'magnitude', magnitude, 'estimatenoise', true);
% 
% % >>>>>>>>>>> Integral-form Seg EPT
% conductivity = conductivityMapping(b1plus_phase, mask, parameters, ...
%     'segmentation', segmentation);
% 
% % >>>>>>>>>>> Laplacian-form Mag+Seg EPT
% % Laplacian-form EPT is evoked with the same function call, automatically,
% % when no integral kernel information is provided as a parameter, i.e. 
% % parameters.kIntegralSize and parameters.kIntegralRadius do not exist or
% % left empty:
% % parameters.kIntegralSize = [];
% % parameters.kIntegralRadius = [];
% [conductivity, options] = conductivityMapping(b1plus_phase, mask, parameters, ...
%     'magnitude', magnitude, 'segmentation', segmentation);
% 
% % >>>>>>>>>>> Integral-form Mag+Seg EPT
% [conductivity, options] = conductivityMapping(b1plus_phase, mask, parameters, ...
%     'magnitude', magnitude, 'segmentation', segmentation);
% 
% % >>>>>>>>>>> Integral-form Mag+Seg EPT (RECOMMENDED)
% [conductivity, options] = conductivityMapping(b1plus_phase, mask, parameters, ...
%     'magnitude', magnitude, 'segmentation', segmentation);
% 
% % >>>>>>>>>>> Integral-form Mag+Seg EPT without post-prcess filtering
% [conductivity, options] = conductivityMapping(b1plus_phase, mask, parameters, ...
%     'magnitude', magnitude, 'segmentation', segmentation, 'isfilter', false);
% 
% % >>>>>>>>>>> Integral-form Mag+Seg EPT with automatic magnitude weighting
% % Voxelwise noisemap may be unnecessary for Mag+Seg methods and will be
% % removed from future release
% [conductivity, options] = conductivityMapping(b1plus_phase, mask, parameters, ...
%     'magnitude', magnitude, 'segmentation', segmentation, 'noisemap', noisemap);

[conductivity, options] = conductivityMapping(b1pulse_phase, mask, parameters, ...
    'magnitude', magnitude, 'segmentation', segmentation);

%% ===================== MR-EPT outputs =====================
% save conductivity map as nifti
temp_nii.img = conductivity;

% OPTIONAL output 'options' is a structure that contains paramters and
% reconstruction methods used for MR-EPT.
% It has following fields:
%    .segmentation
%    .magnitude
%    .noise (if provided)
%    .Parameters   
%    .unphysioMap 
%    .isFilter
%    .runtime
%
% For integral-form EPT, additional fields include intermediate results:
%    .der1
%    .der2ept.img
%    .der2ept.unphysioMap
save(fullfile(saveDir, 'ept_results.mat'), 'options')

