function generate_dataset_json(outputFolder, numTraining, description)
% GENERATE_DATASET_JSON 生成 nnU-Net 所需的 dataset.json 文件
% 适配配置：
%   Channel 00: Phase
%   Channel 01: Segmentation Mask
%   Labels: 0=Background, 1=1, 2=2... (Radius classes)
%   mask_channel_index 使用 Python 0-based 语义:
%   数值 1 表示第 2 个输入通道，也就是 *_0001 的组织 mask

    if nargin < 3
        description = 'EPT Radius Labeling Task';
    end

    jsonStruct = struct();
    
    % --- 1. 通道定义 (关键修改) ---
    % 这里定义了 CNN 的输入层结构
    % "0": "Phase" 对应 _0000.nii.gz
    % "1": "Mask"  对应 _0001.nii.gz
    jsonStruct.channel_names = struct('0', 'Phase', '1', 'Mask'); 
    
    % --- 2. 标签定义 ---
    % 定义输出类别 (Background + 30个半径)
    labels = struct();
    labels.background = 0;
    for i = 1:30
        labels.(['r' num2str(i)]) = i;
    end
    jsonStruct.labels = labels;
    
    % --- 3. 区域定义 (Region) ---
    % 新版 nnU-Net v2 需要 regions 定义，这里简单映射 class -> region
    % 如果是旧版 v1，这个字段可能会被忽略，但保留无害
    % regions = struct();
    % for i = 1:30
    %     regions.(['class_' num2str(i)]) = i;
    % end
    % jsonStruct.regions_class_order = (1:30);
    
    % --- 4. 元数据 ---
    jsonStruct.numTraining = numTraining;
    jsonStruct.file_ending = '.nii.gz';
    % Match current nnunetv2/imageio implementation in this repo.
    jsonStruct.overwrite_image_reader_writer = 'NibabelIO';
    jsonStruct.name = 'Dataset001_EPT';
    jsonStruct.reference = 'UCL EPT Project';
    jsonStruct.release = '1.0';
    jsonStruct.description = description;
    jsonStruct.regression_task = true;
    jsonStruct.kernel_radius_min = 1;
    jsonStruct.kernel_radius_max = 30;
    jsonStruct.mask_channel_index = 1;
    
    % --- 5. 写入文件 ---
    jsonPath = fullfile(outputFolder, 'dataset.json');
    
    % 使用 jsonencode 进行编码
    jsonText = jsonencode(jsonStruct, 'PrettyPrint', true);
    
    fid = fopen(jsonPath, 'w');
    if fid == -1
        error('无法创建 dataset.json 文件: %s', jsonPath);
    end
    fprintf(fid, '%s', jsonText);
    fclose(fid);
    
    fprintf('dataset.json 已生成: %s\n', jsonPath);
end
