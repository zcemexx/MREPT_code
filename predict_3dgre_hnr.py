#!/usr/bin/env python3
"""
nnUNet prediction script for 3DGRE_HNR case
"""
import os
import sys

# Set nnUNet environment paths
os.environ['nnUNet_raw'] = '/home/linux1917366562/MREPT_code/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/home/linux1917366562/MREPT_code/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/home/linux1917366562/MREPT_code/nnUNet_results'

sys.path.insert(0, '/home/linux1917366562/MREPT_code/nnunet')

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

# Input and output paths
input_dir = '/home/linux1917366562/MREPT_code/preds_local/inputs_3dgre_hnr'
output_dir = '/home/linux1917366562/MREPT_code/preds_local/3dgre_hnr'
model_dir = '/home/linux1917366562/MREPT_code/nnUNet_results/nnUNetTrainerMRCT_mae_regfix__nnResUNetPlans__3d_fullres'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Check input files
print(f"[INFO] Input dir: {input_dir}")
input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nii.gz')])
print(f"[INFO] Input files: {input_files}")

if not input_files:
    print("[ERROR] No input files found!")
    sys.exit(1)

# Initialize predictor
print(f"[INFO] Loading model from: {model_dir}")
predictor = nnUNetPredictor(
    tile_step_size=0.3,
    use_gaussian=True,
    use_mirroring=False,
    perform_everything_on_device=False,  # Important for CPU
    device=torch.device('cpu'),
    verbose=True,
    verbose_preprocessing=True,
    allow_tqdm=True
)

# Load best checkpoint from all folds
predictor.initialize_from_trained_model_folder(
    model_dir,
    use_folds=['0', '1', '2', '3', '4'],
    checkpoint_name='checkpoint_best.pth',
)

print(f"[INFO] Predicting...")
predictor.predict_from_files(
    list_of_lists_or_source_folder=input_dir,  # Use input_dir directly
    output_folder_or_list_of_truncated_output_files=output_dir,
    save_probabilities=False,
    overwrite=True,
    num_processes_preprocessing=1,
    num_processes_segmentation_export=1,
    reconstruction_mode='gaussian',
)

print(f"[INFO] Prediction complete!")
print(f"[INFO] Results saved to: {output_dir}")

result_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.nii.gz')])
print(f"[INFO] Output files: {result_files}")
