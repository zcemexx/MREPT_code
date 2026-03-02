import os
from copy import deepcopy
from typing import List, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import isfile, load_json, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.regression import (
    convert_regression_logits_to_radius,
    get_kernel_radius_range,
    is_regression_dataset,
)


def _get_current_spacing(configuration_manager: ConfigurationManager, properties_dict: dict):
    return configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]


def _ensure_numpy(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    return np.asarray(array)


def _revert_single_channel_to_original_shape(array: np.ndarray,
                                             plans_manager: PlansManager,
                                             properties_dict: dict,
                                             dtype) -> np.ndarray:
    restored = np.zeros(properties_dict['shape_before_cropping'], dtype=dtype)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    restored[slicer] = array
    return restored.transpose(plans_manager.transpose_backward)


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    current_spacing = _get_current_spacing(configuration_manager, properties_dict)
    predicted_logits = configuration_manager.resampling_fn_probabilities(
        predicted_logits,
        properties_dict['shape_after_cropping_and_before_resampling'],
        current_spacing,
        properties_dict['spacing'],
    )
    predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    del predicted_logits
    segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    segmentation_reverted_cropping = np.zeros(
        properties_dict['shape_before_cropping'],
        dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16,
    )
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    if return_probabilities:
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(
            predicted_probabilities,
            properties_dict['bbox_used_for_cropping'],
            properties_dict['shape_before_cropping'],
        )
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in plans_manager.transpose_backward])
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities

    torch.set_num_threads(old_threads)
    return segmentation_reverted_cropping


def convert_predicted_regression_to_original_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                   valid_mask_preprocessed: Union[torch.Tensor, np.ndarray],
                                                   plans_manager: PlansManager,
                                                   configuration_manager: ConfigurationManager,
                                                   properties_dict: dict,
                                                   dataset_json: dict) -> tuple[np.ndarray, np.ndarray]:
    if not is_regression_dataset(dataset_json):
        raise ValueError('convert_predicted_regression_to_original_shape requires dataset_json["regression_task"]')

    if isinstance(predicted_logits, np.ndarray):
        channel_count = predicted_logits.shape[0]
    else:
        channel_count = int(predicted_logits.shape[0])
    if channel_count != 1:
        raise ValueError(f'Regression export expects a single-channel prediction, got {channel_count} channels')

    current_spacing = _get_current_spacing(configuration_manager, properties_dict)
    pred_radius = convert_regression_logits_to_radius(predicted_logits, dataset_json)
    pred_radius = configuration_manager.resampling_fn_data(
        pred_radius,
        properties_dict['shape_after_cropping_and_before_resampling'],
        current_spacing,
        properties_dict['spacing'],
    )
    valid_mask_preprocessed = _ensure_numpy(valid_mask_preprocessed).astype(np.uint8, copy=False)
    if valid_mask_preprocessed.shape[0] != 1:
        raise ValueError(
            f'Regression export expects valid_mask_preprocessed with shape (1, ...), got {valid_mask_preprocessed.shape}'
        )
    valid_mask_resampled = configuration_manager.resampling_fn_seg(
        valid_mask_preprocessed,
        properties_dict['shape_after_cropping_and_before_resampling'],
        current_spacing,
        properties_dict['spacing'],
    )

    pred_radius = _ensure_numpy(pred_radius)
    valid_mask_resampled = _ensure_numpy(valid_mask_resampled)
    expected_spatial_ndim = len(properties_dict['shape_after_cropping_and_before_resampling'])
    if pred_radius.ndim == expected_spatial_ndim:
        pred_radius = pred_radius[None]
    if valid_mask_resampled.ndim == expected_spatial_ndim:
        valid_mask_resampled = valid_mask_resampled[None]
    if pred_radius.shape[0] != 1:
        raise ValueError(f'Regression export expected a single output channel after resampling, got {pred_radius.shape}')
    if valid_mask_resampled.shape[0] != 1:
        raise ValueError(f'Regression export expected a single mask channel after resampling, got {valid_mask_resampled.shape}')

    radius_original = _revert_single_channel_to_original_shape(
        pred_radius[0],
        plans_manager,
        properties_dict,
        dtype=np.float32,
    )
    valid_mask_original = _revert_single_channel_to_original_shape(
        valid_mask_resampled[0] > 0,
        plans_manager,
        properties_dict,
        dtype=bool,
    ).astype(bool, copy=False)

    kmin, kmax = get_kernel_radius_range(dataset_json)
    quantized = np.zeros_like(radius_original, dtype=np.float32)
    if np.any(valid_mask_original):
        quantized[valid_mask_original] = np.clip(
            np.rint(radius_original[valid_mask_original]),
            kmin,
            kmax,
        )
    out_dtype = np.uint8 if kmax <= 255 else np.uint16
    return quantized.astype(out_dtype), valid_mask_original


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False):
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)
    if is_regression_dataset(dataset_json_dict_or_file):
        raise RuntimeError('Regression export must use export_regression_prediction')

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file,
        plans_manager,
        configuration_manager,
        label_manager,
        properties_dict,
        return_probabilities=save_probabilities,
    )
    del predicted_array_or_file

    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final
    else:
        segmentation_final = ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'], properties_dict)


def export_regression_prediction(predicted_logits: Union[np.ndarray, torch.Tensor],
                                 valid_mask_preprocessed: Union[np.ndarray, torch.Tensor],
                                 properties_dict: dict,
                                 configuration_manager: ConfigurationManager,
                                 plans_manager: PlansManager,
                                 dataset_json: Union[dict, str],
                                 output_file_truncated: str,
                                 save_probabilities: bool = False):
    if isinstance(dataset_json, str):
        dataset_json = load_json(dataset_json)

    radius_map, valid_mask_original = convert_predicted_regression_to_original_shape(
        predicted_logits,
        valid_mask_preprocessed,
        plans_manager,
        configuration_manager,
        properties_dict,
        dataset_json,
    )
    if save_probabilities:
        np.savez_compressed(
            output_file_truncated + '.npz',
            probabilities=radius_map.astype(np.float32),
            valid_mask=valid_mask_original.astype(np.uint8),
        )
        save_pickle(properties_dict, output_file_truncated + '.pkl')

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(radius_map, output_file_truncated + dataset_json['file_ending'], properties_dict)


def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes) \
        -> None:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    current_spacing = _get_current_spacing(configuration_manager, properties_dict)
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(
        predicted,
        target_shape,
        current_spacing,
        target_spacing,
    )

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    np.savez_compressed(output_file, seg=segmentation.astype(np.uint8))
    torch.set_num_threads(old_threads)
