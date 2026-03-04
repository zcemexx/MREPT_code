from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn


ArrayLike = Union[np.ndarray, Tensor]
TISSUE_LABELS = {
    "WM": 1,
    "GM": 2,
    "CSF": 3,
}


def is_regression_dataset(dataset_json: dict) -> bool:
    if not isinstance(dataset_json, dict):
        return False
    if dataset_json.get("regression_task", False):
        return True
    return ("kernel_radius_min" in dataset_json) or ("kernel_radius_max" in dataset_json)


def get_kernel_radius_range(dataset_json: dict) -> Tuple[float, float]:
    kmin = float(dataset_json.get("kernel_radius_min", dataset_json.get("regression_min", 1.0)))
    kmax = float(dataset_json.get("kernel_radius_max", dataset_json.get("regression_max", 30.0)))
    if kmax <= kmin:
        raise ValueError("kernel radius max must be larger than min")
    return kmin, kmax


def get_mask_channel_index(dataset_json: dict) -> int:
    if "mask_channel_index" not in dataset_json:
        raise KeyError("dataset_json is missing required key 'mask_channel_index'")
    idx = int(dataset_json["mask_channel_index"])
    if idx < 0:
        raise ValueError(f"mask_channel_index must be non-negative, got {idx}")
    return idx


def normalize_radius(x: ArrayLike, dataset_json: dict) -> ArrayLike:
    kmin, kmax = get_kernel_radius_range(dataset_json)
    scale = kmax - kmin
    if isinstance(x, torch.Tensor):
        return torch.clamp((x - kmin) / scale, 0.0, 1.0)
    return np.clip((x - kmin) / scale, 0.0, 1.0)


def denormalize_radius(x: ArrayLike, dataset_json: dict) -> ArrayLike:
    kmin, kmax = get_kernel_radius_range(dataset_json)
    scale = kmax - kmin
    return x * scale + kmin


def extract_valid_mask_from_input(data: ArrayLike, dataset_json: dict) -> ArrayLike:
    idx = get_mask_channel_index(dataset_json)
    num_channels = len(dataset_json.get("channel_names", {}))
    if num_channels <= idx:
        raise ValueError(
            f"mask_channel_index={idx} is incompatible with channel_names={dataset_json.get('channel_names')}"
        )

    if isinstance(data, torch.Tensor):
        if data.ndim in (4, 5) and data.shape[1] == num_channels:
            return data[:, idx:idx + 1] > 0
        if data.ndim in (3, 4) and data.shape[0] == num_channels:
            return data[idx:idx + 1][None] > 0
    else:
        if data.ndim in (4, 5) and data.shape[1] == num_channels:
            return data[:, idx:idx + 1] > 0
        if data.ndim in (3, 4) and data.shape[0] == num_channels:
            return data[idx:idx + 1][None] > 0

    raise ValueError(
        f"Unable to infer batch/channel dimensions for valid-mask extraction from shape={tuple(data.shape)}"
    )


def _ensure_mask_matches_shape(mask: Tensor, reference: Tensor) -> Tensor:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.shape == reference.shape:
        return mask
    if mask.shape[0] != reference.shape[0]:
        raise ValueError(f"Mask batch shape mismatch: {mask.shape} vs {reference.shape}")
    if mask.shape[1] == 1 and reference.shape[1] != 1:
        return mask.expand(reference.shape[0], reference.shape[1], *mask.shape[2:])
    raise ValueError(f"Mask shape {mask.shape} is incompatible with reference shape {reference.shape}")


def masked_l1_loss(pred_norm: Tensor, target_norm: Tensor, valid_mask: Tensor) -> Tensor:
    if pred_norm.shape != target_norm.shape:
        raise ValueError(f"pred_norm and target_norm must have the same shape, got {pred_norm.shape} vs {target_norm.shape}")
    valid_mask = _ensure_mask_matches_shape(valid_mask, pred_norm)
    if not torch.any(valid_mask):
        return (pred_norm * 0.0).sum()

    diff = torch.abs(pred_norm - target_norm)
    weight = valid_mask.to(diff.dtype)
    return (diff * weight).sum() / weight.sum()


def masked_gradient_loss(pred_norm: Tensor, target_norm: Tensor, valid_mask: Tensor) -> Tensor:
    if pred_norm.shape != target_norm.shape:
        raise ValueError(f"pred_norm and target_norm must have the same shape, got {pred_norm.shape} vs {target_norm.shape}")
    valid_mask = _ensure_mask_matches_shape(valid_mask, pred_norm)

    losses = []
    for axis in range(2, pred_norm.ndim):
        if pred_norm.shape[axis] <= 1:
            continue

        source_slice = [slice(None)] * pred_norm.ndim
        target_slice = [slice(None)] * pred_norm.ndim
        source_slice[axis] = slice(None, -1)
        target_slice[axis] = slice(1, None)
        source_slice = tuple(source_slice)
        target_slice = tuple(target_slice)

        pred_diff = pred_norm[source_slice] - pred_norm[target_slice]
        target_diff = target_norm[source_slice] - target_norm[target_slice]
        pair_mask = valid_mask[source_slice] & valid_mask[target_slice]
        if not torch.any(pair_mask):
            continue

        pair_weight = pair_mask.to(pred_diff.dtype)
        axis_loss = (torch.abs(pred_diff - target_diff) * pair_weight).sum() / pair_weight.sum()
        losses.append(axis_loss)

    if len(losses) == 0:
        return (pred_norm * 0.0).sum()
    return torch.stack(losses).mean()


def convert_regression_logits_to_radius(predicted_logits: ArrayLike, dataset_json: dict) -> ArrayLike:
    if isinstance(predicted_logits, torch.Tensor):
        pred_norm = torch.sigmoid(predicted_logits)
    else:
        pred_norm = 1.0 / (1.0 + np.exp(-predicted_logits))
    return denormalize_radius(pred_norm, dataset_json)


def compute_masked_pearson_r(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    if pred_valid.size < 2:
        return np.nan

    pred_centered = pred_valid - np.mean(pred_valid)
    gt_centered = gt_valid - np.mean(gt_valid)
    pred_var = float(np.sum(np.square(pred_centered)))
    gt_var = float(np.sum(np.square(gt_centered)))
    if pred_var == 0.0 or gt_var == 0.0:
        return np.nan

    covariance = float(np.sum(pred_centered * gt_centered))
    return covariance / np.sqrt(pred_var * gt_var)


def compute_masked_gradient_mae(pred: np.ndarray,
                                gt: np.ndarray,
                                valid_mask: np.ndarray,
                                spacing: Optional[Tuple[float, ...]] = None) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if pred.shape != gt.shape or pred.shape != valid_mask.shape:
        raise ValueError(
            f"Shape mismatch for masked gradient MAE: pred={pred.shape}, gt={gt.shape}, mask={valid_mask.shape}"
        )
    if valid_mask.sum() < 2:
        return np.nan

    if spacing is None:
        spacing = (1.0,) * pred.ndim
    elif len(spacing) != pred.ndim:
        raise ValueError(f"spacing must have length {pred.ndim}, got {spacing}")

    grad_err_sum = 0.0
    valid_pairs_count = 0
    for axis in range(pred.ndim):
        if pred.shape[axis] <= 1:
            continue

        diff_pred = np.diff(pred, axis=axis)
        diff_gt = np.diff(gt, axis=axis)

        source_slice = [slice(None)] * pred.ndim
        target_slice = [slice(None)] * pred.ndim
        source_slice[axis] = slice(None, -1)
        target_slice[axis] = slice(1, None)
        pair_mask = valid_mask[tuple(source_slice)] & valid_mask[tuple(target_slice)]
        if not np.any(pair_mask):
            continue

        grad_err = np.abs(diff_pred[pair_mask] - diff_gt[pair_mask]) / float(spacing[axis])
        grad_err_sum += float(np.sum(grad_err))
        valid_pairs_count += int(np.count_nonzero(pair_mask))

    if valid_pairs_count == 0:
        return np.nan
    return grad_err_sum / valid_pairs_count


def _compute_tissue_mae(pred: np.ndarray,
                        gt: np.ndarray,
                        valid_mask: np.ndarray,
                        tissue_mask: Optional[np.ndarray],
                        tissue_value: int) -> Tuple[float, int]:
    if tissue_mask is None:
        return np.nan, 0

    tissue_valid_mask = valid_mask & (tissue_mask == tissue_value)
    tissue_voxels = int(np.count_nonzero(tissue_valid_mask))
    if tissue_voxels == 0:
        return np.nan, 0
    return float(np.mean(np.abs(pred[tissue_valid_mask] - gt[tissue_valid_mask]))), tissue_voxels


def compute_regression_case_metrics(pred_radius: np.ndarray,
                                    gt_radius: np.ndarray,
                                    valid_mask: np.ndarray,
                                    tissue_mask: Optional[np.ndarray] = None,
                                    spacing: Optional[Tuple[float, ...]] = None) -> dict:
    pred_radius = np.asarray(pred_radius, dtype=np.float32)
    gt_radius = np.asarray(gt_radius, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if pred_radius.shape != gt_radius.shape or pred_radius.shape != valid_mask.shape:
        raise ValueError(
            f"Shape mismatch for regression metrics: pred={pred_radius.shape}, gt={gt_radius.shape}, mask={valid_mask.shape}"
        )
    if tissue_mask is not None:
        tissue_mask = np.asarray(tissue_mask)
        if tissue_mask.shape != pred_radius.shape:
            raise ValueError(
                f"Shape mismatch for tissue metrics: tissue_mask={tissue_mask.shape}, expected={pred_radius.shape}"
            )

    valid_voxels = int(valid_mask.sum())
    metrics = {
        "Pearson_r": np.nan,
        "Gradient_MAE": np.nan,
    }
    for tissue_name in TISSUE_LABELS:
        metrics[f"{tissue_name}_MAE"] = np.nan
        metrics[f"{tissue_name}_ValidVoxelCount"] = 0

    if valid_voxels == 0:
        metrics.update({
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "Acc_1": np.nan,
            "Acc_3": np.nan,
            "Acc_5": np.nan,
            "ValidVoxelCount": 0,
        })
        return metrics

    errors = pred_radius[valid_mask] - gt_radius[valid_mask]
    abs_errors = np.abs(errors)
    mse = float(np.mean(np.square(errors)))
    metrics.update({
        "MAE": float(np.mean(abs_errors)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "Acc_1": float(np.mean(abs_errors <= 1.0)),
        "Acc_3": float(np.mean(abs_errors <= 3.0)),
        "Acc_5": float(np.mean(abs_errors <= 5.0)),
        "ValidVoxelCount": valid_voxels,
        "Pearson_r": compute_masked_pearson_r(pred_radius, gt_radius, valid_mask),
        "Gradient_MAE": compute_masked_gradient_mae(pred_radius, gt_radius, valid_mask, spacing=spacing),
    })
    for tissue_name, tissue_value in TISSUE_LABELS.items():
        tissue_mae, tissue_voxels = _compute_tissue_mae(pred_radius, gt_radius, valid_mask, tissue_mask, tissue_value)
        metrics[f"{tissue_name}_MAE"] = tissue_mae
        metrics[f"{tissue_name}_ValidVoxelCount"] = tissue_voxels
    return metrics


class MaskedL1Loss(nn.Module):
    def forward(self, pred_norm: Tensor, target_norm: Tensor, valid_mask: Tensor) -> Tensor:
        return masked_l1_loss(pred_norm, target_norm, valid_mask)


class MaskedL1AndGradientLoss(nn.Module):
    def __init__(self, lambda_l1: float = 1.0, lambda_grad: float = 0.1):
        super().__init__()
        self.lambda_l1 = float(lambda_l1)
        self.lambda_grad = float(lambda_grad)

    def forward(self, pred_norm: Tensor, target_norm: Tensor, valid_mask: Tensor) -> Tensor:
        l1 = masked_l1_loss(pred_norm, target_norm, valid_mask)
        grad = masked_gradient_loss(pred_norm, target_norm, valid_mask)
        return self.lambda_l1 * l1 + self.lambda_grad * grad
