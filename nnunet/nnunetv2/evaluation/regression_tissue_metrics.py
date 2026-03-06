from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join, maybe_mkdir_p, subfiles

from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter

try:
    from scipy.ndimage import binary_erosion as scipy_binary_erosion
except ImportError:  # pragma: no cover - exercised via fallback unit tests
    scipy_binary_erosion = None


TISSUE_ORDER = ("Global", "WM", "GM", "CSF")
TISSUE_LABELS = {"WM": 1, "GM": 2, "CSF": 3}
LONG_METRIC_KEYS = (
    "MAE",
    "MSE",
    "RMSE",
    "Acc_1",
    "Acc_3",
    "Acc_5",
    "Gradient_MAE",
    "Pearson_R",
    "ValidVoxelCount",
    "TotalVoxelCount",
    "ValidRatio",
)


def _unwrap_single_channel_seg(seg: np.ndarray, role: str) -> np.ndarray:
    seg = np.asarray(seg)
    if seg.ndim >= 4 and seg.shape[0] == 1:
        return seg[0]
    if seg.ndim >= 4 and seg.shape[0] != 1:
        raise ValueError(f"{role} must be single-channel for tissue-aware regression evaluation, got shape {seg.shape}")
    return seg


def _is_close_array(a: Sequence[float], b: Sequence[float], atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=atol, rtol=rtol)


def _extract_affine(properties: dict) -> Optional[np.ndarray]:
    affine = properties.get("nibabel_stuff", {}).get("original_affine")
    if affine is None:
        return None
    return np.asarray(affine, dtype=float)


def _transpose_affine_for_reversed_axes(affine: np.ndarray) -> np.ndarray:
    flip = np.eye(4, dtype=float)
    flip[:3, :3] = np.fliplr(np.eye(3, dtype=float))
    return affine @ flip


def align_prediction_to_gt(pred_data: np.ndarray, gt_shape: tuple) -> np.ndarray:
    if pred_data.shape == gt_shape:
        return pred_data
    if pred_data.shape == gt_shape[::-1]:
        return np.transpose(pred_data)
    raise ValueError(f"Shape mismatch: Pred {pred_data.shape} vs GT {gt_shape}. Fails sanity check.")


def build_tissue_masks(tissue_mask_volume: np.ndarray) -> dict[str, np.ndarray]:
    tissue_mask_volume = np.asarray(tissue_mask_volume)
    if tissue_mask_volume.ndim != 3:
        raise ValueError(f"tissue_mask_volume must be 3D, got shape {tissue_mask_volume.shape}")

    masks = {"Global": tissue_mask_volume > 0}
    for tissue_name, label in TISSUE_LABELS.items():
        masks[tissue_name] = tissue_mask_volume == label
    return masks


def compute_masked_gradient_mae(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    spacing: Optional[tuple] = None,
) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if pred.shape != gt.shape or pred.shape != valid_mask.shape:
        raise ValueError(f"Shape mismatch for gradient MAE: {pred.shape}, {gt.shape}, {valid_mask.shape}")
    if valid_mask.sum() < 2:
        return np.nan

    grad_err_sum = 0.0
    valid_pairs_count = 0
    for axis in range(pred.ndim):
        diff_pred = np.diff(pred, axis=axis)
        diff_gt = np.diff(gt, axis=axis)

        slices_curr = [slice(None)] * pred.ndim
        slices_next = [slice(None)] * pred.ndim
        slices_curr[axis] = slice(0, -1)
        slices_next[axis] = slice(1, None)
        mask_diff = valid_mask[tuple(slices_curr)] & valid_mask[tuple(slices_next)]

        if mask_diff.any():
            sp = float(spacing[axis]) if spacing is not None else 1.0
            grad_err = np.abs(diff_pred[mask_diff] - diff_gt[mask_diff]) / sp
            grad_err_sum += float(np.sum(grad_err))
            valid_pairs_count += int(np.count_nonzero(mask_diff))

    if valid_pairs_count == 0:
        return np.nan
    return grad_err_sum / valid_pairs_count


def compute_masked_pearson_r(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if pred.shape != gt.shape or pred.shape != valid_mask.shape:
        raise ValueError(f"Shape mismatch for Pearson R: {pred.shape}, {gt.shape}, {valid_mask.shape}")

    p = pred[valid_mask]
    g = gt[valid_mask]
    if len(p) < 2:
        return np.nan

    p_diff = p - np.mean(p)
    g_diff = g - np.mean(g)
    var_p = float(np.sum(p_diff ** 2))
    var_g = float(np.sum(g_diff ** 2))
    if var_p == 0.0 or var_g == 0.0:
        return np.nan
    return float(np.sum(p_diff * g_diff) / np.sqrt(var_p * var_g))


def compute_residual_histogram(eres: np.ndarray, valid_mask: np.ndarray, limit: int = 5):
    eres = np.asarray(eres, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    valid_res = eres[valid_mask]
    if valid_res.size == 0:
        bins = np.arange(-(limit + 0.5), (limit + 1.5), 1.0, dtype=np.float32)
        stats = {
            "Residual_Mean": np.nan,
            "Residual_Median": np.nan,
            "Residual_STD": np.nan,
            "Residual_Skewness": np.nan,
        }
        return np.zeros(len(bins) - 1, dtype=np.int64), bins, stats

    clipped_res = np.clip(valid_res, -limit, limit)
    bins = np.arange(-(limit + 0.5), (limit + 1.5), 1.0, dtype=np.float32)
    counts, edges = np.histogram(clipped_res, bins=bins)

    mean_val = float(np.mean(valid_res))
    std_val = float(np.std(valid_res))
    if valid_res.size < 3 or std_val == 0.0:
        skewness = np.nan
    else:
        centered = (valid_res - mean_val) / std_val
        skewness = float(np.mean(centered ** 3))
    stats = {
        "Residual_Mean": mean_val,
        "Residual_Median": float(np.median(valid_res)),
        "Residual_STD": std_val,
        "Residual_Skewness": skewness,
    }
    return counts.astype(np.int64), edges.astype(np.float32), stats


def find_optimal_slice(global_mask: np.ndarray, axis: int = 2) -> int:
    global_mask = np.asarray(global_mask, dtype=bool)
    if global_mask.ndim != 3:
        raise ValueError(f"global_mask must be 3D, got shape {global_mask.shape}")
    if axis < 0 or axis >= global_mask.ndim:
        raise ValueError(f"axis must be within [0, {global_mask.ndim - 1}], got {axis}")

    axes_to_sum = tuple(i for i in range(global_mask.ndim) if i != axis)
    valid_counts_per_slice = global_mask.sum(axis=axes_to_sum)
    if valid_counts_per_slice.max() == 0:
        return global_mask.shape[axis] // 2
    return int(np.argmax(valid_counts_per_slice))


def _binary_erosion_numpy(tissue_mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    eroded = np.asarray(tissue_mask, dtype=bool)
    for _ in range(iterations):
        current = eroded.copy()
        for axis in range(eroded.ndim):
            current &= np.roll(eroded, 1, axis=axis)
            current &= np.roll(eroded, -1, axis=axis)
        for axis in range(eroded.ndim):
            leading = [slice(None)] * eroded.ndim
            trailing = [slice(None)] * eroded.ndim
            leading[axis] = 0
            trailing[axis] = -1
            current[tuple(leading)] = False
            current[tuple(trailing)] = False
        eroded = current
    return eroded


def compute_boundary_mask(tissue_mask: np.ndarray, width: int = 1) -> np.ndarray:
    tissue_mask = np.asarray(tissue_mask, dtype=bool)
    if tissue_mask.ndim != 3:
        raise ValueError(f"tissue_mask must be 3D, got shape {tissue_mask.shape}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")

    if scipy_binary_erosion is not None:
        eroded_mask = scipy_binary_erosion(tissue_mask, iterations=width)
    else:
        eroded_mask = _binary_erosion_numpy(tissue_mask, iterations=width)
    return tissue_mask ^ eroded_mask


def compute_metrics_for_named_mask(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    spacing: Optional[tuple] = None,
    include_gradient_mae: bool = True,
    include_pearson_r: bool = True,
) -> dict:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    total_voxel_count = int(np.count_nonzero(valid_mask))
    valid_voxel_count = int(np.count_nonzero(valid_mask))
    if pred.shape != gt.shape or pred.shape != valid_mask.shape:
        raise ValueError(f"Shape mismatch for metrics: pred={pred.shape}, gt={gt.shape}, mask={valid_mask.shape}")

    if valid_voxel_count == 0:
        return {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "Acc_1": np.nan,
            "Acc_3": np.nan,
            "Acc_5": np.nan,
            "Gradient_MAE": np.nan,
            "Pearson_R": np.nan,
            "ValidVoxelCount": 0,
            "TotalVoxelCount": total_voxel_count,
            "ValidRatio": 0.0 if total_voxel_count > 0 else np.nan,
        }

    errors = pred[valid_mask] - gt[valid_mask]
    abs_errors = np.abs(errors)
    mse = float(np.mean(errors ** 2))
    result = {
        "MAE": float(np.mean(abs_errors)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "Acc_1": float(np.mean(abs_errors <= 1.0)),
        "Acc_3": float(np.mean(abs_errors <= 3.0)),
        "Acc_5": float(np.mean(abs_errors <= 5.0)),
        "Gradient_MAE": np.nan,
        "Pearson_R": np.nan,
        "ValidVoxelCount": valid_voxel_count,
        "TotalVoxelCount": total_voxel_count,
        "ValidRatio": float(valid_voxel_count / total_voxel_count) if total_voxel_count > 0 else np.nan,
    }
    if include_gradient_mae:
        result["Gradient_MAE"] = compute_masked_gradient_mae(pred, gt, valid_mask, spacing=spacing)
    if include_pearson_r:
        result["Pearson_R"] = compute_masked_pearson_r(pred, gt, valid_mask)
    return result


def _metrics_with_total(
    pred: np.ndarray,
    gt: np.ndarray,
    tissue_mask: np.ndarray,
    spacing: Optional[tuple],
    include_gradient_mae: bool,
    include_pearson_r: bool,
) -> dict:
    tissue_mask = np.asarray(tissue_mask, dtype=bool)
    valid_mask = tissue_mask & np.isfinite(pred) & np.isfinite(gt)
    result = compute_metrics_for_named_mask(
        pred,
        gt,
        valid_mask,
        spacing=spacing,
        include_gradient_mae=include_gradient_mae,
        include_pearson_r=include_pearson_r,
    )
    result["TotalVoxelCount"] = int(np.count_nonzero(tissue_mask))
    if result["TotalVoxelCount"] == 0:
        result["ValidRatio"] = np.nan
    elif result["ValidVoxelCount"] == 0:
        result["ValidRatio"] = 0.0
    else:
        result["ValidRatio"] = float(result["ValidVoxelCount"] / result["TotalVoxelCount"])
    return result


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _save_json(data: dict, output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, indent=2, sort_keys=True)


def _write_csv(rows: List[dict], output_file: str, fieldnames: Sequence[str]) -> None:
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_serializable(row.get(k)) for k in fieldnames})


def _load_case_volume(path: str, image_reader_writer: BaseReaderWriter, role: str) -> tuple[np.ndarray, dict]:
    arr, props = image_reader_writer.read_seg(path)
    return _unwrap_single_channel_seg(arr, role).astype(np.float32, copy=False), props


def _derive_tissue_mask_path(reference_file: str, folder_images: Optional[str], tissue_channel_suffix: str) -> str:
    case_id = Path(reference_file).name
    if case_id.endswith(".nii.gz"):
        case_id = case_id[:-7]
    else:
        case_id = Path(case_id).stem
    if folder_images is None:
        folder_images = str(Path(reference_file).resolve().parents[1] / "imagesTr")
    return join(folder_images, f"{case_id}{tissue_channel_suffix}.nii.gz")


def _build_case_failure(case_id: str, status: str, warning_messages: Optional[List[str]] = None) -> dict:
    warning_messages = list(warning_messages or [])
    tissue_metrics = {}
    for tissue in TISSUE_ORDER:
        tissue_metrics[tissue] = {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "Acc_1": np.nan,
            "Acc_3": np.nan,
            "Acc_5": np.nan,
            "Gradient_MAE": np.nan,
            "Pearson_R": np.nan,
            "ValidVoxelCount": 0,
            "TotalVoxelCount": 0,
            "ValidRatio": np.nan,
        }
    return {
        "case_id": case_id,
        "status": status,
        "warnings": warning_messages,
        "axis_swap_detected": False,
        "affine_untrusted_after_transpose": False,
        "tissue_metrics": tissue_metrics,
        "boundary_metrics": {
            "Boundary_MAE": np.nan,
            "Boundary_RMSE": np.nan,
            "Boundary_Acc_1": np.nan,
            "Boundary_Acc_3": np.nan,
            "Boundary_Acc_5": np.nan,
            "Boundary_Gradient_MAE": np.nan,
            "Boundary_Pearson_R": np.nan,
            "BoundaryVoxelCount": 0,
            "Interior_MAE": np.nan,
            "Interior_RMSE": np.nan,
            "Status": status,
        },
        "residual_distribution": {
            "Residual_Mean": np.nan,
            "Residual_Median": np.nan,
            "Residual_STD": np.nan,
            "Residual_Skewness": np.nan,
            "Histogram_Counts": [],
            "Histogram_Edges": [],
        },
        "paths": {},
    }


def _check_unknown_labels(tissue_mask_volume: np.ndarray) -> List[int]:
    values = np.unique(np.asarray(tissue_mask_volume))
    return [int(v) for v in values if v not in (0, 1, 2, 3)]


def _render_case_plots(
    case_result: dict,
    pred: np.ndarray,
    gt: np.ndarray,
    residual: np.ndarray,
    combined_boundary: np.ndarray,
    global_valid: np.ndarray,
    output_dir: str,
    slice_axis: int,
    residual_limit: int,
) -> dict:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required to render plots. Install matplotlib or omit plots_dir.") from exc

    maybe_mkdir_p(output_dir)
    slice_idx = find_optimal_slice(global_valid, axis=slice_axis)
    gt_slice = np.take(gt, slice_idx, axis=slice_axis)
    pred_slice = np.take(pred, slice_idx, axis=slice_axis)
    residual_slice = np.take(residual, slice_idx, axis=slice_axis)
    boundary_slice = np.take(combined_boundary.astype(np.uint8), slice_idx, axis=slice_axis).astype(bool)

    triptych_path = join(output_dir, "triptych.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(gt_slice, cmap="viridis")
    axes[0].set_title("GT Radius")
    axes[1].imshow(pred_slice, cmap="viridis")
    axes[1].set_title("Pred Radius")
    im = axes[2].imshow(residual_slice, cmap="RdBu_r", vmin=-residual_limit, vmax=residual_limit)
    axes[2].set_title("Residual")
    for ax in axes:
        ax.set_axis_off()
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    fig.savefig(triptych_path, dpi=150)
    plt.close(fig)

    hist_path = join(output_dir, "residual_histogram.png")
    counts = np.asarray(case_result["residual_distribution"]["Histogram_Counts"], dtype=np.int64)
    edges = np.asarray(case_result["residual_distribution"]["Histogram_Edges"], dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:]) if edges.size > 1 else np.arange(counts.size)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    ax.bar(centers, counts, width=0.9, align="center")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title(f"{case_result['case_id']} residual histogram")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    overlay_path = join(output_dir, "boundary_overlay.png")
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    im = ax.imshow(residual_slice, cmap="RdBu_r", vmin=-residual_limit, vmax=residual_limit)
    if np.any(boundary_slice):
        ax.contour(boundary_slice.astype(np.uint8), levels=[0.5], colors="k", linewidths=0.6)
    ax.set_title("Residual with Tissue Boundary")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)

    return {
        "triptych": triptych_path,
        "residual_histogram": hist_path,
        "boundary_overlay": overlay_path,
        "slice_axis": slice_axis,
        "slice_index": slice_idx,
    }


def _compute_case_from_loaded_arrays(
    case_id: str,
    pred: np.ndarray,
    pred_props: dict,
    gt: np.ndarray,
    gt_props: dict,
    tissue_mask_volume: np.ndarray,
    tissue_props: dict,
    include_gradient_mae: bool,
    include_pearson_r: bool,
    boundary_width: int,
    residual_limit: int,
) -> dict:
    warnings_list: List[str] = []
    axis_swap_detected = False
    affine_untrusted_after_transpose = False

    spacing_gt = tuple(gt_props.get("spacing", ()))
    spacing_pred = tuple(pred_props.get("spacing", ()))
    spacing_tissue = tuple(tissue_props.get("spacing", ()))

    if pred.shape != gt.shape:
        aligned_pred = align_prediction_to_gt(pred, gt.shape)
        if aligned_pred.shape == gt.shape and pred.shape == gt.shape[::-1]:
            axis_swap_detected = True
            warnings_list.append("Detected reversed prediction axes and applied transpose alignment.")
            pred = aligned_pred
        else:
            pred = aligned_pred

    if tissue_mask_volume.shape != gt.shape:
        raise ValueError(f"Tissue mask shape mismatch: mask={tissue_mask_volume.shape}, gt={gt.shape}")
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction shape mismatch after alignment: pred={pred.shape}, gt={gt.shape}")
    if not _is_close_array(spacing_gt, spacing_pred):
        raise ValueError(f"Spacing mismatch between prediction and GT: pred={spacing_pred}, gt={spacing_gt}")
    if not _is_close_array(spacing_gt, spacing_tissue):
        raise ValueError(f"Spacing mismatch between tissue mask and GT: mask={spacing_tissue}, gt={spacing_gt}")

    affine_gt = _extract_affine(gt_props)
    affine_pred = _extract_affine(pred_props)
    affine_tissue = _extract_affine(tissue_props)

    if affine_gt is not None and affine_tissue is not None and not np.allclose(affine_gt, affine_tissue, atol=1e-5, rtol=1e-5):
        raise ValueError("Affine mismatch between tissue mask and GT.")

    status = "ok"
    if affine_gt is not None and affine_pred is not None:
        aff_pred_to_check = affine_pred
        if axis_swap_detected:
            aff_pred_to_check = _transpose_affine_for_reversed_axes(affine_pred)
        if not np.allclose(aff_pred_to_check, affine_gt, atol=1e-5, rtol=1e-5):
            if axis_swap_detected:
                affine_untrusted_after_transpose = True
                status = "warning_alignment"
                warnings_list.append("Prediction affine differs from GT after transpose alignment; proceeding with warning.")
            else:
                raise ValueError("Affine mismatch between prediction and GT.")

    unknown_labels = _check_unknown_labels(tissue_mask_volume)
    if unknown_labels:
        warnings_list.append(f"Ignoring unknown tissue labels: {unknown_labels}")

    tissue_masks = build_tissue_masks(tissue_mask_volume)
    global_valid = tissue_masks["Global"] & np.isfinite(pred) & np.isfinite(gt)
    residual = pred - gt

    tissue_metrics = {}
    for tissue_name in TISSUE_ORDER:
        tissue_metrics[tissue_name] = _metrics_with_total(
            pred,
            gt,
            tissue_masks[tissue_name],
            spacing=spacing_gt if spacing_gt else None,
            include_gradient_mae=include_gradient_mae,
            include_pearson_r=include_pearson_r,
        )

    combined_boundary = np.zeros_like(tissue_masks["Global"], dtype=bool)
    for tissue_name in ("WM", "GM", "CSF"):
        combined_boundary |= compute_boundary_mask(tissue_masks[tissue_name], width=boundary_width)
    boundary_valid = combined_boundary & np.isfinite(pred) & np.isfinite(gt)
    interior_valid = global_valid & ~combined_boundary

    boundary_detail = compute_metrics_for_named_mask(
        pred,
        gt,
        boundary_valid,
        spacing=spacing_gt if spacing_gt else None,
        include_gradient_mae=include_gradient_mae,
        include_pearson_r=include_pearson_r,
    )
    interior_detail = compute_metrics_for_named_mask(
        pred,
        gt,
        interior_valid,
        spacing=spacing_gt if spacing_gt else None,
        include_gradient_mae=include_gradient_mae,
        include_pearson_r=include_pearson_r,
    )
    boundary_metrics = {
        "Boundary_MAE": boundary_detail["MAE"],
        "Boundary_RMSE": boundary_detail["RMSE"],
        "Boundary_Acc_1": boundary_detail["Acc_1"],
        "Boundary_Acc_3": boundary_detail["Acc_3"],
        "Boundary_Acc_5": boundary_detail["Acc_5"],
        "Boundary_Gradient_MAE": boundary_detail["Gradient_MAE"],
        "Boundary_Pearson_R": boundary_detail["Pearson_R"],
        "BoundaryVoxelCount": boundary_detail["ValidVoxelCount"],
        "Interior_MAE": interior_detail["MAE"],
        "Interior_RMSE": interior_detail["RMSE"],
        "Status": status,
    }

    hist_counts, hist_edges, hist_stats = compute_residual_histogram(residual, global_valid, limit=residual_limit)
    residual_distribution = dict(hist_stats)
    residual_distribution["Histogram_Counts"] = hist_counts.tolist()
    residual_distribution["Histogram_Edges"] = hist_edges.tolist()

    return {
        "case_id": case_id,
        "status": status,
        "warnings": warnings_list,
        "axis_swap_detected": axis_swap_detected,
        "affine_untrusted_after_transpose": affine_untrusted_after_transpose,
        "tissue_metrics": tissue_metrics,
        "boundary_metrics": boundary_metrics,
        "residual_distribution": residual_distribution,
        "paths": {},
        "_plot_payload": {
            "pred": pred,
            "gt": gt,
            "residual": residual,
            "combined_boundary": combined_boundary,
            "global_valid": global_valid,
        },
    }


def compute_regression_metrics(
    reference_file: str,
    prediction_file: str,
    image_reader_writer: BaseReaderWriter,
    tissue_mask_file: Optional[str] = None,
    tissue_channel_suffix: str = "_0001",
    include_gradient_mae: bool = True,
    include_pearson_r: bool = True,
    boundary_width: int = 1,
) -> dict:
    case_id = Path(reference_file).name
    if case_id.endswith(".nii.gz"):
        case_id = case_id[:-7]
    else:
        case_id = Path(case_id).stem

    if tissue_mask_file is None:
        tissue_mask_file = _derive_tissue_mask_path(reference_file, None, tissue_channel_suffix)
    if not isfile(reference_file):
        return _build_case_failure(case_id, "missing_reference")
    if not isfile(prediction_file):
        return _build_case_failure(case_id, "missing_prediction")
    if not isfile(tissue_mask_file):
        return _build_case_failure(case_id, "missing_tissue_mask")

    gt, gt_props = _load_case_volume(reference_file, image_reader_writer, "reference")
    pred, pred_props = _load_case_volume(prediction_file, image_reader_writer, "prediction")
    tissue_mask_volume, tissue_props = _load_case_volume(tissue_mask_file, image_reader_writer, "tissue_mask")
    return _compute_case_from_loaded_arrays(
        case_id=case_id,
        pred=pred,
        pred_props=pred_props,
        gt=gt,
        gt_props=gt_props,
        tissue_mask_volume=tissue_mask_volume,
        tissue_props=tissue_props,
        include_gradient_mae=include_gradient_mae,
        include_pearson_r=include_pearson_r,
        boundary_width=boundary_width,
        residual_limit=5,
    )


def _case_result_to_long_rows(case_result: dict) -> List[dict]:
    rows = []
    for tissue_name in TISSUE_ORDER:
        metrics = case_result["tissue_metrics"][tissue_name]
        row = {"Case": case_result["case_id"], "Tissue": tissue_name, "Status": case_result["status"]}
        for key in LONG_METRIC_KEYS:
            row[key] = metrics.get(key, np.nan)
        rows.append(row)
    return rows


def _case_result_to_boundary_row(case_result: dict) -> dict:
    row = {"Case": case_result["case_id"]}
    row.update(case_result["boundary_metrics"])
    row["Status"] = case_result["status"]
    return row


def _case_result_to_residual_row(case_result: dict) -> dict:
    row = {"Case": case_result["case_id"], "Status": case_result["status"]}
    row.update({k: v for k, v in case_result["residual_distribution"].items() if not k.startswith("Histogram_")})
    return row


def _build_wide_rows(long_rows: List[dict]) -> List[dict]:
    by_case: Dict[str, dict] = {}
    for row in long_rows:
        case_row = by_case.setdefault(row["Case"], {"Case": row["Case"], "Status": row["Status"]})
        for key in LONG_METRIC_KEYS:
            case_row[f"{row['Tissue']}_{key}"] = row.get(key, np.nan)
        if row["Status"] != "ok" and case_row["Status"] == "ok":
            case_row["Status"] = row["Status"]
    return [by_case[k] for k in sorted(by_case.keys())]


def _mean_ignore_nan(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def _build_summary(
    case_results: List[dict],
    ref_only_cases: List[str],
    pred_only_cases: List[str],
    missing_tissue_cases: List[str],
    expected_case_count: int,
    predicted_case_count: int,
    evaluated_case_count: int,
) -> dict:
    foreground_mean = {}
    for tissue_name in TISSUE_ORDER:
        foreground_mean[tissue_name] = {}
        for key in LONG_METRIC_KEYS:
            foreground_mean[tissue_name][key] = _mean_ignore_nan(
                [case["tissue_metrics"][tissue_name][key] for case in case_results]
            )

    status_counts: Dict[str, int] = {}
    for case in case_results:
        status_counts[case["status"]] = status_counts.get(case["status"], 0) + 1

    result = {
        "task_type": "regression_tissue_eval",
        "case_counts": {
            "expected_cases": expected_case_count,
            "predicted_cases": predicted_case_count,
            "evaluated_cases": evaluated_case_count,
            # Backward-compatible aliases:
            "missing_cases": ref_only_cases,
            "unexpected_cases": pred_only_cases,
            # Preferred explicit fields:
            "ref_only_cases": ref_only_cases,
            "pred_only_cases": pred_only_cases,
            "missing_tissue_cases": missing_tissue_cases,
            "status_counts": status_counts,
        },
        "metric_per_case": deepcopy(case_results),
        "foreground_mean": foreground_mean,
        "selection_metric": {
            "name": "Global.MAE",
            "mode": "min",
            "value": foreground_mean["Global"]["MAE"],
        },
    }
    for case in result["metric_per_case"]:
        case.pop("_plot_payload", None)
    return result


def _worker_compute_case(task: dict) -> dict:
    rw = task["rw_class"]()
    try:
        gt, gt_props = _load_case_volume(task["reference_file"], rw, "reference")
        pred, pred_props = _load_case_volume(task["prediction_file"], rw, "prediction")
        tissue_mask_volume, tissue_props = _load_case_volume(task["tissue_mask_file"], rw, "tissue_mask")
        return _compute_case_from_loaded_arrays(
            case_id=task["case_id"],
            pred=pred,
            pred_props=pred_props,
            gt=gt,
            gt_props=gt_props,
            tissue_mask_volume=tissue_mask_volume,
            tissue_props=tissue_props,
            include_gradient_mae=task["include_gradient_mae"],
            include_pearson_r=task["include_pearson_r"],
            boundary_width=task["boundary_width"],
            residual_limit=task["residual_limit"],
        )
    except Exception as exc:  # pragma: no cover - covered indirectly in integration-style code path
        return _build_case_failure(task["case_id"], "failed", [str(exc)])


def compute_regression_metrics_on_folder_with_tissues(
    folder_ref: str,
    folder_pred: str,
    folder_images: str,
    output_file: Optional[str],
    image_reader_writer: BaseReaderWriter,
    file_ending: str = ".nii.gz",
    tissue_channel_suffix: str = "_0001",
    num_processes: int = default_num_processes,
    include_gradient_mae: bool = True,
    include_pearson_r: bool = True,
    plots_dir: Optional[str] = None,
    boundary_width: int = 1,
    residual_limit: int = 5,
    slice_axis: int = 2,
) -> dict:
    if output_file is None:
        output_file = join(folder_pred, "summary.json")
    if not output_file.endswith(".json"):
        raise AssertionError("output_file should end with .json")

    summary_dir = str(Path(output_file).resolve().parent)
    maybe_mkdir_p(summary_dir)
    case_dir_root = join(summary_dir, "cases")
    maybe_mkdir_p(case_dir_root)
    if plots_dir is not None:
        maybe_mkdir_p(plots_dir)

    files_ref = sorted(subfiles(folder_ref, suffix=file_ending, join=False))
    files_pred = sorted(subfiles(folder_pred, suffix=file_ending, join=False))
    ref_cases = {f[:-len(file_ending)] for f in files_ref}
    pred_cases = {f[:-len(file_ending)] for f in files_pred}
    eval_cases = sorted(ref_cases & pred_cases)
    ref_only_cases = sorted(ref_cases - pred_cases)
    pred_only_cases = sorted(pred_cases - ref_cases)
    expected_case_count = len(eval_cases)
    predicted_case_count = len(pred_cases)
    evaluated_case_count = len(eval_cases)

    case_results: List[dict] = []
    missing_tissue_cases: List[str] = []
    tasks: List[dict] = []
    for case_id in eval_cases:
        reference_file = join(folder_ref, case_id + file_ending)
        prediction_file = join(folder_pred, case_id + file_ending)
        tissue_mask_file = join(folder_images, f"{case_id}{tissue_channel_suffix}{file_ending}")
        if not isfile(prediction_file):
            # Defensive fallback; eval_cases should guarantee this is present.
            continue
        if not isfile(tissue_mask_file):
            missing_tissue_cases.append(case_id)
            case_results.append(_build_case_failure(case_id, "missing_tissue_mask"))
            continue
        tasks.append(
            {
                "case_id": case_id,
                "reference_file": reference_file,
                "prediction_file": prediction_file,
                "tissue_mask_file": tissue_mask_file,
                "rw_class": image_reader_writer.__class__,
                "include_gradient_mae": include_gradient_mae,
                "include_pearson_r": include_pearson_r,
                "boundary_width": boundary_width,
                "residual_limit": residual_limit,
            }
        )

    if tasks:
        max_workers = max(1, int(num_processes))
        if max_workers == 1:
            for task in tasks:
                case_results.append(_worker_compute_case(task))
        else:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_case = {executor.submit(_worker_compute_case, task): task["case_id"] for task in tasks}
                    for future in as_completed(future_to_case):
                        case_results.append(future.result())
            except PermissionError:
                for task in tasks:
                    result = _worker_compute_case(task)
                    result["warnings"].append(
                        "ProcessPoolExecutor unavailable in current environment; fell back to serial execution."
                    )
                    case_results.append(result)

    case_results.sort(key=lambda x: x["case_id"])

    long_rows = []
    boundary_rows = []
    residual_rows = []
    for case_result in case_results:
        long_rows.extend(_case_result_to_long_rows(case_result))
        boundary_rows.append(_case_result_to_boundary_row(case_result))
        residual_rows.append(_case_result_to_residual_row(case_result))

        case_dir = join(case_dir_root, case_result["case_id"])
        maybe_mkdir_p(case_dir)
        plot_payload = case_result.pop("_plot_payload", None)
        if plots_dir is not None and plot_payload is not None and case_result["status"] in ("ok", "warning_alignment"):
            plot_case_dir = join(plots_dir, case_result["case_id"])
            try:
                plot_paths = _render_case_plots(
                    case_result,
                    pred=plot_payload["pred"],
                    gt=plot_payload["gt"],
                    residual=plot_payload["residual"],
                    combined_boundary=plot_payload["combined_boundary"],
                    global_valid=plot_payload["global_valid"],
                    output_dir=plot_case_dir,
                    slice_axis=slice_axis,
                    residual_limit=residual_limit,
                )
                case_result["paths"].update(plot_paths)
            except Exception as exc:  # pragma: no cover - depends on plotting backend
                case_result["warnings"].append(f"Plot generation skipped: {exc}")
        _save_json(case_result, join(case_dir, "case_report.json"))

    wide_rows = _build_wide_rows(long_rows)
    summary = _build_summary(
        case_results,
        ref_only_cases,
        pred_only_cases,
        missing_tissue_cases,
        expected_case_count=expected_case_count,
        predicted_case_count=predicted_case_count,
        evaluated_case_count=evaluated_case_count,
    )

    long_fields = ["Case", "Tissue", *LONG_METRIC_KEYS, "Status"]
    boundary_fields = [
        "Case",
        "Boundary_MAE",
        "Boundary_RMSE",
        "Boundary_Acc_1",
        "Boundary_Acc_3",
        "Boundary_Acc_5",
        "Boundary_Gradient_MAE",
        "Boundary_Pearson_R",
        "BoundaryVoxelCount",
        "Interior_MAE",
        "Interior_RMSE",
        "Status",
    ]
    residual_fields = [
        "Case",
        "Residual_Mean",
        "Residual_Median",
        "Residual_STD",
        "Residual_Skewness",
        "Status",
    ]
    wide_fields = ["Case"]
    for tissue_name in TISSUE_ORDER:
        for key in LONG_METRIC_KEYS:
            wide_fields.append(f"{tissue_name}_{key}")
    wide_fields.append("Status")

    _write_csv(long_rows, join(summary_dir, "metrics_long.csv"), long_fields)
    _write_csv(wide_rows, join(summary_dir, "metrics_wide.csv"), wide_fields)
    _write_csv(boundary_rows, join(summary_dir, "boundary_metrics.csv"), boundary_fields)
    _write_csv(residual_rows, join(summary_dir, "residual_distribution.csv"), residual_fields)
    _save_json(summary, output_file)
    return summary
