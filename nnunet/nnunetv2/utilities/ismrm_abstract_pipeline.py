from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity


DEFAULT_CASE_IDS = (
    "M6",
    "M8",
    "M12",
    "M19",
    "M22",
    "M24",
    "M39",
    "M40",
    "M41",
    "M42",
    "M43",
    "M50",
    "M66",
    "M70",
    "M75",
    "M79",
    "M84",
)
DEFAULT_SNRS = (10, 20, 30, 40, 50, 75, 100, 150)
SELECTED_DIST_SNRS = (10, 50, 150)
CORRELATION_CASE_IDS = ("M6", "M12")
DIFF_CASE_IDS = ("M6", "M8", "M12")

DEFAULT_SIM_ROOT = Path("/home/zcemexx/Scratch/outputs/nii")
DEFAULT_CNN_RADIUS_ROOT = Path("/home/zcemexx/Scratch/data/maerm")
DEFAULT_METRIC_ROOT = Path("/home/zcemexx/Scratch/outputs/metric")
DEFAULT_TRAINING_ROOT = Path(
    "/myriadfs/home/zcemexx/Scratch/nnUNet_results/"
    "Dataset001_EPT/nnUNetTrainerMRCT_mae_regfix__nnResUNetPlans__3d_fullres"
)
DEFAULT_INVIVO_ROOT = Path("/Users/apple/Documents/mresult/tmp/invivo/insight64")

REGION_ORDER = ("Global", "WM", "GM", "CSF")
REGION_TO_LABEL = {"WM": 1, "GM": 2, "CSF": 3}
METHOD_ORDER = ("Fixed", "CNN", "Oracle")
FIXED_RADIUS_VALUE = 17.0
DEFAULT_MARGIN = 4

TRAINING_LOG_PATTERN = re.compile(r"Epoch\s+(\d+)")
TRAIN_LOSS_PATTERN = re.compile(r"train_loss\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
VAL_LOSS_PATTERN = re.compile(r"val_loss\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


@dataclass(frozen=True)
class SimulationCasePaths:
    case_id: str
    snr: int
    snr_tag: str
    base_dir: Path
    gt_cond: Path
    fixed_cond: Path
    oracle_cond: Path
    cnn_cond: Path
    tissue_mask: Path
    oracle_radius: Path
    cnn_radius: Path


def snr_to_tag(snr: int) -> str:
    return f"SNR{int(snr):03d}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: MutableMapping | None = None) -> MutableMapping:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def update_summary(summary_path: Path, stage_name: str, payload: Mapping) -> None:
    summary = read_json(summary_path, default={})
    summary[stage_name] = payload
    write_json(summary_path, summary)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_nifti_array(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI at {path}, got shape {data.shape}")
    return data


def write_identity_nifti(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    data = np.asarray(array, dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4, dtype=np.float32))
    img.set_qform(np.eye(4, dtype=np.float32), code=1)
    img.set_sform(np.eye(4, dtype=np.float32), code=1)
    nib.save(img, str(path))


def build_region_masks(tissue_mask: np.ndarray) -> dict[str, np.ndarray]:
    tissue_mask = np.asarray(tissue_mask)
    if tissue_mask.ndim != 3:
        raise ValueError(f"tissue_mask must be 3D, got {tissue_mask.shape}")
    masks = {"Global": tissue_mask > 0}
    for region_name, label in REGION_TO_LABEL.items():
        masks[region_name] = tissue_mask == label
    return masks


def safe_crop_bounds_2d(mask_2d: np.ndarray, margin: int = DEFAULT_MARGIN) -> tuple[int, int, int, int]:
    mask_2d = np.asarray(mask_2d, dtype=bool)
    if mask_2d.ndim != 2:
        raise ValueError(f"mask_2d must be 2D, got {mask_2d.shape}")
    coords = np.argwhere(mask_2d)
    if coords.size == 0:
        raise ValueError("Cannot crop an empty 2D mask.")
    y_min_raw, x_min_raw = coords.min(axis=0)
    y_max_raw, x_max_raw = coords.max(axis=0)
    height, width = mask_2d.shape
    y_min = max(0, int(y_min_raw) - margin)
    y_max = min(height, int(y_max_raw) + margin + 1)
    x_min = max(0, int(x_min_raw) - margin)
    x_max = min(width, int(x_max_raw) + margin + 1)
    return y_min, y_max, x_min, x_max


def crop_2d(array_2d: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    y_min, y_max, x_min, x_max = bounds
    return np.asarray(array_2d)[y_min:y_max, x_min:x_max]


def resolve_simulation_case_paths(
    case_id: str,
    snr: int,
    sim_root: Path = DEFAULT_SIM_ROOT,
    cnn_radius_root: Path = DEFAULT_CNN_RADIUS_ROOT,
) -> SimulationCasePaths:
    snr_tag = snr_to_tag(snr)
    base_dir = sim_root / case_id / snr_tag
    oracle_radius = base_dir / "label_final.nii.gz"
    if not oracle_radius.exists():
        fallback = base_dir / "pred_kernel_map_label_final.nii.gz"
        if fallback.exists():
            oracle_radius = fallback
    return SimulationCasePaths(
        case_id=case_id,
        snr=snr,
        snr_tag=snr_tag,
        base_dir=base_dir,
        gt_cond=base_dir / "Conductivity_GT.nii.gz",
        fixed_cond=base_dir / "cond_fixr17.nii.gz",
        oracle_cond=base_dir / "cond_optimal.nii.gz",
        cnn_cond=base_dir / "cond_pred.nii.gz",
        tissue_mask=base_dir / "tissueMask.nii.gz",
        oracle_radius=oracle_radius,
        cnn_radius=cnn_radius_root / f"{case_id}_{snr_tag}.nii.gz",
    )


def find_training_log(fold_dir: Path) -> Path:
    candidates = sorted(fold_dir.glob("training_log_*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No training_log_*.txt found in {fold_dir}")
    return candidates[-1]


def parse_training_log(log_path: Path, fold: int) -> List[dict[str, object]]:
    rows_by_epoch: dict[int, dict[str, object]] = {}
    current_epoch: int | None = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            epoch_match = TRAINING_LOG_PATTERN.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                rows_by_epoch.setdefault(
                    current_epoch,
                    {"Epoch": current_epoch, "Fold": fold, "Train_Loss": math.nan, "Val_Loss": math.nan},
                )
                continue
            if current_epoch is None:
                continue
            train_match = TRAIN_LOSS_PATTERN.search(line)
            if train_match:
                rows_by_epoch[current_epoch]["Train_Loss"] = float(train_match.group(1))
                continue
            val_match = VAL_LOSS_PATTERN.search(line)
            if val_match:
                rows_by_epoch[current_epoch]["Val_Loss"] = float(val_match.group(1))
    return [rows_by_epoch[key] for key in sorted(rows_by_epoch)]


def _largest_valid_odd_window(min_dim: int) -> int | None:
    if min_dim < 3:
        return None
    if min_dim >= 7:
        return None
    return min_dim if min_dim % 2 == 1 else min_dim - 1


def masked_slicewise_ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    eval_mask: np.ndarray,
    bbox_mask: np.ndarray,
    axis: int = 2,
) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    bbox_mask = np.asarray(bbox_mask, dtype=bool)
    if pred.shape != gt.shape or pred.shape != eval_mask.shape or pred.shape != bbox_mask.shape:
        raise ValueError(
            "pred, gt, eval_mask and bbox_mask must have identical shapes: "
            f"{pred.shape}, {gt.shape}, {eval_mask.shape}, {bbox_mask.shape}"
        )

    pred_slices = np.moveaxis(pred, axis, -1)
    gt_slices = np.moveaxis(gt, axis, -1)
    eval_slices = np.moveaxis(eval_mask, axis, -1)
    bbox_slices = np.moveaxis(bbox_mask, axis, -1)

    weighted_sum = 0.0
    weight_total = 0
    for slice_idx in range(pred_slices.shape[-1]):
        pred_slice = pred_slices[..., slice_idx]
        gt_slice = gt_slices[..., slice_idx]
        eval_slice = eval_slices[..., slice_idx]
        bbox_slice = bbox_slices[..., slice_idx]

        finite_slice = np.isfinite(pred_slice) & np.isfinite(gt_slice)
        eval_valid = eval_slice & finite_slice
        bbox_valid = bbox_slice & finite_slice
        if not np.any(eval_valid) or not np.any(bbox_valid):
            continue

        y_min, y_max, x_min, x_max = safe_crop_bounds_2d(bbox_valid, margin=0)
        pred_bbox = pred_slice[y_min:y_max, x_min:x_max]
        gt_bbox = gt_slice[y_min:y_max, x_min:x_max]
        eval_bbox = eval_valid[y_min:y_max, x_min:x_max]
        if not np.any(eval_bbox):
            continue

        min_dim = min(pred_bbox.shape)
        win_size = _largest_valid_odd_window(min_dim)
        if min_dim < 3:
            continue

        gt_values = gt_bbox[eval_bbox]
        data_range = float(gt_values.max() - gt_values.min())
        if not np.isfinite(data_range) or data_range <= 0:
            continue

        _, ssim_map = structural_similarity(
            pred_bbox,
            gt_bbox,
            data_range=data_range,
            full=True,
            win_size=win_size,
        )
        slice_weight = int(np.count_nonzero(eval_bbox))
        weighted_sum += float(np.mean(ssim_map[eval_bbox])) * slice_weight
        weight_total += slice_weight

    if weight_total == 0:
        return math.nan
    return weighted_sum / weight_total


def masked_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(gt)
    if not np.any(valid):
        return math.nan
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float32)[valid] - np.asarray(gt, dtype=np.float32)[valid])))


def masked_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(gt)
    if not np.any(valid):
        return math.nan
    delta = np.asarray(pred, dtype=np.float32)[valid] - np.asarray(gt, dtype=np.float32)[valid]
    return float(np.sqrt(np.mean(delta ** 2)))


def masked_accuracy(
    pred_radius: np.ndarray,
    gt_radius: np.ndarray,
    mask: np.ndarray,
    tolerance: float,
) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred_radius) & np.isfinite(gt_radius)
    if not np.any(valid):
        return math.nan
    delta = np.abs(np.asarray(pred_radius, dtype=np.float32)[valid] - np.asarray(gt_radius, dtype=np.float32)[valid])
    return float(np.mean(delta <= float(tolerance)))


def summarize_missing(paths: SimulationCasePaths) -> List[str]:
    missing = []
    for label, path in (
        ("gt_cond", paths.gt_cond),
        ("fixed_cond", paths.fixed_cond),
        ("oracle_cond", paths.oracle_cond),
        ("cnn_cond", paths.cnn_cond),
        ("tissue_mask", paths.tissue_mask),
        ("oracle_radius", paths.oracle_radius),
        ("cnn_radius", paths.cnn_radius),
    ):
        if not path.exists():
            missing.append(label)
    return missing


def validate_same_shape(named_arrays: Mapping[str, np.ndarray]) -> tuple[int, int, int]:
    shapes = {name: np.asarray(array).shape for name, array in named_arrays.items()}
    unique_shapes = {shape for shape in shapes.values()}
    if len(unique_shapes) != 1:
        raise ValueError(f"Shape mismatch: {shapes}")
    return next(iter(unique_shapes))


def region_name_array(tissue_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    tissue_mask = np.asarray(tissue_mask)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    labels = np.full(valid_mask.shape, "Global", dtype=object)
    labels[tissue_mask == 1] = "WM"
    labels[tissue_mask == 2] = "GM"
    labels[tissue_mask == 3] = "CSF"
    return labels[valid_mask]


def load_simulation_arrays(paths: SimulationCasePaths) -> dict[str, np.ndarray]:
    arrays = {
        "gt_cond": load_nifti_array(paths.gt_cond),
        "fixed_cond": load_nifti_array(paths.fixed_cond),
        "oracle_cond": load_nifti_array(paths.oracle_cond),
        "cnn_cond": load_nifti_array(paths.cnn_cond),
        "tissue_mask": load_nifti_array(paths.tissue_mask),
        "oracle_radius": load_nifti_array(paths.oracle_radius),
        "cnn_radius": load_nifti_array(paths.cnn_radius),
    }
    validate_same_shape(arrays)
    return arrays


def fixed_radius_array(shape: Sequence[int], value: float = FIXED_RADIUS_VALUE) -> np.ndarray:
    return np.full(tuple(shape), float(value), dtype=np.float32)


def save_residual_archive(output_path: Path, residuals: Mapping[str, np.ndarray]) -> None:
    ensure_dir(output_path.parent)
    normalized = {key: np.asarray(value, dtype=np.float32) for key, value in residuals.items()}
    np.savez_compressed(output_path, **normalized)


def is_npz_compressed(path: Path) -> bool:
    with zipfile.ZipFile(path, "r") as archive:
        return all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist())


def first_existing_path(candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of the candidate paths exist: " + ", ".join(str(p) for p in candidates))


def iter_case_snr_pairs(
    case_ids: Sequence[str] = DEFAULT_CASE_IDS,
    snrs: Sequence[int] = DEFAULT_SNRS,
) -> Iterator[tuple[str, int]]:
    for case_id in case_ids:
        for snr in snrs:
            yield case_id, int(snr)
