#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
NNUNET_SRC = REPO_ROOT / "nnunet"
if str(NNUNET_SRC) not in sys.path:
    sys.path.insert(0, str(NNUNET_SRC))

from nnunetv2.utilities.ismrm_abstract_pipeline import (  # noqa: E402
    DEFAULT_INVIVO_ROOT,
    DEFAULT_MARGIN,
    crop_2d,
    first_existing_path,
    load_nifti_array,
    safe_crop_bounds_2d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the in-vivo 1x4 abstract panel.")
    parser.add_argument("--root", type=Path, default=DEFAULT_INVIVO_ROOT)
    parser.add_argument("--slice-index", type=int, default=50)
    parser.add_argument("--slice-axis", type=int, default=2)
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _extract_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    if index < 0 or index >= array.shape[axis]:
        raise IndexError(f"slice_index {index} out of bounds for axis {axis} with size {array.shape[axis]}")
    return np.take(array, indices=index, axis=axis)


def _compute_clim(values: list[np.ndarray], percentile_low: float = 1.0, percentile_high: float = 99.0) -> tuple[float, float]:
    finite_chunks = [v[np.isfinite(v)] for v in values if np.any(np.isfinite(v))]
    if not finite_chunks:
        return 0.0, 1.0
    merged = np.concatenate(finite_chunks)
    lo = float(np.percentile(merged, percentile_low))
    hi = float(np.percentile(merged, percentile_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        hi = lo + 1e-6
    return lo, hi


def main() -> int:
    args = parse_args()
    root = args.root
    output = args.output or (root / f"insight64_axial{args.slice_index:03d}_1x4.png")

    magnitude_path = first_existing_path((root / "magnitude.nii.gz", root / "magnitude.nii"))
    radius_path = root / "insight64.nii.gz"
    cond_fix_path = root / "cond_fixrad17_insight64.nii.gz"
    cond_pred_path = root / "cond_pred_insight64.nii.gz"
    mask_path = first_existing_path((root / "mask.nii.gz", root / "mask.nii", root / "segmentation.nii.gz"))

    magnitude = load_nifti_array(magnitude_path)
    radius = load_nifti_array(radius_path)
    cond_fix = load_nifti_array(cond_fix_path)
    cond_pred = load_nifti_array(cond_pred_path)
    mask = load_nifti_array(mask_path) > 0

    mask_slice = _extract_slice(mask, args.slice_axis, args.slice_index)
    bounds = safe_crop_bounds_2d(mask_slice, margin=args.margin)

    mag_slice = crop_2d(_extract_slice(magnitude, args.slice_axis, args.slice_index), bounds)
    radius_slice = crop_2d(_extract_slice(radius, args.slice_axis, args.slice_index), bounds)
    cond_fix_slice = crop_2d(_extract_slice(cond_fix, args.slice_axis, args.slice_index), bounds)
    cond_pred_slice = crop_2d(_extract_slice(cond_pred, args.slice_axis, args.slice_index), bounds)

    mag_clim = _compute_clim([mag_slice])
    radius_clim = _compute_clim([radius_slice])
    cond_clim = _compute_clim([cond_fix_slice, cond_pred_slice])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    panels = (
        ("Magnitude", mag_slice, "gray", mag_clim),
        ("CNN Radius", radius_slice, "magma", radius_clim),
        ("FixRad17", cond_fix_slice, "viridis", cond_clim),
        ("Cond Pred", cond_pred_slice, "viridis", cond_clim),
    )
    for ax, (title, data, cmap, clim) in zip(axes, panels):
        image = ax.imshow(data, cmap=cmap, vmin=clim[0], vmax=clim[1])
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
