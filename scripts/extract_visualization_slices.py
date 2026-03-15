#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
NNUNET_SRC = REPO_ROOT / "nnunet"
if str(NNUNET_SRC) not in sys.path:
    sys.path.insert(0, str(NNUNET_SRC))

from nnunetv2.utilities.ismrm_abstract_pipeline import (  # noqa: E402
    DEFAULT_CNN_RADIUS_ROOT,
    DEFAULT_MARGIN,
    DEFAULT_METRIC_ROOT,
    DEFAULT_SIM_ROOT,
    SELECTED_DIST_SNRS,
    build_region_masks,
    crop_2d,
    load_simulation_arrays,
    resolve_simulation_case_paths,
    safe_crop_bounds_2d,
    update_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract cropped 2D matrices for ISMRM abstract figures.")
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--cnn-radius-root", type=Path, default=DEFAULT_CNN_RADIUS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METRIC_ROOT)
    parser.add_argument("--case-id", default="M12")
    parser.add_argument("--slice-index", type=int, default=80)
    parser.add_argument("--slice-axis", type=int, default=2)
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    return parser.parse_args()


def _extract_axial(array: np.ndarray, slice_axis: int, slice_index: int) -> np.ndarray:
    axis_len = array.shape[slice_axis]
    if slice_index < 0 or slice_index >= axis_len:
        raise IndexError(f"slice_index {slice_index} out of bounds for axis {slice_axis} with size {axis_len}")
    return np.take(array, indices=slice_index, axis=slice_axis)


def main() -> int:
    args = parse_args()
    output_path = args.output_dir / "visualization_slices.npz"
    summary_path = args.output_dir / "summary.json"

    try:
        reference_paths = resolve_simulation_case_paths(
            args.case_id,
            50,
            sim_root=args.sim_root,
            cnn_radius_root=args.cnn_radius_root,
        )
        reference_arrays = load_simulation_arrays(reference_paths)
        global_mask_slice = _extract_axial(build_region_masks(reference_arrays["tissue_mask"])["Global"], args.slice_axis, args.slice_index)
        bounds = safe_crop_bounds_2d(global_mask_slice, margin=args.margin)

        arrays_out: dict[str, np.ndarray] = {}
        for snr in SELECTED_DIST_SNRS:
            paths = resolve_simulation_case_paths(args.case_id, snr, sim_root=args.sim_root, cnn_radius_root=args.cnn_radius_root)
            arrays = load_simulation_arrays(paths)

            if snr == SELECTED_DIST_SNRS[0]:
                arrays_out["Simulation/GT_Cond"] = crop_2d(_extract_axial(arrays["gt_cond"], args.slice_axis, args.slice_index), bounds)

            arrays_out[f"Simulation/SNR_{snr:03d}/Fixed_Cond"] = crop_2d(
                _extract_axial(arrays["fixed_cond"], args.slice_axis, args.slice_index), bounds
            )
            arrays_out[f"Simulation/SNR_{snr:03d}/CNN_Cond"] = crop_2d(
                _extract_axial(arrays["cnn_cond"], args.slice_axis, args.slice_index), bounds
            )
            arrays_out[f"Simulation/SNR_{snr:03d}/Oracle_Cond"] = crop_2d(
                _extract_axial(arrays["oracle_cond"], args.slice_axis, args.slice_index), bounds
            )
            arrays_out[f"Simulation/SNR_{snr:03d}/CNN_Pred_Radius"] = crop_2d(
                _extract_axial(arrays["cnn_radius"], args.slice_axis, args.slice_index), bounds
            )

            diff_radius = arrays["cnn_radius"] - arrays["oracle_radius"]
            diff_cond = arrays["cnn_cond"] - arrays["gt_cond"]
            arrays_out[f"Difference_Maps/SNR_{snr:03d}/Diff_Radius"] = crop_2d(
                _extract_axial(diff_radius, args.slice_axis, args.slice_index), bounds
            )
            arrays_out[f"Difference_Maps/SNR_{snr:03d}/Diff_Cond"] = crop_2d(
                _extract_axial(diff_cond, args.slice_axis, args.slice_index), bounds
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **{key: np.asarray(value, dtype=np.float32) for key, value in arrays_out.items()})
        shape_set = sorted({tuple(value.shape) for value in arrays_out.values()})
        update_summary(
            summary_path,
            "visualization_slices",
            {
                "status": "ok",
                "output_npz": str(output_path),
                "case_id": args.case_id,
                "slice_axis": args.slice_axis,
                "slice_index": args.slice_index,
                "crop_bounds": {"y_min": bounds[0], "y_max": bounds[1], "x_min": bounds[2], "x_max": bounds[3]},
                "array_count": len(arrays_out),
                "array_shapes": shape_set,
            },
        )
        return 0
    except Exception as exc:
        update_summary(
            summary_path,
            "visualization_slices",
            {
                "status": "failed",
                "case_id": args.case_id,
                "slice_axis": args.slice_axis,
                "slice_index": args.slice_index,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
