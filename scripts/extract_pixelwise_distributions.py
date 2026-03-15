#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
NNUNET_SRC = REPO_ROOT / "nnunet"
if str(NNUNET_SRC) not in sys.path:
    sys.path.insert(0, str(NNUNET_SRC))

from nnunetv2.utilities.ismrm_abstract_pipeline import (  # noqa: E402
    CORRELATION_CASE_IDS,
    DEFAULT_CASE_IDS,
    DEFAULT_CNN_RADIUS_ROOT,
    DEFAULT_METRIC_ROOT,
    DEFAULT_SIM_ROOT,
    DIFF_CASE_IDS,
    SELECTED_DIST_SNRS,
    build_region_masks,
    load_simulation_arrays,
    region_name_array,
    resolve_simulation_case_paths,
    save_residual_archive,
    summarize_missing,
    update_summary,
    write_identity_nifti,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pixel-wise residual distributions and correlation data.")
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--cnn-radius-root", type=Path, default=DEFAULT_CNN_RADIUS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METRIC_ROOT)
    return parser.parse_args()


def _append_residual(bucket: dict[str, list[np.ndarray]], key: str, values: np.ndarray) -> None:
    bucket.setdefault(key, []).append(np.asarray(values, dtype=np.float32))


def main() -> int:
    args = parse_args()
    summary_path = args.output_dir / "summary.json"
    residual_output = args.output_dir / "residuals_dist.npz"
    correlation_output = args.output_dir / "correlation_data.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    residual_buckets: dict[str, list[np.ndarray]] = {}
    residual_stage_summary = []
    correlation_rows_written = 0
    skipped_case_snrs = []
    diff_outputs: list[str] = []

    with correlation_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Case_ID", "SNR", "Delta_Radius", "Delta_Sigma", "Region"))
        writer.writeheader()

        for case_id in DEFAULT_CASE_IDS:
            snrs_to_visit = SELECTED_DIST_SNRS
            for snr in snrs_to_visit:
                paths = resolve_simulation_case_paths(case_id, snr, sim_root=args.sim_root, cnn_radius_root=args.cnn_radius_root)
                missing = summarize_missing(paths)
                if missing:
                    skipped_case_snrs.append({"case_id": case_id, "snr": snr, "missing": missing})
                    continue

                arrays = load_simulation_arrays(paths)
                global_mask = build_region_masks(arrays["tissue_mask"])["Global"]
                finite_global = global_mask & np.isfinite(arrays["gt_cond"])

                if snr in SELECTED_DIST_SNRS:
                    for method_name, cond_key in (("Fixed", "fixed_cond"), ("CNN", "cnn_cond"), ("Oracle", "oracle_cond")):
                        valid = finite_global & np.isfinite(arrays[cond_key])
                        residual = (arrays[cond_key] - arrays["gt_cond"])[valid].astype(np.float32)
                        _append_residual(residual_buckets, f"SNR_{snr:03d}_{method_name}", residual)

                if case_id in CORRELATION_CASE_IDS and snr == 50:
                    valid = (
                        global_mask
                        & np.isfinite(arrays["cnn_cond"])
                        & np.isfinite(arrays["gt_cond"])
                        & np.isfinite(arrays["cnn_radius"])
                        & np.isfinite(arrays["oracle_radius"])
                    )
                    if np.any(valid):
                        delta_radius = (arrays["cnn_radius"] - arrays["oracle_radius"])[valid].astype(np.float32)
                        delta_sigma = (arrays["cnn_cond"] - arrays["gt_cond"])[valid].astype(np.float32)
                        regions = region_name_array(arrays["tissue_mask"], valid)
                        for dr, ds, region_name in zip(delta_radius, delta_sigma, regions):
                            writer.writerow(
                                {
                                    "Case_ID": case_id,
                                    "SNR": snr,
                                    "Delta_Radius": float(dr),
                                    "Delta_Sigma": float(ds),
                                    "Region": region_name,
                                }
                            )
                            correlation_rows_written += 1

                if case_id in DIFF_CASE_IDS and snr in SELECTED_DIST_SNRS:
                    delta_radius = (arrays["cnn_radius"] - arrays["oracle_radius"]).astype(np.float32)
                    delta_sigma = (arrays["cnn_cond"] - arrays["gt_cond"]).astype(np.float32)
                    radius_output = paths.base_dir / "delta_radius_cnn_vs_oracle.nii.gz"
                    sigma_output = paths.base_dir / "delta_sigma_cnn_vs_gt.nii.gz"
                    write_identity_nifti(radius_output, delta_radius)
                    write_identity_nifti(sigma_output, delta_sigma)
                    diff_outputs.extend((str(radius_output), str(sigma_output)))

    archive_payload = {key: np.concatenate(chunks).astype(np.float32) for key, chunks in sorted(residual_buckets.items()) if chunks}
    save_residual_archive(residual_output, archive_payload)

    for key, values in sorted(archive_payload.items()):
        residual_stage_summary.append({"key": key, "voxel_count": int(values.size), "dtype": str(values.dtype)})

    update_summary(
        summary_path,
        "pixelwise_distributions",
        {
            "residual_output": str(residual_output),
            "correlation_output": str(correlation_output),
            "residual_keys": residual_stage_summary,
            "correlation_rows_written": correlation_rows_written,
            "diff_outputs_written": diff_outputs,
            "skipped_case_snrs": skipped_case_snrs,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
