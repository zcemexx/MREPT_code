#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
NNUNET_SRC = REPO_ROOT / "nnunet"
if str(NNUNET_SRC) not in sys.path:
    sys.path.insert(0, str(NNUNET_SRC))

from nnunetv2.utilities.ismrm_abstract_pipeline import (  # noqa: E402
    DEFAULT_CASE_IDS,
    DEFAULT_CNN_RADIUS_ROOT,
    DEFAULT_METRIC_ROOT,
    DEFAULT_SIM_ROOT,
    DEFAULT_SNRS,
    METHOD_ORDER,
    REGION_ORDER,
    build_region_masks,
    fixed_radius_array,
    iter_case_snr_pairs,
    load_simulation_arrays,
    masked_accuracy,
    masked_mae,
    masked_rmse,
    masked_slicewise_ssim,
    resolve_simulation_case_paths,
    snr_to_tag,
    summarize_missing,
    update_summary,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract conductivity/radius quantitative metrics for ISMRM figures.")
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--cnn-radius-root", type=Path, default=DEFAULT_CNN_RADIUS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METRIC_ROOT)
    parser.add_argument("--num-processes", type=int, default=None)
    return parser.parse_args()


def _cond_map_for_method(method: str, arrays: dict[str, object]) -> object:
    if method == "Fixed":
        return arrays["fixed_cond"]
    if method == "CNN":
        return arrays["cnn_cond"]
    if method == "Oracle":
        return arrays["oracle_cond"]
    raise KeyError(method)


def _radius_map_for_method(method: str, arrays: dict[str, object]) -> object:
    if method == "Fixed":
        return fixed_radius_array(arrays["oracle_radius"].shape)
    if method == "CNN":
        return arrays["cnn_radius"]
    if method == "Oracle":
        return arrays["oracle_radius"]
    raise KeyError(method)


def _default_num_processes() -> int:
    env_nslots = os.environ.get("NSLOTS", "").strip()
    if env_nslots.isdigit() and int(env_nslots) > 0:
        return int(env_nslots)
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count)


def _compute_case_snr_rows(
    case_id: str,
    snr: int,
    sim_root: Path,
    cnn_radius_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    paths = resolve_simulation_case_paths(case_id, snr, sim_root=sim_root, cnn_radius_root=cnn_radius_root)
    missing = summarize_missing(paths)
    if missing:
        return [], {"case_id": case_id, "snr": snr, "missing": missing}

    arrays = load_simulation_arrays(paths)
    region_masks = build_region_masks(arrays["tissue_mask"])
    bbox_mask = region_masks["Global"]
    gt_cond = arrays["gt_cond"]
    oracle_radius = arrays["oracle_radius"]

    rows = []
    for method in METHOD_ORDER:
        cond_map = _cond_map_for_method(method, arrays)
        radius_map = _radius_map_for_method(method, arrays)
        for region in REGION_ORDER:
            region_mask = region_masks[region]
            rows.append(
                {
                    "Case_ID": case_id,
                    "SNR": snr,
                    "Method": method,
                    "Region": region,
                    "MAE": masked_mae(cond_map, gt_cond, region_mask),
                    "SSIM": masked_slicewise_ssim(cond_map, gt_cond, region_mask, bbox_mask=bbox_mask),
                    "RMSE": masked_rmse(cond_map, gt_cond, region_mask),
                    "Acc_1": masked_accuracy(radius_map, oracle_radius, region_mask, tolerance=1),
                    "Acc_3": masked_accuracy(radius_map, oracle_radius, region_mask, tolerance=3),
                    "Acc_5": masked_accuracy(radius_map, oracle_radius, region_mask, tolerance=5),
                }
            )
    return rows, None


def main() -> int:
    args = parse_args()
    output_csv = args.output_dir / "quantitative_results.csv"
    summary_path = args.output_dir / "summary.json"
    num_processes = max(1, int(args.num_processes or _default_num_processes()))

    rows = []
    complete_cases = []
    skipped_cases = []
    tasks = [(case_id, snr) for case_id, snr in iter_case_snr_pairs(DEFAULT_CASE_IDS, DEFAULT_SNRS)]

    if num_processes == 1:
        for case_id, snr in tasks:
            case_rows, skipped = _compute_case_snr_rows(case_id, snr, args.sim_root, args.cnn_radius_root)
            if skipped is not None:
                skipped_cases.append(skipped)
                continue
            rows.extend(case_rows)
            complete_cases.append({"case_id": case_id, "snr": snr})
            print(f"[DONE] {case_id} {snr_to_tag(snr)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_map = {
                executor.submit(_compute_case_snr_rows, case_id, snr, args.sim_root, args.cnn_radius_root): (case_id, snr)
                for case_id, snr in tasks
            }
            for future in as_completed(future_map):
                case_id, snr = future_map[future]
                case_rows, skipped = future.result()
                if skipped is not None:
                    skipped_cases.append(skipped)
                    print(f"[SKIP] {case_id} {snr_to_tag(snr)} missing={','.join(skipped['missing'])}", flush=True)
                    continue
                rows.extend(case_rows)
                complete_cases.append({"case_id": case_id, "snr": snr})
                print(f"[DONE] {case_id} {snr_to_tag(snr)}", flush=True)

    rows.sort(key=lambda row: (DEFAULT_CASE_IDS.index(str(row["Case_ID"])), int(row["SNR"]), METHOD_ORDER.index(str(row["Method"])), REGION_ORDER.index(str(row["Region"]))))
    complete_cases.sort(key=lambda item: (DEFAULT_CASE_IDS.index(str(item["case_id"])), int(item["snr"])))
    skipped_cases.sort(key=lambda item: (DEFAULT_CASE_IDS.index(str(item["case_id"])), int(item["snr"])))

    write_csv(
        output_csv,
        ("Case_ID", "SNR", "Method", "Region", "MAE", "SSIM", "RMSE", "Acc_1", "Acc_3", "Acc_5"),
        rows,
    )
    update_summary(
        summary_path,
        "quantitative_metrics",
        {
            "output_csv": str(output_csv),
            "row_count": len(rows),
            "expected_row_count": len(DEFAULT_CASE_IDS) * len(DEFAULT_SNRS) * len(METHOD_ORDER) * len(REGION_ORDER),
            "num_processes": num_processes,
            "completed_case_snr_count": len(complete_cases),
            "skipped_case_snr_count": len(skipped_cases),
            "skipped_case_snrs": skipped_cases,
            "oracle_accuracy_contract_holds": all(
                math.isclose(float(row["Acc_1"]), 1.0)
                and math.isclose(float(row["Acc_3"]), 1.0)
                and math.isclose(float(row["Acc_5"]), 1.0)
                for row in rows
                if row["Method"] == "Oracle" and not math.isnan(float(row["Acc_1"]))
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
