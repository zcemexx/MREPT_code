from __future__ import annotations

import argparse

from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.regression_tissue_metrics import compute_regression_metrics_on_folder_with_tissues
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_file_ending


def evaluate_regression_tissue_predictions_entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-ref", required=True, type=str, help="folder with GT radius maps (labelsTr)")
    parser.add_argument("--folder-pred", required=True, type=str, help="folder with predicted radius maps")
    parser.add_argument("--folder-images", required=True, type=str, help="folder with image channels (imagesTr)")
    parser.add_argument("--output-json", required=True, type=str, help="output summary JSON path")
    parser.add_argument("--plots-dir", required=False, default=None, type=str, help="optional directory for PNG plots")
    parser.add_argument("--file-ending", required=False, default=".nii.gz", type=str, help="file ending to evaluate")
    parser.add_argument(
        "--tissue-channel-suffix",
        required=False,
        default="_0001",
        type=str,
        help="tissue mask channel suffix, default: _0001",
    )
    parser.add_argument(
        "--num-processes",
        required=False,
        default=default_num_processes,
        type=int,
        help=f"number of worker processes, default: {default_num_processes}",
    )
    parser.add_argument("--disable-gradient-mae", action="store_true", help="disable Gradient_MAE computation")
    parser.add_argument("--disable-pearson-r", action="store_true", help="disable Pearson_R computation")
    parser.add_argument("--boundary-width", required=False, default=1, type=int, help="boundary width in voxels")
    parser.add_argument("--slice-axis", required=False, default=2, type=int, help="slice axis for plots")
    parser.add_argument("--residual-limit", required=False, default=5, type=int, help="clipping limit for residual histogram")
    args = parser.parse_args()

    image_reader_writer = determine_reader_writer_from_file_ending(
        args.file_ending,
        example_file=None,
        allow_nonmatching_filename=False,
        verbose=False,
    )()

    summary = compute_regression_metrics_on_folder_with_tissues(
        folder_ref=args.folder_ref,
        folder_pred=args.folder_pred,
        folder_images=args.folder_images,
        output_file=args.output_json,
        image_reader_writer=image_reader_writer,
        file_ending=args.file_ending,
        tissue_channel_suffix=args.tissue_channel_suffix,
        num_processes=args.num_processes,
        include_gradient_mae=not args.disable_gradient_mae,
        include_pearson_r=not args.disable_pearson_r,
        plots_dir=args.plots_dir,
        boundary_width=args.boundary_width,
        residual_limit=args.residual_limit,
        slice_axis=args.slice_axis,
    )
    case_counts = summary.get("case_counts", {})
    print(
        "Regression tissue eval case counts: "
        f"evaluated={case_counts.get('evaluated_cases', 0)}, "
        f"missing_tissue={len(case_counts.get('missing_tissue_cases', []))}, "
        f"ref_only={len(case_counts.get('ref_only_cases', []))}, "
        f"pred_only={len(case_counts.get('pred_only_cases', []))}"
    )

    ok_cases = summary["case_counts"]["status_counts"].get("ok", 0)
    warning_cases = summary["case_counts"]["status_counts"].get("warning_alignment", 0)
    if ok_cases + warning_cases == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    evaluate_regression_tissue_predictions_entry_point()
