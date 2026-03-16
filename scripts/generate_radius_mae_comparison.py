#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ismrm_figure_helpers import (
    DEFAULT_METRIC_ROOT,
    DEFAULT_OUTPUT_DIR,
    load_visualization_arrays,
    make_radius_mae_comparison_figure,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the radius-difference vs MAE comparison figure.")
    parser.add_argument("--metric-root", type=Path, default=DEFAULT_METRIC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    visualization = load_visualization_arrays(args.metric_root)
    outputs = save_figure(
        make_radius_mae_comparison_figure(visualization),
        args.output_dir,
        "ISMRM_RadiusDiff_vs_MAE_SNR10_50_150",
        args.formats,
        args.dpi,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
