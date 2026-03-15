#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ismrm_figure_helpers import (
    DEFAULT_METRIC_ROOT,
    DEFAULT_OUTPUT_DIR,
    load_quantitative_data,
    load_visualization_arrays,
    make_fig1,
    make_fig2,
    make_fig3,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the three ISMRM abstract figures.")
    parser.add_argument("--metric-root", type=Path, default=DEFAULT_METRIC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quantitative = load_quantitative_data(args.metric_root)
    visualization = load_visualization_arrays(args.metric_root)

    outputs = []
    outputs.extend(
        save_figure(
            make_fig1(quantitative),
            args.output_dir,
            "ISMRM_Fig1_Quantitative",
            args.formats,
            args.dpi,
        )
    )
    outputs.extend(
        save_figure(
            make_fig2(visualization),
            args.output_dir,
            "ISMRM_Fig2_Simulation_6x4",
            args.formats,
            args.dpi,
        )
    )
    outputs.extend(
        save_figure(
            make_fig3(visualization),
            args.output_dir,
            "ISMRM_Fig3_Simulation_4x4",
            args.formats,
            args.dpi,
        )
    )

    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
