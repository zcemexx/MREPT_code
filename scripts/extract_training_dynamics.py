#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
NNUNET_SRC = REPO_ROOT / "nnunet"
if str(NNUNET_SRC) not in sys.path:
    sys.path.insert(0, str(NNUNET_SRC))

from nnunetv2.utilities.ismrm_abstract_pipeline import (  # noqa: E402
    DEFAULT_METRIC_ROOT,
    DEFAULT_TRAINING_ROOT,
    find_training_log,
    parse_training_log,
    update_summary,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 5-fold nnU-Net training losses into one CSV.")
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METRIC_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_csv = output_dir / "training_losses.csv"
    summary_path = output_dir / "summary.json"

    all_rows = []
    fold_summaries = []
    for fold in range(5):
        fold_dir = args.training_root / f"fold_{fold}"
        log_path = find_training_log(fold_dir)
        rows = parse_training_log(log_path, fold)
        all_rows.extend(rows)
        fold_summaries.append(
            {
                "fold": fold,
                "fold_dir": str(fold_dir),
                "log_path": str(log_path),
                "epochs_found": len(rows),
                "has_progress_png": (fold_dir / "progress.png").exists(),
            }
        )

    write_csv(
        output_csv,
        ("Epoch", "Fold", "Train_Loss", "Val_Loss"),
        all_rows,
    )
    update_summary(
        summary_path,
        "training_dynamics",
        {
            "output_csv": str(output_csv),
            "row_count": len(all_rows),
            "folds": fold_summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
