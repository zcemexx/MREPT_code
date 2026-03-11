#!/usr/bin/env python3
"""
Normalize NIfTI geometry metadata to explicit identity affine.

This is useful when files were saved with invalid/empty qform+sform codes and
downstream tools infer different fallback world coordinates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        required=True,
        help="Input NIfTI file or directory.",
    )
    p.add_argument(
        "--pattern",
        default="*.nii.gz",
        help="Glob pattern when --input is a directory. Default: *.nii.gz",
    )
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input files in place.",
    )
    p.add_argument(
        "--suffix",
        default="_geomfix",
        help="Suffix for output files when not using --inplace.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without writing output.",
    )
    return p.parse_args()


def collect_files(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob(pattern) if p.is_file())
    raise FileNotFoundError(f"Input does not exist: {input_path}")


def output_path(src: Path, inplace: bool, suffix: str) -> Path:
    if inplace:
        return src
    name = src.name
    if name.endswith(".nii.gz"):
        return src.with_name(name[:-7] + suffix + ".nii.gz")
    if name.endswith(".nii"):
        return src.with_name(name[:-4] + suffix + ".nii")
    return src.with_name(name + suffix)


def rewrite_identity(src: Path, dst: Path) -> None:
    import nibabel as nib
    import numpy as np

    img = nib.load(str(src))
    data = np.asanyarray(img.dataobj)

    # Build diagonal affine from actual voxel sizes so zooms are preserved.
    # Using np.eye(4) would silently force 1 mm spacing for all axes.
    zooms = img.header.get_zooms()
    vox = [float(abs(z)) for z in zooms[:3]] if len(zooms) >= 3 else [1.0, 1.0, 1.0]
    aff = np.diag([*vox, 1.0]).astype(np.float64)

    out = nib.Nifti1Image(data, aff)
    out.set_qform(aff, code=1)
    out.set_sform(aff, code=1)
    nib.save(out, str(dst))


def main() -> None:
    args = parse_args()
    try:
        import nibabel  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: nibabel. Activate your nnUNet environment first, "
            "then rerun this script."
        ) from exc

    input_path = Path(args.input).expanduser().resolve()
    files = collect_files(input_path, args.pattern)
    if not files:
        print("No files matched.")
        return

    print(f"Matched files: {len(files)}")
    for src in files:
        dst = output_path(src, args.inplace, args.suffix)
        print(f"{src} -> {dst}")
        if not args.dry_run:
            rewrite_identity(src, dst)

    if args.dry_run:
        print("Dry run only: no files written.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
