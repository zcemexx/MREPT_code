#!/usr/bin/env python3
"""
Normalize NIfTI geometry metadata to explicit axis-aligned affine.

Goals:
1) Robust file discovery in Python (avoid shell glob pitfalls).
2) Idempotent rewrite behavior for repeated runs.
3) Safe handling of 3D/4D/5D headers by touching only spatial geometry.

This tool rewrites header geometry only; voxel index layout is unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

EXIT_GENERAL_ERROR = 1
EXIT_UNDEFINED_FOUND = 2
EXIT_NO_FILES_MATCHED = 3

VALID_ACTIONS = ("report", "fix-undefined", "fix-all")


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
        "--action",
        default="report",
        choices=VALID_ACTIONS,
        help=(
            "report: inspect only; "
            "fix-undefined: rewrite only qform=0 and sform=0 files; "
            "fix-all: rewrite all matched files."
        ),
    )
    p.add_argument(
        "--fail-on-undefined",
        action="store_true",
        help="When action=report, return non-zero if undefined geometry exists.",
    )
    p.add_argument(
        "--fail-on-no-match",
        action="store_true",
        help="Return non-zero if no file matches input/pattern.",
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


def read_qsform_codes(src: Path) -> tuple[int, int]:
    import nibabel as nib

    img = nib.load(str(src))
    qcode = int(img.header["qform_code"])
    scode = int(img.header["sform_code"])
    return qcode, scode


def spatial_affine_and_zooms(header, ndim: int):
    import numpy as np

    old_zooms = tuple(float(z) for z in header.get_zooms())
    spatial = [1.0, 1.0, 1.0]
    for i in range(min(3, len(old_zooms))):
        z = float(old_zooms[i])
        if not np.isfinite(z) or z == 0:
            z = 1.0
        spatial[i] = abs(z)

    aff = np.eye(4, dtype=np.float64)
    aff[0, 0] = spatial[0]
    aff[1, 1] = spatial[1]
    aff[2, 2] = spatial[2]

    # Keep non-spatial zoom metadata (for example TR in 4D files).
    new_zooms = list(old_zooms)
    if len(new_zooms) < ndim:
        new_zooms.extend([1.0] * (ndim - len(new_zooms)))
    for i in range(min(3, len(new_zooms))):
        new_zooms[i] = spatial[i]

    return aff, tuple(new_zooms[:ndim])


def rewrite_identity(src: Path, dst: Path) -> None:
    import nibabel as nib

    img = nib.load(str(src))
    hdr = img.header.copy()

    # Prefer raw on-disk values for strong idempotency of storage behavior.
    proxy = img.dataobj
    if hasattr(proxy, "get_unscaled"):
        data = proxy.get_unscaled()
    else:
        data = proxy[:]

    aff, zooms = spatial_affine_and_zooms(hdr, img.ndim)

    out = nib.Nifti1Image(data, aff, header=hdr)
    out.set_data_dtype(hdr.get_data_dtype())

    slope, inter = hdr.get_slope_inter()
    if slope is not None or inter is not None:
        out.header.set_slope_inter(slope, inter)

    out.header.set_zooms(zooms)
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
        print(f"No files to check (input={input_path}, pattern={args.pattern}).")
        if args.fail_on_no_match:
            raise SystemExit(EXIT_NO_FILES_MATCHED)
        return

    print(f"Matched files: {len(files)}")
    print(f"Action: {args.action}")

    undefined_count = 0
    rewritten_count = 0

    for src in files:
        qcode, scode = read_qsform_codes(src)
        undefined = qcode == 0 and scode == 0
        if undefined:
            undefined_count += 1

        dst = output_path(src, args.inplace, args.suffix)

        if args.action == "report":
            status = "UNDEFINED" if undefined else "OK"
            print(f"{src} | qform={qcode} sform={scode} | {status}")
            continue

        should_rewrite = args.action == "fix-all" or (args.action == "fix-undefined" and undefined)
        if should_rewrite:
            rewritten_count += 1
            print(f"{src} -> {dst} | qform={qcode} sform={scode} | REWRITE")
            if not args.dry_run:
                rewrite_identity(src, dst)
        else:
            print(f"{src} | qform={qcode} sform={scode} | SKIP")

    print(
        f"Summary: total={len(files)} undefined={undefined_count} rewritten={rewritten_count}"
    )

    if args.dry_run:
        print("Dry run only: no files written.")
    elif args.action != "report":
        print("Done.")

    if args.action == "report" and args.fail_on_undefined and undefined_count > 0:
        raise SystemExit(EXIT_UNDEFINED_FOUND)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(EXIT_GENERAL_ERROR)
