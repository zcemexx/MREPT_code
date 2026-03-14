#!/usr/bin/env python3
"""Remap integer labels in a NIfTI file while preserving geometry/header."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input NIfTI path.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output NIfTI path. Use a new file unless you are sure about overwriting.",
    )
    parser.add_argument(
        "--map",
        nargs="+",
        required=True,
        metavar="SRC:DST",
        help="Label mapping pairs, for example: 1:2 2:1",
    )
    return parser.parse_args()


def parse_mapping(items: list[str]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid mapping '{item}'. Expected SRC:DST.")
        src_text, dst_text = item.split(":", 1)
        src = int(src_text)
        dst = int(dst_text)
        mapping[src] = dst
    return mapping


def main() -> None:
    args = parse_args()
    mapping = parse_mapping(args.map)

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(src))
    header = img.header.copy()
    data = np.asanyarray(img.dataobj)

    out = np.array(data, copy=True)
    for src_label, dst_label in mapping.items():
        out[data == src_label] = dst_label

    remapped = nib.Nifti1Image(out.astype(data.dtype, copy=False), img.affine, header=header)
    qform, qcode = img.get_qform(coded=True)
    sform, scode = img.get_sform(coded=True)
    remapped.set_qform(qform, int(qcode))
    remapped.set_sform(sform, int(scode))
    nib.save(remapped, str(dst))

    old_labels = np.unique(data)
    new_labels = np.unique(out)
    print(f"input={src}")
    print(f"output={dst}")
    print(f"mapping={mapping}")
    print(f"labels_before={old_labels.tolist()}")
    print(f"labels_after={new_labels.tolist()}")


if __name__ == "__main__":
    main()
