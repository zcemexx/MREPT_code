#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

try:
    import nibabel as nib
except Exception as e:
    print(f"[FATAL] nibabel import failed: {e}")
    print("Install with: pip install nibabel")
    sys.exit(2)


def case_id_from_path(path: str, suffix: str) -> str:
    name = os.path.basename(path)
    if not name.endswith(suffix):
        return ""
    return name[: -len(suffix)]


def collect_cases(images_tr: str, labels_tr: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    ch0_paths = glob.glob(os.path.join(images_tr, "M*_0000.nii.gz"))
    ch1_paths = glob.glob(os.path.join(images_tr, "M*_0001.nii.gz"))
    lab_paths = glob.glob(os.path.join(labels_tr, "M*.nii.gz"))

    ch0 = {case_id_from_path(p, "_0000.nii.gz"): p for p in ch0_paths}
    ch1 = {case_id_from_path(p, "_0001.nii.gz"): p for p in ch1_paths}
    lab = {case_id_from_path(p, ".nii.gz"): p for p in lab_paths}

    ch0.pop("", None)
    ch1.pop("", None)
    lab.pop("", None)
    return ch0, ch1, lab


def check_one_case(case: str, p0: str, p1: str, pl: str, label_max: int) -> List[str]:
    issues: List[str] = []

    i0 = nib.load(p0)
    i1 = nib.load(p1)
    il = nib.load(pl)

    s0 = i0.shape
    s1 = i1.shape
    sl = il.shape

    if s0 != s1 or s0 != sl:
        issues.append(f"shape mismatch: _0000={s0}, _0001={s1}, label={sl}")

    a0 = i0.affine
    a1 = i1.affine
    al = il.affine
    if not np.allclose(a0, a1, atol=1e-5):
        issues.append("affine mismatch between _0000 and _0001")
    if not np.allclose(a0, al, atol=1e-5):
        issues.append("affine mismatch between _0000 and label")

    d1 = np.asarray(i1.get_fdata(dtype=np.float32))
    dl = np.asarray(il.get_fdata(dtype=np.float32))

    if not np.all(np.isfinite(dl)):
        issues.append("label contains non-finite values")

    if not np.allclose(dl, np.round(dl), atol=1e-6):
        issues.append("label has non-integer values")

    dli = np.round(dl).astype(np.int32)
    if dli.min() < 0 or dli.max() > label_max:
        issues.append(f"label out of range [0,{label_max}]: min={dli.min()}, max={dli.max()}")

    mask = d1 > 0.5
    if mask.sum() == 0:
        issues.append("mask(_0001) is empty")
    else:
        outside_nonzero = np.count_nonzero((~mask) & (dli != 0))
        if outside_nonzero > 0:
            issues.append(f"label non-zero outside mask: voxels={outside_nonzero}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EPT nnUNet dataset consistency.")
    parser.add_argument("--dataset-root", required=True, help="e.g. /.../nnUNet_raw/Dataset001_EPT")
    parser.add_argument("--label-max", type=int, default=30, help="max valid label id (default 30)")
    parser.add_argument("--strict", action="store_true", help="exit non-zero if any warning")
    args = parser.parse_args()

    images_tr = os.path.join(args.dataset_root, "imagesTr")
    labels_tr = os.path.join(args.dataset_root, "labelsTr")

    if not os.path.isdir(images_tr) or not os.path.isdir(labels_tr):
        print(f"[FATAL] missing folder: {images_tr} or {labels_tr}")
        return 2

    ch0, ch1, lab = collect_cases(images_tr, labels_tr)

    cases_ch0 = set(ch0)
    cases_ch1 = set(ch1)
    cases_lab = set(lab)

    all_cases = sorted(cases_ch0 | cases_ch1 | cases_lab)

    miss_ch0 = sorted((cases_ch1 | cases_lab) - cases_ch0)
    miss_ch1 = sorted((cases_ch0 | cases_lab) - cases_ch1)
    miss_lab = sorted((cases_ch0 | cases_ch1) - cases_lab)

    print(f"[INFO] cases: _0000={len(cases_ch0)}, _0001={len(cases_ch1)}, label={len(cases_lab)}, union={len(all_cases)}")

    if miss_ch0:
        print(f"[ERR] missing _0000 for {len(miss_ch0)} cases: {', '.join(miss_ch0[:15])}")
    if miss_ch1:
        print(f"[ERR] missing _0001 for {len(miss_ch1)} cases: {', '.join(miss_ch1[:15])}")
    if miss_lab:
        print(f"[ERR] missing label for {len(miss_lab)} cases: {', '.join(miss_lab[:15])}")

    common = sorted(cases_ch0 & cases_ch1 & cases_lab)
    print(f"[INFO] fully paired cases: {len(common)}")

    bad_cases: List[Tuple[str, List[str]]] = []
    for c in common:
        issues = check_one_case(c, ch0[c], ch1[c], lab[c], args.label_max)
        if issues:
            bad_cases.append((c, issues))

    if bad_cases:
        print(f"[ERR] inconsistent cases: {len(bad_cases)}")
        for c, issues in bad_cases[:50]:
            print(f"  - {c}")
            for it in issues:
                print(f"    * {it}")
    else:
        print("[OK] no per-case consistency issues found")

    has_errors = bool(miss_ch0 or miss_ch1 or miss_lab or bad_cases)
    if has_errors and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
