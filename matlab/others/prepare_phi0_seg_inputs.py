#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import nibabel as nib
except Exception as e:
    print(f"[FATAL] nibabel import failed: {e}")
    print("Install with: pip install nibabel")
    sys.exit(2)


def _sanitize_case_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _split_tokens(value: str) -> List[str]:
    return [tok for tok in re.split(r"[^A-Za-z0-9]+", value) if tok]


def _has_alpha_and_digit(token: str) -> bool:
    has_alpha = any(ch.isalpha() for ch in token)
    has_digit = any(ch.isdigit() for ch in token)
    return has_alpha and has_digit


def _first_alpha_upper(token: str) -> str:
    for ch in token:
        if ch.isalpha():
            return ch.upper()
    return ""


def auto_case_id_from_parent_name(parent_name: str) -> str:
    tokens = _split_tokens(parent_name)
    if not tokens:
        return "CASE"

    main_idx = -1
    main_token = ""
    for idx, token in enumerate(tokens):
        if len(token) >= 4 and _has_alpha_and_digit(token):
            main_idx = idx
            main_token = token.upper()
            break

    if main_idx < 0:
        for idx, token in enumerate(tokens):
            if len(token) >= 3:
                main_idx = idx
                main_token = token.upper()
                break

    if main_idx < 0:
        main_idx = 0
        main_token = tokens[0].upper()

    suffix = "".join(_first_alpha_upper(tok) for tok in tokens[main_idx + 1 :])
    case_id = f"{main_token}_{suffix}" if suffix else main_token
    case_id = _sanitize_case_id(case_id)
    return case_id if case_id else "CASE"


def remap_seg_1_2(seg_path: Path) -> Tuple[np.ndarray, List[int], List[int]]:
    img = nib.load(str(seg_path))
    seg = np.asarray(img.get_fdata())
    if not np.all(np.isfinite(seg)):
        raise ValueError("segmentation contains non-finite values")

    seg_int = np.rint(seg).astype(np.int32)
    unique_before = sorted(int(x) for x in np.unique(seg_int))

    remapped = seg_int.copy()
    remapped[seg_int == 1] = 2
    remapped[seg_int == 2] = 1

    if remapped.min() < 0 or remapped.max() > 255:
        raise ValueError(f"segmentation labels out of uint8 range after remap: [{remapped.min()}, {remapped.max()}]")

    remapped_u8 = remapped.astype(np.uint8)
    unique_after = sorted(int(x) for x in np.unique(remapped_u8))
    return remapped_u8, unique_before, unique_after


def _ensure_writable_target(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"target exists and overwrite=false: {path}")


def process_case(
    *,
    mode: str,
    phi_path: Path,
    seg_path: Path,
    out_dir: Path,
    case_id: str,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    case_id = _sanitize_case_id(case_id)
    if not case_id:
        raise ValueError("empty case_id after sanitization")

    phi_out = out_dir / f"{case_id}_0000.nii.gz"
    seg_out = out_dir / f"{case_id}_0001.nii.gz"

    result: Dict[str, Any] = {
        "mode": mode,
        "case_id": case_id,
        "phi0_in": str(phi_path),
        "seg_in": str(seg_path),
        "out_dir": str(out_dir),
        "phi0_out": str(phi_out),
        "seg_out": str(seg_out),
        "status": "planned" if dry_run else "success",
        "message": "",
        "seg_labels_before": "",
        "seg_labels_after": "",
    }

    _ensure_writable_target(phi_out, overwrite)
    _ensure_writable_target(seg_out, overwrite)

    remapped_u8, unique_before, unique_after = remap_seg_1_2(seg_path)
    result["seg_labels_before"] = ",".join(str(v) for v in unique_before)
    result["seg_labels_after"] = ",".join(str(v) for v in unique_after)

    if dry_run:
        return result

    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(phi_path, phi_out)

    seg_img = nib.load(str(seg_path))
    seg_hdr = seg_img.header.copy()
    seg_hdr.set_data_dtype(np.uint8)
    out_seg_img = nib.Nifti1Image(remapped_u8, seg_img.affine, seg_hdr)
    nib.save(out_seg_img, str(seg_out))
    return result


def _write_report(report_path: Path, rows: List[Dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ext = report_path.suffix.lower()
    if ext == ".json":
        status_counts: Dict[str, int] = {}
        for row in rows:
            status = str(row.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
        payload = {
            "summary": {"count": len(rows), "status_counts": status_counts},
            "results": rows,
        }
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    if ext == ".csv":
        fieldnames: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    raise ValueError(f"unsupported report extension: {report_path} (use .json or .csv)")


def _find_phi_candidates(input_root: Path, phi_name: str, recursive: bool) -> List[Path]:
    candidates: List[Path]
    if recursive:
        candidates = sorted(p for p in input_root.rglob(phi_name) if p.is_file())
    else:
        candidates = []
        root_phi = input_root / phi_name
        if root_phi.is_file():
            candidates.append(root_phi)
        for child in sorted(input_root.iterdir()):
            if child.is_dir():
                p = child / phi_name
                if p.is_file():
                    candidates.append(p)
    return candidates


def run_single(args: argparse.Namespace) -> int:
    phi_path = Path(args.phi0).expanduser().resolve()
    seg_path = Path(args.seg).expanduser().resolve()
    if not phi_path.is_file():
        print(f"[FATAL] phi0 file not found: {phi_path}")
        return 2
    if not seg_path.is_file():
        print(f"[FATAL] segmentation file not found: {seg_path}")
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else phi_path.parent
    case_id = args.case_id.strip() if args.case_id else auto_case_id_from_parent_name(phi_path.parent.name)

    try:
        row = process_case(
            mode="single",
            phi_path=phi_path,
            seg_path=seg_path,
            out_dir=out_dir,
            case_id=case_id,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(f"[{row['status'].upper()}] case={row['case_id']}")
        print(f"  phi0: {row['phi0_in']} -> {row['phi0_out']}")
        print(f"  seg : {row['seg_in']} -> {row['seg_out']}")
        if row["seg_labels_before"] != "":
            print(f"  seg labels before: {row['seg_labels_before']}")
            print(f"  seg labels after : {row['seg_labels_after']}")
        rows = [row]
    except Exception as e:
        row = {
            "mode": "single",
            "case_id": case_id,
            "phi0_in": str(phi_path),
            "seg_in": str(seg_path),
            "out_dir": str(out_dir),
            "phi0_out": str(out_dir / f"{_sanitize_case_id(case_id)}_0000.nii.gz"),
            "seg_out": str(out_dir / f"{_sanitize_case_id(case_id)}_0001.nii.gz"),
            "status": "error",
            "message": str(e),
            "seg_labels_before": "",
            "seg_labels_after": "",
        }
        rows = [row]
        print(f"[ERROR] case={case_id}: {e}")
        if args.report:
            _write_report(Path(args.report).expanduser().resolve(), rows)
        return 1

    if args.report:
        _write_report(Path(args.report).expanduser().resolve(), rows)
        print(f"[INFO] report written: {Path(args.report).expanduser().resolve()}")
    return 0


def run_batch(args: argparse.Namespace) -> int:
    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.is_dir():
        print(f"[FATAL] input root not found: {input_root}")
        return 2

    phi_candidates = _find_phi_candidates(input_root, args.phi_name, args.recursive)
    if not phi_candidates:
        print(f"[FATAL] no '{args.phi_name}' found under: {input_root}")
        return 2

    rows: List[Dict[str, Any]] = []
    for phi_path in phi_candidates:
        seg_path = phi_path.parent / args.seg_name
        auto_case_id = auto_case_id_from_parent_name(phi_path.parent.name)
        out_dir = phi_path.parent

        if not seg_path.is_file():
            row = {
                "mode": "batch",
                "case_id": auto_case_id,
                "phi0_in": str(phi_path),
                "seg_in": str(seg_path),
                "out_dir": str(out_dir),
                "phi0_out": str(out_dir / f"{auto_case_id}_0000.nii.gz"),
                "seg_out": str(out_dir / f"{auto_case_id}_0001.nii.gz"),
                "status": "error",
                "message": f"missing segmentation: {seg_path.name}",
                "seg_labels_before": "",
                "seg_labels_after": "",
            }
            rows.append(row)
            print(f"[ERROR] {phi_path.parent}: missing {args.seg_name}")
            continue

        try:
            row = process_case(
                mode="batch",
                phi_path=phi_path,
                seg_path=seg_path,
                out_dir=out_dir,
                case_id=auto_case_id,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            rows.append(row)
            print(f"[{row['status'].upper()}] case={row['case_id']} dir={phi_path.parent}")
        except Exception as e:
            row = {
                "mode": "batch",
                "case_id": auto_case_id,
                "phi0_in": str(phi_path),
                "seg_in": str(seg_path),
                "out_dir": str(out_dir),
                "phi0_out": str(out_dir / f"{auto_case_id}_0000.nii.gz"),
                "seg_out": str(out_dir / f"{auto_case_id}_0001.nii.gz"),
                "status": "error",
                "message": str(e),
                "seg_labels_before": "",
                "seg_labels_after": "",
            }
            rows.append(row)
            print(f"[ERROR] case={auto_case_id}: {e}")

    n_success = sum(1 for r in rows if r["status"] == "success")
    n_planned = sum(1 for r in rows if r["status"] == "planned")
    n_error = sum(1 for r in rows if r["status"] == "error")
    print(f"[INFO] done. total={len(rows)} success={n_success} planned={n_planned} error={n_error}")

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        _write_report(report_path, rows)
        print(f"[INFO] report written: {report_path}")

    return 1 if n_error > 0 else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare nnUNet inference inputs from phi0/segmentation pairs. "
            "Writes <case_id>_0000.nii.gz (phi0) and <case_id>_0001.nii.gz "
            "(segmentation with labels 1<->2 swapped)."
        )
    )
    parser.add_argument("--mode", required=True, choices=("single", "batch"))

    parser.add_argument("--phi0", help="single mode: path to phi0 NIfTI")
    parser.add_argument("--seg", help="single mode: path to segmentation NIfTI")
    parser.add_argument("--case-id", help="single mode: optional output case id")
    parser.add_argument("--out-dir", help="single mode: output folder (default: phi0 parent)")

    parser.add_argument("--input-root", help="batch mode: root folder to scan")
    parser.add_argument("--phi-name", default="phi0.nii.gz", help="batch mode: phi filename (default: phi0.nii.gz)")
    parser.add_argument(
        "--seg-name",
        default="segmentation.nii.gz",
        help="batch mode: segmentation filename (default: segmentation.nii.gz)",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="batch mode: recursively scan input root (default: true)",
    )

    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="overwrite existing output files (default: true)",
    )
    parser.add_argument("--dry-run", action="store_true", help="plan only, do not write output files")
    parser.add_argument("--report", help="optional report path (.json or .csv)")
    args = parser.parse_args()

    if args.mode == "single":
        if not args.phi0 or not args.seg:
            parser.error("--mode single requires --phi0 and --seg")
    if args.mode == "batch":
        if not args.input_root:
            parser.error("--mode batch requires --input-root")
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "single":
        return run_single(args)
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
