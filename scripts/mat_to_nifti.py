#!/usr/bin/env python3
"""
Convert MATLAB .mat arrays into .nii.gz files with explicit geometry control.

Recommended runtime:
  nnunet/.venv/bin/python scripts/mat_to_nifti.py --input ...

Important behavior:
- Geometry defaults to identity sform and a qform with equivalent spatial meaning.
- Complex handling is explicit via --complex-mode to avoid silently writing
  nnUNet-incompatible complex NIfTI files.
- Real outputs are stored as float32; complex outputs are stored as complex64.
- Arrays are normalized to C-contiguous memory before constructing the NIfTI
  object. This avoids slow slicing and warnings caused by MATLAB-style F-order
  arrays.

Examples:
1) Convert one named variable in one file to float32 real:
   nnunet/.venv/bin/python scripts/mat_to_nifti.py \
     --input /Users/apple/Documents/deeplc/ADEPT_Dataset/Healthy/M1.mat \
     --mat-key B1minus_mag \
     --complex-mode real

2) Convert one named variable using magnitude:
   nnunet/.venv/bin/python scripts/mat_to_nifti.py \
     --input /Users/apple/Documents/deeplc/ADEPT_Dataset/Healthy/M1.mat \
     --mat-key B1minus_mag \
     --complex-mode magnitude

3) Batch convert all numeric variables in a directory:
   nnunet/.venv/bin/python scripts/mat_to_nifti.py \
     --input /Users/apple/Documents/deeplc/ADEPT_Dataset/Healthy \
     --batch \
     --complex-mode real

4) Preserve complex values as complex64 NIfTI (not nnUNet-compatible):
   nnunet/.venv/bin/python scripts/mat_to_nifti.py \
     --input /Users/apple/Documents/deeplc/ADEPT_Dataset/Healthy/M1.mat \
     --mat-key B1minus_mag \
     --complex-mode complex \
     --sform-matrix identity
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

try:
    import nibabel as nib
    from scipy.io import loadmat
except Exception as exc:  # pragma: no cover - handled in main for CLI usage
    nib = None
    loadmat = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from normalize_nifti_identity import (
    ConfigError,
    canonical_qform_affine,
    encode_descrip,
    matrix_spacing_and_qfac,
    parse_matrix_spec,
    parse_xyzt_units,
)

EXIT_GENERAL_ERROR = 1
EXIT_SELECTION_ERROR = 2

VALID_COMPLEX_MODES = ("complex", "real", "magnitude")
MATLAB_META_KEYS = {"__header__", "__version__", "__globals__"}


@dataclass(frozen=True)
class MatVariable:
    path: Path
    key: str
    array: np.ndarray


@dataclass(frozen=True)
class PreparedArray:
    data: np.ndarray
    source_dtype: str
    output_dtype: str
    was_complex: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input .mat file or directory.")
    p.add_argument(
        "--pattern",
        default="*.mat",
        help="Glob pattern for directory input. Default: *.mat",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: write next to the input .mat file.",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Convert all numeric variables from each .mat file.",
    )
    p.add_argument(
        "--mat-key",
        default=None,
        help="Variable name to convert in non-batch mode.",
    )
    p.add_argument(
        "--filename-template",
        default="{case_id}_{var_name}.nii.gz",
        help="Output filename template using {case_id} and {var_name}.",
    )
    p.add_argument(
        "--sform-matrix",
        default="identity",
        help="Target sform matrix: identity or 12/16 numeric values.",
    )
    p.add_argument("--qform-code", type=int, default=1, help="NIfTI qform_code. Default: 1")
    p.add_argument("--sform-code", type=int, default=1, help="NIfTI sform_code. Default: 1")
    p.add_argument(
        "--xyzt-units",
        default=None,
        help="Optional units, e.g. mm,sec or NIFTI_UNITS_MM,NIFTI_UNITS_SEC.",
    )
    p.add_argument("--descrip", default=None, help="ASCII text for NIfTI descrip (max 80 bytes).")
    p.add_argument(
        "--complex-mode",
        required=True,
        choices=VALID_COMPLEX_MODES,
        help=(
            "Complex handling policy: complex keeps complex values as complex64 "
            "(not nnUNet-compatible), real keeps only the real part, magnitude "
            "stores the absolute value."
        ),
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing.")
    args = p.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.batch and args.mat_key:
        raise ConfigError("--batch and --mat-key are mutually exclusive.")

    for key in ("qform_code", "sform_code"):
        value = getattr(args, key)
        if value < 0 or value > 4:
            raise ConfigError(f"--{key.replace('_', '-')} must be in [0, 4], got {value}.")

    _ = parse_matrix_spec(args.sform_matrix)

    if args.xyzt_units is not None:
        _ = parse_xyzt_units(args.xyzt_units)

    if args.descrip is not None:
        _ = encode_descrip(args.descrip)

    try:
        rendered = args.filename_template.format(case_id="CASE", var_name="VAR")
    except KeyError as exc:
        raise ConfigError(f"Unknown filename template placeholder: {exc}") from exc
    except Exception as exc:
        raise ConfigError(f"Invalid --filename-template: {exc}") from exc

    if not rendered:
        raise ConfigError("--filename-template rendered an empty filename.")

    if Path(rendered).name != rendered:
        raise ConfigError("--filename-template must render a filename, not a path.")


def collect_mat_files(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".mat":
            raise ConfigError(f"Input file must end with .mat: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob(pattern) if p.is_file())
    raise ConfigError(f"Input does not exist: {input_path}")


def load_mat_variables(path: Path) -> dict:
    assert loadmat is not None
    try:
        return loadmat(str(path))
    except NotImplementedError as exc:
        raise ConfigError(
            f"Unsupported MAT format for {path}. This script supports MATLAB v5/v6/v7 MAT files "
            "via scipy.io.loadmat. For v7.3/HDF5 files, use a dedicated h5py-based reader."
        ) from exc
    except ValueError as exc:
        raise ConfigError(f"Failed to read {path}: {exc}") from exc


def is_numeric_ndarray(value: object) -> bool:
    if not isinstance(value, np.ndarray):
        return False
    if value.dtype.kind in {"O", "V", "S", "U"}:
        return False
    if value.ndim == 0 or value.size <= 1:
        return False
    return value.dtype.kind == "b" or np.issubdtype(value.dtype, np.number)


def discover_numeric_variables(path: Path) -> List[MatVariable]:
    variables = load_mat_variables(path)
    candidates: List[MatVariable] = []
    for key, value in variables.items():
        if key in MATLAB_META_KEYS:
            continue
        if not is_numeric_ndarray(value):
            continue
        candidates.append(MatVariable(path=path, key=key, array=value))
    return candidates


def select_variables(path: Path, args: argparse.Namespace) -> List[MatVariable]:
    candidates = discover_numeric_variables(path)
    if not candidates:
        raise ConfigError(f"No numeric array variables found in {path}.")

    if args.batch:
        return candidates

    if args.mat_key is not None:
        for candidate in candidates:
            if candidate.key == args.mat_key:
                return [candidate]
        available = ", ".join(c.key for c in candidates)
        raise ConfigError(
            f'Variable "{args.mat_key}" not found in {path}. Available numeric variables: {available}'
        )

    if len(candidates) == 1:
        return candidates

    available = ", ".join(c.key for c in candidates)
    raise ConfigError(
        f"Multiple numeric variables found in {path}; pass --mat-key or use --batch. "
        f"Available: {available}"
    )


def prepare_array(array: np.ndarray, complex_mode: str) -> PreparedArray:
    was_complex = bool(np.iscomplexobj(array))
    source_dtype = str(array.dtype)

    if was_complex:
        if complex_mode == "real":
            converted = np.real(array)
            output_dtype = np.dtype(np.float32)
        elif complex_mode == "magnitude":
            converted = np.abs(array)
            output_dtype = np.dtype(np.float32)
        elif complex_mode == "complex":
            converted = array
            output_dtype = np.dtype(np.complex64)
        else:  # pragma: no cover - guarded by argparse
            raise ConfigError(f"Unsupported --complex-mode: {complex_mode}")
    else:
        converted = array
        output_dtype = np.dtype(np.float32)

    data = np.asanyarray(converted, dtype=output_dtype, order="C")
    return PreparedArray(
        data=data,
        source_dtype=source_dtype,
        output_dtype=str(data.dtype),
        was_complex=was_complex,
    )


def build_output_name(case_id: str, var_name: str, template: str) -> str:
    rendered = template.format(case_id=case_id, var_name=var_name)
    name = Path(rendered).name
    if name != rendered:
        raise ConfigError("--filename-template must not create subdirectories.")
    if not name.lower().endswith((".nii.gz", ".nii")):
        name = f"{name}.nii.gz"
    return name


def resolve_output_path(variable: MatVariable, args: argparse.Namespace) -> Path:
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = variable.path.parent
    output_name = build_output_name(variable.path.stem, variable.key, args.filename_template)
    return base_dir / output_name


def ensure_outputs_writable(paths: Iterable[Path], overwrite: bool) -> None:
    seen = set()
    for path in paths:
        if path in seen:
            raise ConfigError(f"Output path collision detected: {path}")
        seen.add(path)
        if path.exists() and not overwrite:
            raise ConfigError(f"Output exists, use --overwrite to replace it: {path}")


def build_nifti_image(
    prepared: PreparedArray,
    sform_matrix: np.ndarray,
    qform_code: int,
    sform_code: int,
    xyzt_units: str | None,
    descrip: str | None,
):
    assert nib is not None
    target_sform = np.asarray(sform_matrix, dtype=np.float64)
    spacing, qfac, _ = matrix_spacing_and_qfac(target_sform[:3, :3])
    if qform_code > 0:
        target_qform = canonical_qform_affine(target_sform, qform_code)
    else:
        target_qform = np.asarray(target_sform, dtype=np.float64)

    header = nib.Nifti1Header()
    header.set_data_dtype(prepared.data.dtype)
    header["pixdim"][0] = np.float32(qfac)
    header["pixdim"][1:4] = np.asarray(spacing, dtype=np.float32)
    if xyzt_units is not None:
        xyz_unit, t_unit = parse_xyzt_units(xyzt_units)
        header.set_xyzt_units(xyz=xyz_unit, t=t_unit)
    if descrip is not None:
        header["descrip"] = encode_descrip(descrip)

    image = nib.Nifti1Image(prepared.data, affine=target_sform, header=header)
    image.set_qform(target_qform, code=int(qform_code))
    image.set_sform(target_sform, code=int(sform_code))
    image.header["pixdim"][0] = np.float32(qfac)
    image.header["pixdim"][1:4] = np.asarray(spacing, dtype=np.float32)
    return image


def write_nifti(output_path: Path, image) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))


def main() -> None:
    if IMPORT_ERROR is not None:
        raise ConfigError(
            "Missing dependency. Run this script with nnunet/.venv/bin/python "
            f"or install numpy/scipy/nibabel. Original error: {IMPORT_ERROR}"
        )

    args = parse_args()
    input_path = Path(args.input).expanduser()
    sform_matrix = parse_matrix_spec(args.sform_matrix)
    files = collect_mat_files(input_path, args.pattern)
    if not files:
        raise ConfigError(f"No .mat files matched {args.pattern!r} under {input_path}")

    selections: List[tuple[MatVariable, Path, PreparedArray]] = []
    for mat_path in files:
        for variable in select_variables(mat_path, args):
            prepared = prepare_array(variable.array, args.complex_mode)
            output_path = resolve_output_path(variable, args)
            selections.append((variable, output_path, prepared))

    ensure_outputs_writable((dst for _, dst, _ in selections), overwrite=args.overwrite)

    converted_count = 0
    for variable, output_path, prepared in selections:
        message = (
            f"{variable.path}:{variable.key} -> {output_path} | "
            f"shape={tuple(int(x) for x in prepared.data.shape)} "
            f"source_dtype={prepared.source_dtype} "
            f"output_dtype={prepared.output_dtype} "
            f"complex_input={prepared.was_complex} "
            f"C_contiguous={prepared.data.flags['C_CONTIGUOUS']}"
        )
        print(message)
        if args.dry_run:
            continue

        image = build_nifti_image(
            prepared=prepared,
            sform_matrix=sform_matrix,
            qform_code=args.qform_code,
            sform_code=args.sform_code,
            xyzt_units=args.xyzt_units,
            descrip=args.descrip,
        )
        write_nifti(output_path, image)
        converted_count += 1

    print(f"Summary: matched_files={len(files)} selected_arrays={len(selections)} written={converted_count}")
    if args.complex_mode == "complex":
        print("[WARN] complex mode writes complex64 NIfTI output and is not nnUNet-compatible.")
    elif not args.dry_run:
        print("Outputs are float32 and nnUNet-compatible with respect to dtype.")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(EXIT_SELECTION_ERROR)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(EXIT_GENERAL_ERROR)
