#!/usr/bin/env python3
"""
Normalize/fix NIfTI geometry metadata with matrix-first consistency.

Key guarantees:
1) File discovery is performed in Python (no shell glob pitfalls).
2) Matrix-first behavior: spacing/qfac are always derived from the target spatial matrix.
3) Handedness fail-fast checks run before skip/dirty decisions.
4) Idempotent rewrites use tolerant floating-point comparisons.

Important scope:
- This script rewrites NIfTI metadata/header geometry only.
- Voxel indexing order and raw data values are not resampled or reoriented.

How to use:
1) Report only (fail if any undefined q/s form exists):
   python scripts/normalize_nifti_identity.py \
     --input /path/to/images \
     --pattern '*.nii.gz' \
     --action report \
     --fail-on-undefined

2) Fix only fully-undefined files (qform_code=0 and sform_code=0):
   python scripts/normalize_nifti_identity.py \
     --input /path/to/images \
     --action fix-undefined \
     --inplace

3) Fill missing form from the existing one (without forcing identity):
   python scripts/normalize_nifti_identity.py \
     --input /path/to/images \
     --action fix-missing-form \
     --inplace

4) Rewrite all files using an explicit target matrix:
   python scripts/normalize_nifti_identity.py \
     --input /path/to/images \
     --action fix-all \
     --inplace \
     --sform-matrix identity

5) Enforce fail-fast handedness conflict blocking:
   python scripts/normalize_nifti_identity.py \
     --input /path/to/images \
     --action fix-all \
     --inplace \
     --handedness-policy fail-fast

6) Write ASCII descrip (max 80 bytes):
   python scripts/normalize_nifti_identity.py \
     --input /path/to/images \
     --action fix-all \
     --inplace \
     --descrip 'geom_v1'
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

EXIT_GENERAL_ERROR = 1
EXIT_UNDEFINED_FOUND = 2
EXIT_NO_FILES_MATCHED = 3

VALID_ACTIONS = ("report", "fix-undefined", "fix-missing-form", "fix-all")
VALID_HANDEDNESS_POLICY = ("unify", "fail-fast")

# Fixed constants (intentionally not exposed as CLI to keep behavior predictable).
SPACING_MIN = 1e-6
DET_DIR_ATOL = 1e-4

# Tolerances for drift detection and conflict checks.
DIRTY_ATOL = 1e-5
DIRTY_RTOL = 1e-6
CONFLICT_ATOL = 1e-4
CONFLICT_RTOL = 1e-5


@dataclass
class CaseSummary:
    # Snapshot read during precheck, used later in main state transitions.
    path: Path
    qform_code: int
    sform_code: int
    qform_affine: np.ndarray
    sform_affine: np.ndarray


@dataclass
class TargetState:
    # Fully resolved target header state for one file.
    qform_affine: np.ndarray
    sform_affine: np.ndarray
    qform_code: int
    sform_code: int
    pixdim: np.ndarray
    xyzt_units_code: Optional[int]
    cal_min: Optional[float]
    cal_max: Optional[float]
    descrip_bytes: Optional[bytes]


class ConfigError(RuntimeError):
    # Raised for invalid CLI combinations and geometry consistency violations.
    pass


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and run cross-argument validation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input NIfTI file or directory.")
    p.add_argument("--pattern", default="*.nii.gz", help="Glob pattern for directory input.")
    p.add_argument("--inplace", action="store_true", help="Overwrite input files in place.")
    p.add_argument("--action", default="report", choices=VALID_ACTIONS)
    p.add_argument("--fail-on-undefined", action="store_true")
    p.add_argument("--fail-on-no-match", action="store_true")
    p.add_argument("--suffix", default="_geomfix")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument(
        "--handedness-policy",
        default="unify",
        choices=VALID_HANDEDNESS_POLICY,
        help="How to handle qform/sform handedness mismatch when both are defined.",
    )
    p.add_argument(
        "--sform-matrix",
        default=None,
        help="Target sform matrix: 'identity' or 12/16 numbers (comma/space separated).",
    )
    p.add_argument("--qform-code", type=int, default=None)
    p.add_argument("--sform-code", type=int, default=None)

    p.add_argument(
        "--pixdim-spatial",
        default=None,
        help="Expected spatial pixdim (3 numbers). Validation only when matrix is given.",
    )
    p.add_argument(
        "--pixdim",
        default=None,
        help="Optional pixdim override list (1..8 values). Spatial/qfac entries must match matrix-derived values.",
    )
    p.add_argument(
        "--xyzt-units",
        default=None,
        help="Optional units as 'xyz,t' (for example: mm,sec or NIFTI_UNITS_MM,NIFTI_UNITS_SEC).",
    )
    p.add_argument("--cal-min", type=float, default=None)
    p.add_argument("--cal-max", type=float, default=None)
    p.add_argument("--descrip", default=None, help="ASCII text for NIfTI descrip (max 80 bytes).")

    args = p.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate semantic constraints that argparse alone cannot express."""
    for key in ("qform_code", "sform_code"):
        val = getattr(args, key)
        if val is not None and (val < 0 or val > 4):
            raise ConfigError(f"--{key.replace('_', '-')} must be in [0, 4], got {val}.")

    if args.action == "report":
        write_args = {
            "--sform-matrix": args.sform_matrix,
            "--qform-code": args.qform_code,
            "--sform-code": args.sform_code,
            "--pixdim-spatial": args.pixdim_spatial,
            "--pixdim": args.pixdim,
            "--xyzt-units": args.xyzt_units,
            "--cal-min": args.cal_min,
            "--cal-max": args.cal_max,
            "--descrip": args.descrip,
            "--inplace": args.inplace,
            "--suffix": None if args.suffix == "_geomfix" else args.suffix,
        }
        used = [k for k, v in write_args.items() if v is not None and v is not False]
        if used:
            raise ConfigError(
                "report action is read-only; remove write parameters: " + ", ".join(used)
            )

    # Parse now so malformed values fail before touching any image file.
    if args.sform_matrix is not None:
        _ = parse_matrix_spec(args.sform_matrix)
    if args.pixdim_spatial is not None:
        vals = parse_numeric_list(args.pixdim_spatial)
        if len(vals) != 3:
            raise ConfigError("--pixdim-spatial requires exactly 3 numbers.")
    if args.pixdim is not None:
        vals = parse_numeric_list(args.pixdim)
        if len(vals) == 0 or len(vals) > 8:
            raise ConfigError("--pixdim requires 1..8 numbers.")
    if args.xyzt_units is not None:
        _ = parse_xyzt_units(args.xyzt_units)
    if args.descrip is not None:
        _ = encode_descrip(args.descrip)



def collect_files(input_path: Path, pattern: str) -> List[Path]:
    """
    Resolve files from --input and --pattern in Python.

    This avoids shell expansion edge cases such as no-match literals and very large
    argument lists.
    """
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob(pattern) if p.is_file())
    raise FileNotFoundError(f"Input does not exist: {input_path}")



def output_path(src: Path, inplace: bool, suffix: str) -> Path:
    """Build output filename for in-place or side-by-side rewrite modes."""
    if inplace:
        return src
    name = src.name
    if name.endswith(".nii.gz"):
        return src.with_name(name[:-7] + suffix + ".nii.gz")
    if name.endswith(".nii"):
        return src.with_name(name[:-4] + suffix + ".nii")
    return src.with_name(name + suffix)



def parse_numeric_list(spec: str) -> List[float]:
    """Parse numbers from comma/space/semicolon separated text."""
    tokens = [t for t in re.split(r"[\s,;]+", spec.strip()) if t]
    if not tokens:
        return []
    try:
        return [float(t) for t in tokens]
    except ValueError as exc:
        raise ConfigError(f"Invalid numeric list: {spec}") from exc



def parse_matrix_spec(spec: str) -> np.ndarray:
    """
    Parse target matrix spec into a 4x4 affine.

    Supported forms:
    - 'identity'
    - 12 values (interpreted as 3x4 row-major)
    - 16 values (full 4x4, last row must be [0,0,0,1] within tolerance)
    """
    if spec.strip().lower() == "identity":
        return np.eye(4, dtype=np.float64)

    vals = parse_numeric_list(spec)
    if len(vals) == 12:
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
        return out
    if len(vals) == 16:
        out = np.asarray(vals, dtype=np.float64).reshape(4, 4)
        if not np.allclose(out[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=CONFLICT_ATOL, rtol=CONFLICT_RTOL):
            raise ConfigError("4x4 matrix must have last row [0, 0, 0, 1].")
        return out

    raise ConfigError("--sform-matrix must be 'identity' or 12/16 numbers.")



def parse_xyzt_units(spec: str) -> Tuple[str, str]:
    """Parse flexible xyzt unit aliases into nibabel-compatible tokens."""
    token_map = {
        "unknown": "unknown",
        "nifti_units_unknown": "unknown",
        "meter": "meter",
        "meters": "meter",
        "m": "meter",
        "nifti_units_meter": "meter",
        "mm": "mm",
        "millimeter": "mm",
        "millimeters": "mm",
        "nifti_units_mm": "mm",
        "micron": "micron",
        "microns": "micron",
        "um": "micron",
        "nifti_units_micron": "micron",
        "sec": "sec",
        "s": "sec",
        "second": "sec",
        "seconds": "sec",
        "nifti_units_sec": "sec",
        "msec": "msec",
        "ms": "msec",
        "millisecond": "msec",
        "milliseconds": "msec",
        "nifti_units_msec": "msec",
        "usec": "usec",
        "us": "usec",
        "microsecond": "usec",
        "microseconds": "usec",
        "nifti_units_usec": "usec",
        "hz": "hz",
        "nifti_units_hz": "hz",
        "ppm": "ppm",
        "nifti_units_ppm": "ppm",
        "rads": "rads",
        "nifti_units_rads": "rads",
    }

    parts = [p for p in re.split(r"[\s,;/]+", spec.strip()) if p]
    if len(parts) == 0 or len(parts) > 2:
        raise ConfigError("--xyzt-units expects one or two tokens, e.g. mm,sec")

    normalized = []
    for p in parts:
        key = p.strip().lower()
        if key not in token_map:
            raise ConfigError(f"Unsupported unit token: {p}")
        normalized.append(token_map[key])

    xyz_units = {"unknown", "meter", "mm", "micron"}
    t_units = {"unknown", "sec", "msec", "usec", "hz", "ppm", "rads"}

    if len(normalized) == 1:
        if normalized[0] in xyz_units:
            return normalized[0], "sec"
        if normalized[0] in t_units:
            return "mm", normalized[0]
        raise ConfigError(f"Invalid single xyzt unit token: {parts[0]}")

    xyz, t = normalized
    if xyz not in xyz_units:
        raise ConfigError(f"Invalid xyz unit token: {parts[0]}")
    if t not in t_units:
        raise ConfigError(f"Invalid t unit token: {parts[1]}")
    return xyz, t



def encode_descrip(text: str) -> bytes:
    """
    Encode header descrip safely for NIfTI-1.

    - ASCII only (prevents UTF-8 split-byte truncation issues)
    - 80-byte hard limit
    """
    try:
        encoded = text.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise ConfigError("--descrip must be ASCII-only to avoid UTF-8 truncation issues.") from exc

    if len(encoded) > 80:
        print(f"[WARN] descrip is {len(encoded)} bytes; truncating to 80 bytes.")
        encoded = encoded[:80]
    return encoded



def matrix_spacing_and_qfac(spatial_matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Compute spacing and qfac from a spatial matrix.

    spacing is the L2 norm of matrix columns (axis=0), matching affine basis vectors.
    """
    # Column norms are the physically meaningful spacing implied by the affine axes.
    spacing = np.linalg.norm(spatial_matrix, axis=0)
    if not np.all(np.isfinite(spacing)):
        raise ConfigError("Target spatial matrix produced non-finite spacing.")
    if np.any(spacing < SPACING_MIN):
        raise ConfigError(
            f"Target spatial matrix is degenerate: spacing={spacing}, threshold={SPACING_MIN}."
        )

    # Normalize out scale before determinant check to detect directional singularity.
    direction = spatial_matrix / spacing
    det_dir = float(np.linalg.det(direction))
    if np.isclose(det_dir, 0.0, atol=DET_DIR_ATOL, rtol=0.0):
        raise ConfigError(
            f"Target spatial matrix is computationally singular after normalization: det_dir={det_dir:.6g}."
        )

    qfac = 1.0 if det_dir > 0 else -1.0
    return spacing, qfac, det_dir



def canonical_qform_affine(affine: np.ndarray, code: int) -> np.ndarray:
    """Canonicalize qform via nibabel's quaternion conversion pathway."""
    import nibabel as nib

    tmp = nib.Nifti1Header()
    tmp.set_qform(affine, int(code))
    return np.asarray(tmp.get_qform(), dtype=np.float64)



def default_spatial_affine_from_header(header) -> np.ndarray:
    """Build default axis-aligned affine from existing zooms."""
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
    return aff



def forms_defined(qcode: int, scode: int) -> Tuple[bool, bool]:
    """Return (qform_defined, sform_defined)."""
    return qcode > 0, scode > 0



def matrix_handedness_sign(affine: np.ndarray) -> int:
    """Return handedness sign from determinant: +1 / -1 / 0."""
    det_val = float(np.linalg.det(np.asarray(affine[:3, :3], dtype=np.float64)))
    if det_val > 0:
        return 1
    if det_val < 0:
        return -1
    return 0



def should_rewrite_geometry(action: str, qcode: int, scode: int) -> bool:
    """Action gate for whether geometry should be recalculated for this file."""
    if action == "fix-all":
        return True
    if action == "fix-undefined":
        return qcode == 0 and scode == 0
    if action == "fix-missing-form":
        return qcode == 0 or scode == 0
    return False



def pick_target_sform(
    action: str,
    args: argparse.Namespace,
    qcode: int,
    scode: int,
    qaff: np.ndarray,
    saff: np.ndarray,
    handedness_conflict: bool,
    header,
) -> np.ndarray:
    """
    Select target sform matrix according to action and policy.

    Priority:
    1) explicit --sform-matrix
    2) fix-missing-form copy behavior
    3) unify conflict behavior (prefer existing sform)
    4) fallback default axis-aligned affine
    """
    if args.sform_matrix is not None:
        return parse_matrix_spec(args.sform_matrix)

    if action == "fix-missing-form":
        if qcode == 0 and scode > 0:
            return np.asarray(saff, dtype=np.float64)
        if scode == 0 and qcode > 0:
            return np.asarray(qaff, dtype=np.float64)

    if handedness_conflict and args.handedness_policy == "unify" and scode > 0:
        return np.asarray(saff, dtype=np.float64)

    return default_spatial_affine_from_header(header)



def default_target_codes(
    args: argparse.Namespace,
    action: str,
    qcode: int,
    scode: int,
    copied_from_code: Optional[int],
    rewrite_geometry: bool,
) -> Tuple[int, int]:
    """Resolve target qform/sform codes with explicit override precedence."""
    if args.qform_code is not None:
        q_target = int(args.qform_code)
    elif rewrite_geometry:
        if action == "fix-missing-form" and copied_from_code is not None:
            q_target = int(copied_from_code)
        else:
            q_target = 1
    else:
        q_target = int(qcode)

    if args.sform_code is not None:
        s_target = int(args.sform_code)
    elif rewrite_geometry:
        if action == "fix-missing-form" and copied_from_code is not None:
            s_target = int(copied_from_code)
        else:
            s_target = 1
    else:
        s_target = int(scode)

    return q_target, s_target



def build_target_state(
    img,
    args: argparse.Namespace,
    rewrite_geometry: bool,
    handedness_conflict: bool,
) -> TargetState:
    """
    Build the complete target header state for one image.

    This is the central matrix-first stage: once target_sform is chosen, pixdim/qfac
    are derived from that matrix and kept consistent.
    """
    hdr = img.header
    qaff, qcode = img.get_qform(coded=True)
    saff, scode = img.get_sform(coded=True)
    qcode = int(qcode)
    scode = int(scode)

    qaff = np.asarray(qaff, dtype=np.float64)
    saff = np.asarray(saff, dtype=np.float64)

    # When fix-missing-form copies one form to the other, preserve the source code.
    copied_from_code: Optional[int] = None
    if args.action == "fix-missing-form":
        if qcode == 0 and scode > 0:
            copied_from_code = scode
        elif scode == 0 and qcode > 0:
            copied_from_code = qcode

    if rewrite_geometry:
        target_sform = pick_target_sform(
            action=args.action,
            args=args,
            qcode=qcode,
            scode=scode,
            qaff=qaff,
            saff=saff,
            handedness_conflict=handedness_conflict,
            header=hdr,
        )
    else:
        if scode > 0:
            target_sform = np.asarray(saff, dtype=np.float64)
        elif qcode > 0:
            target_sform = np.asarray(qaff, dtype=np.float64)
        else:
            target_sform = default_spatial_affine_from_header(hdr)

    # Matrix is authoritative: derive spacing and qfac from target_sform.
    spacing, qfac, _ = matrix_spacing_and_qfac(target_sform[:3, :3])

    if args.pixdim_spatial is not None:
        expected_spatial = np.asarray(parse_numeric_list(args.pixdim_spatial), dtype=np.float64)
        if not np.allclose(spacing, expected_spatial, atol=CONFLICT_ATOL, rtol=CONFLICT_RTOL):
            raise ConfigError(
                f"Matrix-derived spacing {spacing.tolist()} conflicts with --pixdim-spatial {expected_spatial.tolist()}."
            )

    q_target_code, s_target_code = default_target_codes(
        args=args,
        action=args.action,
        qcode=qcode,
        scode=scode,
        copied_from_code=copied_from_code,
        rewrite_geometry=rewrite_geometry,
    )

    if q_target_code > 0:
        target_qform = canonical_qform_affine(target_sform, q_target_code)
    else:
        target_qform = np.asarray(qaff, dtype=np.float64)

    # Start from existing pixdim to preserve non-spatial dimensions unless overridden.
    target_pixdim = np.asarray(hdr["pixdim"], dtype=np.float64).copy()
    target_pixdim[0] = qfac
    target_pixdim[1:4] = spacing[:3]

    if args.pixdim is not None:
        user_pixdim = np.asarray(parse_numeric_list(args.pixdim), dtype=np.float64)
        if user_pixdim.size >= 1 and not np.allclose(
            user_pixdim[0], target_pixdim[0], atol=CONFLICT_ATOL, rtol=CONFLICT_RTOL
        ):
            raise ConfigError(
                f"--pixdim[0]={user_pixdim[0]} conflicts with matrix-derived qfac={target_pixdim[0]}."
            )
        if user_pixdim.size >= 4 and not np.allclose(
            user_pixdim[1:4], target_pixdim[1:4], atol=CONFLICT_ATOL, rtol=CONFLICT_RTOL
        ):
            raise ConfigError(
                "--pixdim[1:3] conflicts with matrix-derived spacing "
                f"{target_pixdim[1:4].tolist()}."
            )
        # Apply user pixdim then force matrix-derived qfac/spatial fields back in.
        target_pixdim[: user_pixdim.size] = user_pixdim
        target_pixdim[0] = qfac
        target_pixdim[1:4] = spacing[:3]

    xyzt_units_code: Optional[int] = None
    if args.xyzt_units is not None:
        import nibabel as nib

        xyz_u, t_u = parse_xyzt_units(args.xyzt_units)
        tmp_hdr = nib.Nifti1Header()
        tmp_hdr.set_xyzt_units(xyz=xyz_u, t=t_u)
        xyzt_units_code = int(tmp_hdr["xyzt_units"])

    descrip_bytes: Optional[bytes] = None
    if args.descrip is not None:
        descrip_bytes = encode_descrip(args.descrip)

    return TargetState(
        qform_affine=target_qform,
        sform_affine=np.asarray(target_sform, dtype=np.float64),
        qform_code=q_target_code,
        sform_code=s_target_code,
        pixdim=target_pixdim,
        xyzt_units_code=xyzt_units_code,
        cal_min=args.cal_min,
        cal_max=args.cal_max,
        descrip_bytes=descrip_bytes,
    )



def current_descrip_bytes(header) -> bytes:
    """Read descrip bytes up to first null terminator for stable comparisons."""
    raw = bytes(header["descrip"])
    return raw.split(b"\x00", 1)[0]



def has_header_drift(img, target: TargetState) -> bool:
    """Return True if any tracked header field differs from target state."""
    hdr = img.header
    qaff, qcode = img.get_qform(coded=True)
    saff, scode = img.get_sform(coded=True)

    qcode = int(qcode)
    scode = int(scode)
    qaff = np.asarray(qaff, dtype=np.float64)
    saff = np.asarray(saff, dtype=np.float64)

    if qcode != int(target.qform_code):
        return True
    if target.qform_code > 0 and not np.allclose(
        qaff, target.qform_affine, atol=DIRTY_ATOL, rtol=DIRTY_RTOL
    ):
        return True

    if scode != int(target.sform_code):
        return True
    if target.sform_code > 0 and not np.allclose(
        saff, target.sform_affine, atol=DIRTY_ATOL, rtol=DIRTY_RTOL
    ):
        return True

    # Compare full pixdim vector; includes qfac at index 0 and spatial/extra dims.
    current_pixdim = np.asarray(hdr["pixdim"], dtype=np.float64)
    if not np.allclose(current_pixdim, target.pixdim, atol=DIRTY_ATOL, rtol=DIRTY_RTOL):
        return True

    if target.xyzt_units_code is not None:
        if int(hdr["xyzt_units"]) != int(target.xyzt_units_code):
            return True

    if target.cal_min is not None:
        if not np.allclose(float(hdr["cal_min"]), float(target.cal_min), atol=DIRTY_ATOL, rtol=DIRTY_RTOL):
            return True

    if target.cal_max is not None:
        if not np.allclose(float(hdr["cal_max"]), float(target.cal_max), atol=DIRTY_ATOL, rtol=DIRTY_RTOL):
            return True

    if target.descrip_bytes is not None:
        if current_descrip_bytes(hdr) != target.descrip_bytes:
            return True

    return False



def rewrite_file(src: Path, dst: Path, target: TargetState) -> None:
    """
    Rewrite one file with target metadata while preserving storage behavior.

    - Prefer raw unscaled values when proxy supports it.
    - Keep original dtype and slope/inter metadata.
    """
    import nibabel as nib

    img = nib.load(str(src))
    hdr = img.header.copy()

    proxy = img.dataobj
    # Preserve raw stored values when available for stronger idempotent I/O behavior.
    if hasattr(proxy, "get_unscaled"):
        data = proxy.get_unscaled()
    else:
        data = proxy[:]

    out = nib.Nifti1Image(data, target.sform_affine, header=hdr)
    out.set_data_dtype(hdr.get_data_dtype())

    slope, inter = hdr.get_slope_inter()
    if slope is not None or inter is not None:
        out.header.set_slope_inter(slope, inter)

    out.set_qform(target.qform_affine, code=int(target.qform_code))
    out.set_sform(target.sform_affine, code=int(target.sform_code))

    # NIfTI header stores pixdim as float32.
    out.header["pixdim"] = np.asarray(target.pixdim, dtype=np.float32)

    if target.xyzt_units_code is not None:
        out.header["xyzt_units"] = int(target.xyzt_units_code)

    if target.cal_min is not None:
        out.header["cal_min"] = float(target.cal_min)
    if target.cal_max is not None:
        out.header["cal_max"] = float(target.cal_max)

    if target.descrip_bytes is not None:
        out.header["descrip"] = target.descrip_bytes

    nib.save(out, str(dst))



def summarize_case(c: CaseSummary) -> str:
    """Compact per-file log line prefix."""
    return f"{c.path} | qform={c.qform_code} sform={c.sform_code}"



def precheck_cases(files: Sequence[Path]) -> Tuple[List[CaseSummary], List[CaseSummary]]:
    """
    Pre-read all files to collect q/s codes and handedness conflicts.

    This decouples global fail-fast checks from per-file rewrite decisions.
    """
    import nibabel as nib

    cases: List[CaseSummary] = []
    handedness_conflicts: List[CaseSummary] = []

    for src in files:
        img = nib.load(str(src))
        qaff, qcode = img.get_qform(coded=True)
        saff, scode = img.get_sform(coded=True)

        qcode = int(qcode)
        scode = int(scode)
        qaff = np.asarray(qaff, dtype=np.float64)
        saff = np.asarray(saff, dtype=np.float64)

        case = CaseSummary(
            path=src,
            qform_code=qcode,
            sform_code=scode,
            qform_affine=qaff,
            sform_affine=saff,
        )
        cases.append(case)

        qdef, sdef = forms_defined(qcode, scode)
        if qdef and sdef:
            qsign = matrix_handedness_sign(qaff)
            ssign = matrix_handedness_sign(saff)
            if qsign != 0 and ssign != 0 and qsign != ssign:
                handedness_conflicts.append(case)

    return cases, handedness_conflicts



def main() -> None:
    """
    Main state machine.

    Order is intentional:
    1) validate config
    2) collect files
    3) precheck handedness
    4) optional fail-fast abort
    5) per-file report/rewrite with dirty gating
    """
    args = parse_args()

    try:
        import nibabel as nib  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: nibabel. Activate your nnUNet environment first, then rerun this script."
        ) from exc

    input_path = Path(args.input).expanduser().resolve()
    files = collect_files(input_path, args.pattern)
    if not files:
        print(f"No files to check (input={input_path}, pattern={args.pattern}).")
        if args.fail_on_no_match:
            raise SystemExit(EXIT_NO_FILES_MATCHED)
        return

    cases, handedness_conflicts = precheck_cases(files)
    conflict_paths = {c.path for c in handedness_conflicts}

    print(f"Matched files: {len(files)}")
    print(f"Action: {args.action}")
    print(f"Handedness policy: {args.handedness_policy}")

    # Fail-fast runs before any per-file skip/dirty decision.
    if args.handedness_policy == "fail-fast" and handedness_conflicts:
        print("[ERROR] qform/sform handedness conflicts detected:")
        for c in handedness_conflicts:
            print("  " + summarize_case(c))
        raise SystemExit(EXIT_GENERAL_ERROR)

    undefined_count = 0
    rewritten_count = 0
    skipped_count = 0

    for case in cases:
        src = case.path
        qcode = case.qform_code
        scode = case.sform_code
        undefined = qcode == 0 and scode == 0
        if undefined:
            undefined_count += 1

        if args.action == "report":
            status = "UNDEFINED" if undefined else "OK"
            print(f"{summarize_case(case)} | {status}")
            continue

        # Action-specific rewrite intent for this file.
        rewrite_geometry = should_rewrite_geometry(args.action, qcode, scode)
        if (
            args.handedness_policy == "unify"
            and case.path in conflict_paths
            and args.action in ("fix-undefined", "fix-missing-form", "fix-all")
        ):
            rewrite_geometry = True

        import nibabel as nib

        img = nib.load(str(src))
        target = build_target_state(
            img=img,
            args=args,
            rewrite_geometry=rewrite_geometry,
            handedness_conflict=(case.path in conflict_paths),
        )

        # Tolerant dirty check avoids rewrite churn from tiny float representation drift.
        dirty = has_header_drift(img, target)
        dst = output_path(src, args.inplace, args.suffix)

        if not dirty:
            skipped_count += 1
            print(f"{summarize_case(case)} | SKIP")
            continue

        rewritten_count += 1
        print(f"{src} -> {dst} | qform={qcode} sform={scode} | REWRITE")
        if not args.dry_run:
            rewrite_file(src, dst, target)

    print(
        "Summary: "
        f"total={len(files)} undefined={undefined_count} rewritten={rewritten_count} skipped={skipped_count}"
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
