#!/bin/bash -l
#$ -S /bin/bash
#$ -N invPredR
#$ -l h_rt=11:59:00
#$ -l mem=5G
#$ -l tmpfs=8G
#$ -pe smp 16
#$ -wd /home/zcemexx/Scratch
#$ -o /home/zcemexx/Scratch/logs/
#$ -j y
#$ -m abe
#$ -M zcemexx@ucl.ac.uk
# invivo.sh
# Purpose:
#   Single-case in-vivo conductivity reconstruction wrapper.
#   This script prepares environment variables and calls:
#   matlab/functions/reconstruct_conductivity_from_radiusmap.m
#
# Default data layout under DATA_ROOT:
#   phi0.nii.gz
#   mask.nii.gz
#   magnitude.nii.gz
#   segmentation.nii.gz
#   (optional auto-detect) radius.nii.gz / radius_map.nii.gz
#
# Required input:
#   RADIUS_NII (if not present in DATA_ROOT with common names)
#   OR set FIXED_RADIUS to run fixed-radius mode without predicted radius map.
#
# Common usage:
#   RADIUS_NII=/path/to/radius_map.nii.gz bash invivo.sh
#
# Fixed-radius usage:
#   FIXED_RADIUS=8 bash invivo.sh
#
# Override example:
#   DATA_ROOT=/path/to/case \
#   RADIUS_NII=/path/to/radius_map.nii.gz \
#   OUT_NII=/path/to/sigma_recon.nii.gz \
#   VOXEL_MM="0.85938,0.85938,1.5" \
#   bash invivo.sh
#
# Important env vars:
#   DATA_ROOT       case directory
#   CODE_DIR        matlab/code directory in this repo
#   MATLAB_BIN      MATLAB executable (fallback: matlab from PATH)
#   PHASE_NII       phase NIfTI input
#   MASK_NII        mask NIfTI input
#   MAG_NII         magnitude NIfTI input
#   SEG_NII         segmentation NIfTI input
#   RADIUS_NII      predicted radius-map NIfTI input
#   FIXED_RADIUS    optional fixed radius (integer > 0), bypass radius-map requirement
#   OUT_NII         output conductivity NIfTI
#   B0_T            scanner field strength (Tesla)
#   VOXEL_MM        optional voxel size override, format: "x,y,z"
#                   if unset, read from NIfTI header automatically
#   ESTIMATE_NOISE  true/false
#   RADIUS_MIN/MAX  clamp predicted radius before reconstruction
#
# Exit codes:
#   0 on success, non-zero on validation or MATLAB failure.
set -euo pipefail
module purge
module load matlab/full/r2023a/9.14

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- Config (override with env vars) ----------
DATA_ROOT="${DATA_ROOT:-/Users/apple/Documents/deeplc/UCLMRT/data/3T_in-vivo_3DGRE_Head_Neck_repeatibility}"
CODE_DIR="${CODE_DIR:-$SCRIPT_DIR/matlab/code}"
MATLAB_BIN="${MATLAB_BIN:-/Applications/MATLAB_R2025a.app/bin/matlab}"

PHASE_NII="${PHASE_NII:-$DATA_ROOT/phi0.nii.gz}"
MASK_NII="${MASK_NII:-$DATA_ROOT/mask.nii.gz}"
MAG_NII="${MAG_NII:-$DATA_ROOT/magnitude.nii.gz}"
SEG_NII="${SEG_NII:-$DATA_ROOT/segmentation.nii.gz}"

# Required: predicted radius map.
RADIUS_NII="${RADIUS_NII:-}"
FIXED_RADIUS="${FIXED_RADIUS:-}"

OUT_NII="${OUT_NII:-$DATA_ROOT/sigma_recon.nii.gz}"
B0_T="${B0_T:-3}"
VOXEL_MM="${VOXEL_MM:-}"
ESTIMATE_NOISE="${ESTIMATE_NOISE:-false}"
RADIUS_MIN="${RADIUS_MIN:-1}"
RADIUS_MAX="${RADIUS_MAX:-30}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "[ERROR] DATA_ROOT not found: $DATA_ROOT"
    exit 1
fi
if [[ ! -d "$CODE_DIR" ]]; then
    echo "[ERROR] CODE_DIR not found: $CODE_DIR"
    exit 1
fi

for f in "$PHASE_NII" "$MASK_NII" "$MAG_NII" "$SEG_NII"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Required file missing: $f"
        exit 1
    fi
done

if [[ -n "$FIXED_RADIUS" ]]; then
    if ! [[ "$FIXED_RADIUS" =~ ^[0-9]+$ ]] || [[ "$FIXED_RADIUS" -le 0 ]]; then
        echo "[ERROR] FIXED_RADIUS must be a positive integer, got: $FIXED_RADIUS"
        exit 1
    fi

    # In fixed-radius mode, use MASK_NII as a shape reference for radius input.
    # The MATLAB function will clamp all in-mask voxels to RADIUS_MIN=RADIUS_MAX=FIXED_RADIUS.
    if [[ -z "$RADIUS_NII" ]]; then
        RADIUS_NII="$MASK_NII"
    fi
    RADIUS_MIN="$FIXED_RADIUS"
    RADIUS_MAX="$FIXED_RADIUS"
else
    if [[ -z "$RADIUS_NII" ]]; then
        # Best-effort local auto-discovery for common radius-map filenames.
        for cand in \
            "$DATA_ROOT/radius.nii.gz" \
            "$DATA_ROOT/radius.nii" \
            "$DATA_ROOT/radius_map.nii.gz" \
            "$DATA_ROOT/radius_map.nii"
        do
            if [[ -f "$cand" ]]; then
                RADIUS_NII="$cand"
                break
            fi
        done
    fi

    if [[ -z "$RADIUS_NII" || ! -f "$RADIUS_NII" ]]; then
        echo "[ERROR] RADIUS_NII is required and file not found."
        echo "        Set it explicitly, for example:"
        echo "        RADIUS_NII=/path/to/radius_map.nii.gz bash $SCRIPT_DIR/invivo.sh"
        echo "        Or use fixed mode:"
        echo "        FIXED_RADIUS=8 bash $SCRIPT_DIR/invivo.sh"
        exit 1
    fi
fi

OUT_DIR="$(dirname "$OUT_NII")"
mkdir -p "$OUT_DIR"

if [[ ! -x "$MATLAB_BIN" ]]; then
    # Fallback to MATLAB available in PATH for non-macOS or custom installs.
    if command -v matlab >/dev/null 2>&1; then
        MATLAB_BIN="matlab"
    else
        echo "[ERROR] MATLAB not found. Set MATLAB_BIN or ensure 'matlab' is in PATH."
        exit 1
    fi
fi

echo "================================================="
echo "DATA_ROOT      : $DATA_ROOT"
echo "CODE_DIR       : $CODE_DIR"
echo "MATLAB_BIN     : $MATLAB_BIN"
echo "PHASE_NII      : $PHASE_NII"
echo "MASK_NII       : $MASK_NII"
echo "MAG_NII        : $MAG_NII"
echo "SEG_NII        : $SEG_NII"
echo "RADIUS_NII     : $RADIUS_NII"
if [[ -n "$FIXED_RADIUS" ]]; then
    echo "RADIUS_MODE    : fixed (FIXED_RADIUS=$FIXED_RADIUS)"
else
    echo "RADIUS_MODE    : predicted radius map"
fi
echo "OUT_NII        : $OUT_NII"
echo "B0_T           : $B0_T"
if [[ -n "$VOXEL_MM" ]]; then
    echo "VOXEL_MM       : $VOXEL_MM (env override)"
else
    echo "VOXEL_MM       : <auto from NIfTI header>"
fi
echo "ESTIMATE_NOISE : $ESTIMATE_NOISE"
echo "RADIUS_MIN/MAX : $RADIUS_MIN / $RADIUS_MAX"
echo "================================================="

export PHASE_NII MASK_NII MAG_NII SEG_NII RADIUS_NII OUT_NII
export B0_T VOXEL_MM ESTIMATE_NOISE RADIUS_MIN RADIUS_MAX

# MATLAB entrypoint: the called function reads all inputs from env vars above.
"$MATLAB_BIN" -nodesktop -nosplash -r "cd('$CODE_DIR'); addpath(pwd); addpath(fullfile(fileparts(pwd),'functions')); addpath(genpath(fullfile(fileparts(pwd),'toolboxes'))); rehash; try, reconstruct_conductivity_from_radiusmap(); catch ME, disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end; exit(0);"

echo "[DONE] Saved conductivity map: $OUT_NII"
