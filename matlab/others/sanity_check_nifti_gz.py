#!/usr/bin/env python3
import argparse
import gzip
import os
import struct
from glob import glob


def check_one(path: str):
    with open(path, "rb") as f:
        sig = f.read(2)
    if sig != b"\x1f\x8b":
        return False, "not_gzip"

    try:
        with gzip.open(path, "rb") as f:
            header = f.read(352)
    except Exception as e:
        return False, f"gzip_read_error: {e}"

    if len(header) < 352:
        return False, f"short_header:{len(header)}"

    sizeof_hdr = struct.unpack("<I", header[0:4])[0]
    if sizeof_hdr != 348:
        return False, f"bad_sizeof_hdr:{sizeof_hdr}"

    magic = header[344:348]
    if magic not in (b"n+1\x00", b"ni1\x00"):
        return False, f"bad_magic:{magic!r}"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Sanity check .nii.gz files (gzip container + NIfTI header).")
    parser.add_argument("--root", required=True, help="Folder to recursively scan")
    args = parser.parse_args()

    paths = sorted(glob(os.path.join(args.root, "**", "*.nii.gz"), recursive=True))
    if not paths:
        print(f"[WARN] no .nii.gz files found under: {args.root}")
        return 0

    bad = []
    for p in paths:
        ok, msg = check_one(p)
        if not ok:
            bad.append((p, msg))

    print(f"[INFO] scanned={len(paths)}, bad={len(bad)}")
    for p, msg in bad:
        print(f"[BAD] {p} :: {msg}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
