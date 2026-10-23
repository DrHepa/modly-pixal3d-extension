#!/usr/bin/env python3
"""Build a candidate pure-Python Pixal3D MV wheel from immutable upstream source.

This does not install or publish anything. Native wheelhouse lanes still need their
own exact-stack rebuild and live validation before setup may depend on this wheel.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


SOURCE_COMMIT = "f7cf38429b0bd264f1995f0f8743a88b1c728b94"
BASE_SHA256 = "c46502c5ed195351efd150a229856095119435d356511f9343f71b8974a872f2"
SOURCE_HASHES = {
    "pixal3d/pipelines/pixal3d_mv_image_to_3d.py": "b4cf20f0cfc5028efd790d1878753cf63182941223ed75d594879c2c5ffbe12b",
    "pixal3d/pipelines/__init__.py": "ea43e99063ae919c6791a6200a842d612c8bb2d29708b9814834c2df4e36d0d2",
    "pixal3d/trainers/__init__.py": "6d8ef6bac127813d3130347e2cbe3c424e5fd09afccc3fd6d5fdcf403309f8c8",
    "pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py": "3e02246e00a047d1bc8be170a4313632c32a4077b22447f044956b9810340f56",
}
RECORD = "pixal3d_core-0.1.0+modly.dist-info/RECORD"
METADATA = "pixal3d_core-0.1.0+modly.dist-info/METADATA"
PLATFORM_REQUIREMENTS = {
    "o-voxel==0.0.1": "o-voxel-vb-ap==0.0.1",
    "cumesh==0.0.1": "cumesh-vb==1.0",
    "flex-gemm==1.0.0": "flex-gemm-ap==1.0.0",
    "nvdiffrec-render==0.0.0": "nvdiffrec-render==0.0.1",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def platform_metadata(data: bytes, *, policy: str = "stable") -> bytes:
    """Select the real Windows distributions without weakening other requirements."""
    for linux, windows in PLATFORM_REQUIREMENTS.items():
        original = f"Requires-Dist: {linux}\n".encode()
        if data.count(original) != 1:
            raise ValueError(f"unexpected base dependency metadata: {linux}")
        replacement = (
            f'Requires-Dist: {linux}; platform_system != "Windows"\n'
            f'Requires-Dist: {windows}; platform_system == "Windows"\n'
        ).encode()
        data = data.replace(original, replacement)
    if policy not in {"stable", "blackwell"}:
        raise ValueError("unknown MV metadata policy")
    if policy == "blackwell":
        for original, replacement in (
            (b"Requires-Dist: natten==0.21.0\n", b"Requires-Dist: natten==0.21.6\n"),
            (b"Version: 0.1.0+modly\n", b"Version: 0.1.0+modly.blackwell.mv\n"),
        ):
            if data.count(original) != 1:
                raise ValueError("unexpected Blackwell base metadata")
            data = data.replace(original, replacement)
    return data


def build(source: Path, base: Path, output: Path, *, policy: str = "stable") -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"unexpected Pixal3D source commit: {commit}")
    if digest(base.read_bytes()) != BASE_SHA256:
        raise ValueError("base pixal3d-core wheel checksum mismatch")
    with ZipFile(base) as archive:
        entries = {name: archive.read(name) for name in archive.namelist() if name != RECORD}
    entries[METADATA] = platform_metadata(entries[METADATA], policy=policy)
    for name, expected in SOURCE_HASHES.items():
        data = (source / name).read_bytes()
        if digest(data) != expected:
            raise ValueError(f"upstream source checksum mismatch: {name}")
        entries[name] = data
    record = RECORD
    if policy == "blackwell":
        expected = "pixal3d_core-0.1.0+modly.blackwell.mv-py3-none-any.whl"
        if output.name != expected:
            raise ValueError(f"Blackwell output must be named {expected}")
        entries = {name.replace("pixal3d_core-0.1.0+modly.dist-info/", "pixal3d_core-0.1.0+modly.blackwell.mv.dist-info/"): data
                   for name, data in entries.items()}
        record = RECORD.replace("0.1.0+modly", "0.1.0+modly.blackwell.mv")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, str]] = []
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            data = entries[name]
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            rows.append((name, "sha256=" + encoded, str(len(data))))
        rows.append((record, "", ""))
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        info = ZipInfo(record, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        archive.writestr(info, stream.getvalue())
    return digest(output.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pixal3D checkout at the pinned commit")
    parser.add_argument("--base-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("stable", "blackwell"), default="stable")
    args = parser.parse_args()
    print(build(args.source.resolve(), args.base_wheel.resolve(), args.output.resolve(), policy=args.policy))
