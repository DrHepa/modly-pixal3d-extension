"""Lightweight integrity authority for the official valeoai/NAF release checkpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path

# Source: https://github.com/valeoai/NAF/releases/tag/model (naf_release.pth).
NAF_SIZE = 2664431
NAF_SHA256 = "c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f"


def verify_naf_checkpoint(path: Path) -> None:
    if not path.is_file() or path.stat().st_size != NAF_SIZE:
        raise RuntimeError(f"local NAF checkpoint size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != NAF_SHA256:
        raise RuntimeError(f"local NAF checkpoint SHA256 mismatch: {path}")


