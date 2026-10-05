"""Narrow loader for the official DA3 inference API.

DA3's public ``api.py`` imports every optional exporter eagerly, including
COLMAP and Gaussian-splatting dependencies that are not available on all
Modly platforms. Scene preparation never exports through DA3 and never passes
external camera poses, so this adapter replaces only those two unused import
boundaries with fail-closed modules before loading the official API class.
The model, preprocessing, forward pass, output processor, and local checkpoint
loading all remain the pinned official implementation.
"""

from __future__ import annotations

import sys
from types import ModuleType


def _unsupported(*_args, **_kwargs):
    raise RuntimeError("The scene-preparation DA3 adapter does not enable optional DA3 export or pose-alignment APIs")


def _install_unused_boundary(name: str, symbol: str) -> None:
    if name in sys.modules:
        return
    module = ModuleType(name)
    setattr(module, symbol, _unsupported)
    module.__dict__["__modly_fail_closed__"] = True
    sys.modules[name] = module


def load_depth_anything3():
    """Return the pinned official ``DepthAnything3`` class for inference only."""
    _install_unused_boundary("depth_anything_3.utils.export", "export")
    _install_unused_boundary("depth_anything_3.utils.pose_align", "align_poses_umeyama")
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3
