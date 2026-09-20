"""Offline, private WorldSculpt dependency overlay for the validated GB10 lane."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "worldsculpt-wheelhouse.manifest.json"
WHEELS = ROOT / "wheels/worldsculpt"
VENV = ROOT / "venv-worldsculpt"
EXPECTED = {"schema": "modly.worldsculpt-wheelhouse.v1", "platform": "linux_aarch64", "python": "3.12", "cuda": "13.0", "torch": "2.12.0+cu130"}


def python_path(root: Path = ROOT) -> Path:
    return root / "venv-worldsculpt/bin/python"


def verify_wheels(manifest: Path = MANIFEST, wheel_dir: Path = WHEELS) -> list[Path]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if any(data.get(key) != value for key, value in EXPECTED.items()):
        raise RuntimeError("WorldSculpt wheelhouse lane metadata mismatch")
    entries = data.get("wheels")
    if not isinstance(entries, list) or len(entries) != 9 or any(not isinstance(item, dict) for item in entries) or len({item.get("file") for item in entries}) != 9:
        raise RuntimeError("WorldSculpt wheelhouse requires nine distinct locked wheels")
    result = []
    for item in entries:
        name, digest = item.get("file"), item.get("sha256")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".whl") or not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("Invalid WorldSculpt wheelhouse manifest entry")
        path = wheel_dir / name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"WorldSculpt wheel missing or hash mismatch: {name}")
        result.append(path)
    return result


def _probe(python: Path) -> dict:
    code = '''import json,platform,sys,sysconfig,torch
print(json.dumps({"machine":platform.machine(),"python":list(sys.version_info[:2]),"site":sysconfig.get_paths()["purelib"],"prefix":sys.prefix,"executable":sys.executable,"torch":torch.__version__,"cuda":torch.version.cuda}))'''
    completed = subprocess.run([str(python), "-c", code], text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)


def _platform_gate(info: dict) -> None:
    if sys.platform != "linux" or platform.machine() != "aarch64" or info.get("machine") != "aarch64" or info.get("python") != [3, 12] or info.get("torch") != EXPECTED["torch"] or info.get("cuda") != EXPECTED["cuda"]:
        raise RuntimeError("WorldSculpt supported only on validated Linux aarch64/Python 3.12/torch 2.12.0+cu130/CUDA 13.0 lane")


def _confined(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _validate_target(target: Path) -> None:
    if target.is_symlink() or not target.is_dir():
        raise RuntimeError("WorldSculpt venv path conflict")
    for relative in ("pyvenv.cfg", "bin", "bin/python", "lib", "lib/python3.12", "lib/python3.12/site-packages"):
        candidate = target / relative
        if candidate.is_symlink() or not candidate.exists() or not _confined(candidate.resolve(strict=True), target):
            raise RuntimeError(f"WorldSculpt venv provenance/path conflict: {relative}")
    if not (target / "pyvenv.cfg").is_file() or not (target / "bin/python").is_file():
        raise RuntimeError("WorldSculpt venv provenance/path conflict")
    # The interpreter itself, not just the directory layout, must own the target prefix.
    code = 'import json,sys,sysconfig; print(json.dumps({"prefix":sys.prefix,"executable":sys.executable,"site":sysconfig.get_paths()["purelib"],"base_prefix":sys.base_prefix}))'
    result = subprocess.run([str(target / "bin/python"), "-c", code], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    if (Path(info["prefix"]).resolve() != target or
        Path(info["executable"]).resolve() != (target / "bin/python").resolve() or
        Path(info["site"]).resolve() != (target / "lib/python3.12/site-packages").resolve() or
        Path(info["base_prefix"]).resolve() == target):
        raise RuntimeError("WorldSculpt interpreter belongs to a foreign environment")


def _verify_overlay(interpreter: Path, wheels: list[Path], overlay_site: Path) -> None:
    # Compare actual installed bytes to the hashed, vendored wheel, not inherited metadata.
    code = '''import hashlib,importlib.metadata,json,sys,zipfile
from pathlib import Path
site=Path(sys.argv[1]).resolve()
for name in sys.argv[2:]:
    with zipfile.ZipFile(name) as wheel:
        metadata=next(n for n in wheel.namelist() if n.endswith('.dist-info/METADATA'))
        fields=dict(line.split(': ',1) for line in wheel.read(metadata).decode().splitlines() if line.startswith(('Name: ', 'Version: ')))
        dist=importlib.metadata.distribution(fields['Name'])
        if dist.version != fields['Version'] or Path(dist.locate_file('')).resolve() != site or Path(dist._path).resolve().parent != site:
            raise RuntimeError('WorldSculpt overlay distribution provenance/version mismatch: '+fields['Name'])
        for entry in wheel.namelist():
            if entry.endswith('/') or entry.endswith('.dist-info/RECORD'):
                continue
            installed=site / entry
            if not installed.is_file() or installed.is_symlink() or hashlib.sha256(installed.read_bytes()).digest() != hashlib.sha256(wheel.read(entry)).digest():
                raise RuntimeError('WorldSculpt installed overlay hash mismatch: '+entry)
'''
    result = subprocess.run([str(interpreter), "-c", code, str(overlay_site), *map(str, wheels)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("WorldSculpt overlay verification failed: " + result.stderr[-2000:])


def _validate_runtime(root: Path, interpreter: Path, wheels: list[Path], overlay_site: Path) -> None:
    _validate_target(interpreter.parent.parent)
    validated = _probe(interpreter)
    _platform_gate(validated)
    _verify_overlay(interpreter, wheels, overlay_site)
    check = subprocess.run([str(interpreter), "-m", "pip", "check"], cwd=root, text=True, capture_output=True)
    if check.returncode:
        raise RuntimeError("WorldSculpt dependency closure invalid: " + check.stdout + check.stderr)
    code = "import peft,fpsample,iopath,pycocotools,ftfy,natten,o_voxel; from importlib.metadata import version; assert version('transformers') == '4.57.1'"
    imports = subprocess.run([str(interpreter), "-c", code], cwd=root, text=True, capture_output=True)
    if imports.returncode:
        raise RuntimeError("WorldSculpt native import failed: " + imports.stderr[-2000:])


def repair(root: Path = ROOT) -> dict:
    """Create/repair only venv-worldsculpt; never install into primary venv."""
    root = Path(root).resolve()
    primary = root / "venv/bin/python"
    if not primary.is_file():
        raise RuntimeError("WorldSculpt requires a prepared Pixal3D primary venv")
    info = _probe(primary)
    _platform_gate(info)
    site = Path(info["site"]).resolve(strict=True)
    if root / "venv" not in site.parents:
        raise RuntimeError("Primary site-packages path escapes Pixal3D venv")
    wheels = verify_wheels(root / MANIFEST.name, root / "wheels/worldsculpt")
    target = root / "venv-worldsculpt"
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise RuntimeError("WorldSculpt venv path conflict")
    if target.exists():
        _validate_target(target)
    else:
        created = subprocess.run([str(primary), "-m", "venv", "--copies", str(target)], cwd=root, capture_output=True, text=True)
        if created.returncode:
            raise RuntimeError("WorldSculpt venv creation failed: " + created.stderr[-2000:])
        _validate_target(target)
    overlay_site = target / "lib/python3.12/site-packages"
    link = overlay_site / "primary-pixal3d.pth"
    if link.is_symlink():
        raise RuntimeError("WorldSculpt primary site-packages link cannot be a symlink")
    if not link.is_file() or link.read_text(encoding="utf-8") != str(site) + "\n":
        link.write_text(str(site) + "\n", encoding="utf-8")
    try:
        _validate_runtime(root, python_path(root), wheels, overlay_site)
    except (RuntimeError, subprocess.CalledProcessError):
        pass  # Incomplete or drifted isolated dependencies require offline repair.
    else:
        return {"status": "already-installed", "venv": str(target), "primary_site": str(site), "wheel_count": len(wheels)}
    # Isolated packages precede inherited primary packages on sys.path.
    command = [str(python_path(root)), "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", *map(str, wheels)]
    install = subprocess.run(command, cwd=root, text=True, capture_output=True)
    if install.returncode:
        raise RuntimeError("WorldSculpt offline wheel installation failed: " + install.stderr[-2000:])
    _validate_runtime(root, python_path(root), wheels, overlay_site)
    return {"status": "installed", "venv": str(target), "primary_site": str(site), "wheel_count": len(wheels)}
