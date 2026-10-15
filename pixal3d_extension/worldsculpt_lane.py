"""Offline, private WorldSculpt dependency overlay for the validated GB10 and experimental Windows Blackwell lanes."""
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


WINDOWS_EXPECTED = {"schema": "modly.worldsculpt-wheelhouse.v1", "platform": "win_amd64", "python": "3.11", "cuda": "12.8", "torch": "2.7.1+cu128"}


def expected_stack() -> dict:
    return WINDOWS_EXPECTED if sys.platform == "win32" else EXPECTED


def python_path(root: Path = ROOT) -> Path:
    return root / "venv-worldsculpt" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def primary_python(root: Path = ROOT) -> Path:
    return root / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def manifest_path(root: Path = ROOT) -> Path:
    return root / ("worldsculpt-windows-wheelhouse.manifest.json" if sys.platform == "win32" else MANIFEST.name)


def verify_wheels(manifest: Path = MANIFEST, wheel_dir: Path = WHEELS) -> list[Path]:
    if manifest == MANIFEST and sys.platform == "win32":
        manifest = manifest_path()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if any(data.get(key) != value for key, value in expected_stack().items()):
        raise RuntimeError("WorldSculpt wheelhouse lane metadata mismatch")
    entries = data.get("wheels")
    count = 10 if sys.platform == "win32" else 9
    if not isinstance(entries, list) or len(entries) != count or any(not isinstance(item, dict) for item in entries) or len({item.get("file") for item in entries}) != count:
        raise RuntimeError(f"WorldSculpt wheelhouse requires {count} distinct locked wheels")
    result = []
    for item in entries:
        name, digest = item.get("file"), item.get("sha256")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".whl") or not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("Invalid WorldSculpt wheelhouse manifest entry")
        path = wheel_dir / name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            if sys.platform == "win32" and item.get("url"):
                from .windows_scene_runtime import artifact
                path = artifact(manifest.parent, item)
            else:
                raise RuntimeError(f"WorldSculpt wheel missing or hash mismatch: {name}")
        result.append(path)
    return result


def _probe(python: Path) -> dict:
    code = '''import json,platform,sys,sysconfig,torch
print(json.dumps({"machine":platform.machine(),"python":list(sys.version_info[:2]),"site":sysconfig.get_paths()["purelib"],"prefix":sys.prefix,"executable":sys.executable,"torch":torch.__version__,"cuda":torch.version.cuda}))'''
    completed = subprocess.run([str(python), "-c", code], text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"WorldSculpt runtime probe failed (returncode={completed.returncode}, python={python})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def _platform_gate(info: dict) -> None:
    machine = platform.machine().lower()
    observed_machine = str(info.get("machine", "")).lower()
    expected = expected_stack()
    supported = ((sys.platform == "linux" and machine == observed_machine == "aarch64") or
                 (sys.platform == "win32" and machine in {"amd64", "x86_64"} and observed_machine in {"amd64", "x86_64"}))
    if (not supported or info.get("python") != [int(part) for part in expected["python"].split(".")] or
            info.get("torch") != expected["torch"] or info.get("cuda") != expected["cuda"]):
        raise RuntimeError("WorldSculpt supported only on validated Linux ARM64 cp312/cu130 or experimental Windows x64 cp311/torch 2.7.1+cu128")


def _confined(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _validate_target(target: Path) -> Path:
    if target.is_symlink() or not target.is_dir():
        raise RuntimeError("WorldSculpt venv path conflict")
    interpreter = python_path(target.parent)
    target = target.resolve(strict=True)
    for candidate in (target / "pyvenv.cfg", interpreter.parent, interpreter):
        if candidate.is_symlink() or not candidate.exists() or not _confined(candidate.resolve(strict=True), target):
            raise RuntimeError("WorldSculpt venv provenance/path conflict")
    code = 'import json,sys,sysconfig; print(json.dumps({"prefix":sys.prefix,"executable":sys.executable,"site":sysconfig.get_paths()["purelib"],"base_prefix":sys.base_prefix,"python":list(sys.version_info[:2])}))'
    result = subprocess.run([str(interpreter), "-c", code], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    site = Path(info["site"])
    if (Path(info["prefix"]).resolve() != target or
        Path(info["executable"]).resolve() != interpreter.resolve() or
        not site.is_dir() or site.is_symlink() or not _confined(site.resolve(strict=True), target) or
        Path(info["base_prefix"]).resolve() == target or
        info.get("python") != [int(part) for part in expected_stack()["python"].split(".")]):
        raise RuntimeError("WorldSculpt interpreter belongs to a foreign environment")
    current = site
    while current != target:
        if current.is_symlink():
            raise RuntimeError("WorldSculpt site-packages uses a symlink")
        current = current.parent
    return site.resolve(strict=True)


def _verify_overlay(interpreter: Path, wheels: list[Path], overlay_site: Path) -> None:
    # Compare actual installed bytes to the hashed, vendored wheel, not inherited metadata.
    code = '''import hashlib,importlib.metadata,json,sys,sysconfig,zipfile
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
            parts=Path(entry).parts
            if parts[0].endswith('.data'):
                destinations={'purelib':site,'platlib':site,'scripts':Path(sysconfig.get_path('scripts')),'data':Path(sys.prefix)}
                if len(parts)<3 or parts[1] not in destinations:
                    raise RuntimeError('Unsupported wheel installation layout: '+entry)
                installed=destinations[parts[1]].joinpath(*parts[2:])
            else:
                installed=site / entry
            if not installed.is_file() or installed.is_symlink() or hashlib.sha256(installed.read_bytes()).digest() != hashlib.sha256(wheel.read(entry)).digest():
                raise RuntimeError('WorldSculpt installed overlay hash mismatch: '+entry)
'''
    result = subprocess.run([str(interpreter), "-c", code, str(overlay_site), *map(str, wheels)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("WorldSculpt overlay verification failed: " + result.stderr[-2000:])


def _validate_runtime(root: Path, interpreter: Path, wheels: list[Path], overlay_site: Path, *, native: bool = True) -> None:
    _validate_target(interpreter.parent.parent)
    validated = _probe(interpreter)
    _platform_gate(validated)
    _verify_overlay(interpreter, wheels, overlay_site)
    check = subprocess.run([str(interpreter), "-m", "pip", "check"], cwd=root, text=True, capture_output=True)
    if check.returncode:
        raise RuntimeError("WorldSculpt dependency closure invalid: " + check.stdout + check.stderr)
    if not native:
        return
    code = "from pixal3d_extension.runtime import _prepare_runtime_compat, _install_windows_native_module_aliases, _install_natten_fallback; _prepare_runtime_compat(); _install_windows_native_module_aliases(); _install_natten_fallback(); import peft,fpsample,iopath,pycocotools,ftfy,natten,o_voxel; from importlib.metadata import version; assert version('transformers') == '4.57.1'"
    imports = subprocess.run([str(interpreter), "-c", code], cwd=root, text=True, capture_output=True)
    if imports.returncode:
        raise RuntimeError("WorldSculpt native import failed: " + imports.stderr[-2000:])


def provision(root: Path = ROOT) -> dict:
    """Install and verify real isolated dependencies, without claiming native readiness."""
    root = Path(root).resolve()
    if sys.platform == "win32":
        from .windows_scene_runtime import _reject_alias
        _reject_alias(root / "venv")
        _reject_alias(root / "venv-worldsculpt")
    primary = primary_python(root)
    if not primary.is_file():
        raise RuntimeError("WorldSculpt requires a prepared Pixal3D primary venv")
    info = _probe(primary)
    _platform_gate(info)
    if sys.platform == "win32" and (Path(info.get("prefix", "")).resolve() != root / "venv" or Path(info.get("executable", "")).resolve() != primary.resolve()):
        raise RuntimeError("Primary WorldSculpt interpreter identity mismatch")
    site = Path(info["site"]).resolve(strict=True)
    if root / "venv" not in site.parents:
        raise RuntimeError("Primary site-packages path escapes Pixal3D venv")
    wheels = verify_wheels(manifest_path(root), root / "wheels/worldsculpt")
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
    overlay_site = _validate_target(target)
    link = overlay_site / "primary-pixal3d.pth"
    if link.is_symlink():
        raise RuntimeError("WorldSculpt primary site-packages link cannot be a symlink")
    # CPython 3.11 reads .pth files with the Windows locale, even in UTF-8 mode.
    # Python ascii() preserves non-BMP codepoints without JSON surrogate pairs.
    expected_pth = f"import sys; sys.path.append({ascii(str(site))})\n".encode("ascii")
    if not link.is_file() or link.read_bytes() != expected_pth:
        link.write_bytes(expected_pth)
    try:
        _validate_runtime(root, python_path(root), wheels, overlay_site, native=False)
    except (RuntimeError, subprocess.CalledProcessError):
        pass  # Incomplete or drifted isolated dependencies require offline repair.
    else:
        return {"status": "already-installed", "venv": str(target), "primary_site": str(site), "wheel_count": len(wheels), "runtime_prepared": False, "inference_validated": False}
    # Isolated packages precede inherited primary packages on sys.path.
    command = [str(python_path(root)), "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", *map(str, wheels)]
    install = subprocess.run(command, cwd=root, text=True, capture_output=True)
    if install.returncode:
        raise RuntimeError("WorldSculpt offline wheel installation failed: " + install.stderr[-2000:])
    _validate_runtime(root, python_path(root), wheels, overlay_site, native=False)
    return {"status": "installed", "venv": str(target), "primary_site": str(site), "wheel_count": len(wheels), "runtime_prepared": False, "inference_validated": False}


def repair(root: Path = ROOT) -> dict:
    """Normal UI repair: dependency provisioning AND mandatory native imports."""
    root = Path(root).resolve()
    result = provision(root)
    wheels = verify_wheels(manifest_path(root), root / "wheels/worldsculpt")
    site = _validate_target(root / "venv-worldsculpt")
    _validate_runtime(root, python_path(root), wheels, site)
    return {**result, "native_imports": "PASS", "inference_validated": False}
