"""Checksum-pinned Windows CPython 3.12 runtime for experimental SAM3/DA3.

Provisioning is weights-free. Installation evidence is deliberately separate from
GPU readiness; the production repair path always requires the native GPU probe.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
import uuid
import warnings
from pathlib import Path, PurePosixPath

LOCK_NAME = "scene-prep-windows.lock.json"
RUNTIME_NAME = "python-scene-prep312"
EXPECTED = {"python": [3, 12], "torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128", "cuda": "12.8", "triton-windows": "3.3.1.post21"}
PTXAS_MEMBER = "triton/backends/nvidia/bin/ptxas.exe"


def _reject_alias(path: Path) -> None:
    for candidate in (path, *path.parents):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        attributes = getattr(metadata, "st_file_attributes", 0)
        if candidate.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024):
            raise RuntimeError(f"Runtime path cannot use symlinks or reparse points: {candidate}")


def load_lock(root: Path) -> dict:
    data = json.loads((root / LOCK_NAME).read_text(encoding="utf-8"))
    if data.get("schema") != "modly.scene-prep-windows.v1" or data.get("stack") != EXPECTED:
        raise RuntimeError("Windows Scene Prep immutable lock mismatch")
    return data


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(root: Path, item: dict) -> Path:
    """Use only a genuine local wheel or HTTPS artifact with a pinned hash."""
    name, digest = item["file"], item["sha256"]
    if not isinstance(name, str) or Path(name).name != name or len(digest) != 64:
        raise RuntimeError("Invalid runtime artifact identity")
    vendor = root / "wheels/scene-prep" / name
    cache = root / ".runtime-cache/scene-prep" / name
    _reject_alias(vendor); _reject_alias(cache)
    for path in (vendor, cache):
        if path.is_file() and _sha(path) == digest:
            return path
    url = item.get("url") or ""
    if not url.startswith("https://"):
        raise RuntimeError(f"Pinned runtime artifact missing or tampered: {name}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".download-", dir=cache.parent, delete=False) as temporary:
        pending = Path(temporary.name)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, temporary, length=1024 * 1024)
            temporary.flush(); os.fsync(temporary.fileno())
            if _sha(pending) != digest:
                raise RuntimeError(f"Runtime artifact checksum mismatch: {name}")
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    pending.replace(cache)
    return cache


def _runtime_valid(root: Path, item: dict) -> bool:
    target = root / RUNTIME_NAME
    _reject_alias(target)
    if not target.is_dir():
        return False
    for member, expected in item["installed_files"].items():
        path = target / member
        _reject_alias(path)
        if not path.is_file() or _sha(path) != expected:
            return False
    return True


RUNTIME_OWNER_MARKER = ".modly-python-runtime.json"


def _runtime_owner(item: dict) -> dict:
    return {"schema": "modly.owned-python-runtime.v1", "runtime": RUNTIME_NAME,
            "archive_sha256": item.get("sha256"), "version": item["version"]}


def _owned_runtime(path: Path, root: Path, item: dict, *, legacy: bool = False) -> None:
    """Prove reserved runtime custody before any rename or recursive removal."""
    if path.parent != root or not path.is_dir():
        raise RuntimeError("Portable runtime ownership/path conflict")
    _reject_alias(path)
    for directory, directories, files in os.walk(path, followlinks=False):
        for name in (*directories, *files):
            _reject_alias(Path(directory) / name)
    marker = path / RUNTIME_OWNER_MARKER
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != _runtime_owner(item):
            raise RuntimeError("Portable runtime ownership marker mismatch")
    elif not (legacy and (path.name == RUNTIME_NAME or path.name.startswith(f".{RUNTIME_NAME}-backup-")) and
              (path / "python.exe").is_file() and
              _sha(path / "python.exe") == item["installed_files"].get("python.exe")):
        raise RuntimeError("Existing portable runtime is not extension-owned; preserved without mutation")


def _probe_python(root: Path, item: dict) -> Path:
    target = root / RUNTIME_NAME
    python = target / "python.exe"
    probe = subprocess.run([str(python), "-I", "-c", "import json,platform,sys; print(json.dumps({'version':list(sys.version_info[:3]),'machine':platform.machine(),'prefix':sys.prefix}))"], capture_output=True, text=True, check=True, timeout=30)
    info = json.loads(probe.stdout)
    if info.get("version") != item["version"] or str(info.get("machine", "")).lower() not in {"amd64", "x86_64"} or Path(info["prefix"]).resolve() != target:
        raise RuntimeError("Portable Scene Prep CPython version/architecture/prefix mismatch")
    return python


def _promote_python(staging: Path, root: Path, item: dict) -> Path:
    target = root / RUNTIME_NAME
    _owned_runtime(staging, root, item)
    backup = None
    if target.exists():
        _owned_runtime(target, root, item, legacy=True)
        backup = root / f".{RUNTIME_NAME}-backup-{uuid.uuid4().hex}"
        _reject_alias(backup)
        if backup.exists():
            raise RuntimeError("Portable runtime backup path conflict")
        # On Windows a locked executable may prevent this rename. In that case
        # the previous runtime is untouched; never delete it as a workaround.
        target.replace(backup)
    try:
        staging.replace(target)
        python = _probe_python(root, item)
    except BaseException as exc:
        if backup is not None:
            try:
                _owned_runtime(backup, root, item, legacy=True)
                if target.exists():
                    _owned_runtime(target, root, item)
                    target.replace(staging)
                backup.replace(target)
            except (OSError, RuntimeError) as rollback:
                raise RuntimeError(f"Portable runtime promotion failed; previous runtime retained at {backup}; rollback blocked: {rollback}") from exc
        raise
    if backup is not None:
        _owned_runtime(backup, root, item, legacy=True)
        try:
            shutil.rmtree(backup)
        except OSError:
            warnings.warn(f"Previous owned Python runtime backup is locked and was retained: {backup}", RuntimeWarning)
    return python


def ensure_python(root: Path) -> Path:
    root = root.resolve(strict=True)
    item = load_lock(root)["python_runtime"]
    target = root / RUNTIME_NAME
    _reject_alias(target)
    if not _runtime_valid(root, item):
        archive = artifact(root, item)
        with tempfile.TemporaryDirectory(prefix=".python-scene-prep-", dir=root) as temporary:
            staging = Path(temporary)
            with zipfile.ZipFile(archive) as package:
                for info in package.infolist():
                    member = PurePosixPath(info.filename)
                    mode = info.external_attr >> 16
                    if (member.is_absolute() or ".." in member.parts or "\\" in info.filename or
                            stat.S_ISLNK(mode)):
                        raise RuntimeError("Unsafe member in pinned CPython runtime archive")
                    prefix = item["archive_prefix"].rstrip("/") + "/"
                    if not info.filename.startswith(prefix) or info.is_dir():
                        continue
                    relative = PurePosixPath(info.filename[len(prefix):])
                    destination = staging.joinpath(*relative.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(info) as source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
            for member, expected in item["installed_files"].items():
                path = staging / member
                if not path.is_file() or _sha(path) != expected:
                    raise RuntimeError(f"Official CPython runtime member checksum mismatch: {member}")
            (staging / RUNTIME_OWNER_MARKER).write_text(json.dumps(_runtime_owner(item), sort_keys=True) + "\n", encoding="utf-8")
            return _promote_python(staging, root, item)
    return _probe_python(root, item)


def managed_ptxas_environment(root: Path, environment: dict[str, str]) -> dict[str, str]:
    """Pin owned Windows children to the verified compiler from the locked wheel.

    Never fall back to an inherited override or system CUDA toolkit. Bootstrap
    venv/pip commands deliberately do not use this post-install policy.
    """
    repair = "use Modly extension Repair to reinstall the locked private triton-windows wheel"
    try:
        _reject_alias(root)
        root = root.resolve(strict=True)
        item = load_lock(root)["ptxas"]
        if item["member"] != PTXAS_MEMBER:
            raise RuntimeError("locked member is not the canonical bundled compiler")
        site = root / "venv-scene-prep/Lib/site-packages"
        compiler = site / PTXAS_MEMBER
        _reject_alias(compiler)
        metadata = compiler.lstat()
        if not stat.S_ISREG(metadata.st_mode) or not compiler.resolve(strict=True).is_relative_to(site.resolve(strict=True)):
            raise RuntimeError("compiler is not a confined regular file")
        if metadata.st_size != item["size"] or _sha(compiler) != item["sha256"]:
            raise RuntimeError("bundled compiler size/checksum mismatch")
        env = {key: value for key, value in environment.items() if key.upper() != "TRITON_PTXAS_PATH"}
        env["TRITON_PTXAS_PATH"] = str(compiler)
        result = subprocess.run([str(compiler), "--version"], cwd=root, env=env, capture_output=True,
                                text=True, check=True, timeout=30)
        version = re.search(r"\brelease\s+(\d+\.\d+),\s+V(\d+\.\d+\.\d+)\b", result.stdout + result.stderr)
        if result.returncode or version is None or version.groups() != (item["release"], item["version"]):
            raise RuntimeError("bundled compiler release/version mismatch")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Windows Scene Prep bundled PTXAS invalid: {exc}; {repair}") from exc
    identity = {"path": str(compiler), "sha256": item["sha256"], "size": metadata.st_size,
                "release": version.group(1), "version": version.group(2)}
    print("Windows Scene Prep managed PTXAS: " + json.dumps(identity, sort_keys=True), file=sys.stderr, flush=True)
    return env


def _run(command: list[str], root: Path, *, timeout: int = 900, managed_compiler: bool = False) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONUTF8": "1"}
    env.pop("PYTHONPATH", None); env.pop("PYTHONHOME", None); env.pop("PYTHONOPTIMIZE", None)
    if managed_compiler:
        env = managed_ptxas_environment(root, env)
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("Windows Scene Prep provisioning failed: " + result.stdout[-2000:] + result.stderr[-2000:])
    return result


def source_wheels(root: Path) -> list[Path]:
    return [artifact(root, item) for item in load_lock(root)["source_wheels"]]


def source_metadata(root: Path) -> dict:
    """Verify installed source bytes against immutable wheels, not invented VCS URLs."""
    wheels = source_wheels(root)
    python = root / "venv-scene-prep/Scripts/python.exe"
    code = '''import hashlib,importlib.metadata as md,json,sys,zipfile
from pathlib import Path
site=Path(sys.prefix)/'Lib/site-packages'
result={}
for path in sys.argv[1:]:
 with zipfile.ZipFile(path) as wheel:
  metadata=next(n for n in wheel.namelist() if n.endswith('.dist-info/METADATA'))
  fields=dict(line.split(': ',1) for line in wheel.read(metadata).decode().splitlines() if line.startswith(('Name: ','Version: ')))
  dist=md.distribution(fields['Name'])
  if dist.version!=fields['Version'] or Path(dist.locate_file('')).resolve()!=site.resolve():
   raise RuntimeError('Foreign source wheel installation: '+fields['Name'])
  for name in wheel.namelist():
   if name.endswith('/') or name.endswith('.dist-info/RECORD'): continue
   installed=site/name
   if not installed.is_file() or installed.is_symlink() or hashlib.sha256(installed.read_bytes()).digest()!=hashlib.sha256(wheel.read(name)).digest():
    raise RuntimeError('Installed source wheel hash mismatch: '+name)
  provenance=json.loads(dist.read_text('provenance.json'))
  result[fields['Name']]={'release_wheel':True,'wheel_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'provenance':provenance}
print(json.dumps(result,sort_keys=True))
'''
    result = _run([str(python), "-c", code, *map(str, wheels)], root, managed_compiler=True)
    metadata = json.loads(result.stdout)
    from .scene_prepare_lane import SOURCE_DISTRIBUTIONS, _normalize_repository_url
    if set(metadata) != set(SOURCE_DISTRIBUTIONS):
        raise RuntimeError("Scene Prep source wheel distribution set mismatch")
    for name, (repository, revision) in SOURCE_DISTRIBUTIONS.items():
        source = metadata[name].get("provenance", {}).get("source", {})
        if source.get("commit_id") != revision or _normalize_repository_url(source.get("repository")) != _normalize_repository_url(repository):
            raise RuntimeError("Scene Prep release-wheel upstream provenance mismatch: " + name)
    return metadata


SOURCE_IMPORT_CHECKS = ("sam3_builder", "sam3_tracker", "sam3_video_inference", "da3_adapter")


def source_import_probe_code() -> str:
    """Import the selected genuine inference boundaries; never construct models."""
    return '''import json
receipt={"schema":"modly.scene-source-imports.v1","cuda_available":None,"checks":{name:{"status":"NOT_RUN"} for name in ("sam3_builder","sam3_tracker","sam3_video_inference","da3_adapter")}}
def emit():
 print(json.dumps(receipt,sort_keys=True),flush=True)
def sam3_builder():
 from sam3.model_builder import build_sam3_video_predictor
 if not callable(build_sam3_video_predictor): raise RuntimeError("SAM3 video builder is not callable")
def sam3_tracker():
 from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
 if not callable(Sam3TrackerPredictor): raise RuntimeError("SAM3 tracker is not callable")
def sam3_video_inference():
 from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
 if not callable(Sam3VideoInferenceWithInstanceInteractivity): raise RuntimeError("SAM3 video inference class is not callable")
def da3_adapter():
 from pixal3d_extension.da3_official_adapter import load_depth_anything3
 if not callable(load_depth_anything3()): raise RuntimeError("DA3 inference class is not callable")
emit()
try:
 import torch
 receipt["cuda_available"]=bool(torch.cuda.is_available())
except Exception as exc:
 receipt["error"]=str(exc)
 receipt["error_type"]=type(exc).__name__
else:
 for name,function in (("sam3_builder",sam3_builder),("sam3_tracker",sam3_tracker),("sam3_video_inference",sam3_video_inference),("da3_adapter",da3_adapter)):
  receipt["checks"][name]={"status":"RUNNING"}
  emit()
  try:
   function()
  except Exception as exc:
   receipt["checks"][name]={"status":"FAIL","error_type":type(exc).__name__,"error":str(exc)}
  else:
   receipt["checks"][name]={"status":"PASS"}
  emit()
emit()
'''


def validate_source_imports(root: Path, *, allow_no_driver: bool = False) -> dict:
    """Missing dependencies/DLLs fail; only an exact no-driver error may be blocked."""
    python = root / "venv-scene-prep/Scripts/python.exe"
    try:
        completed = _run([str(python), "-c", source_import_probe_code()], root, timeout=180, managed_compiler=True)
        output = completed.stdout
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        raise RuntimeError("Scene Prep source import timeout; reached/unreached checks: " + output[-4000:]) from exc
    receipt = None
    for line in reversed(output.strip().splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("schema") == "modly.scene-source-imports.v1":
            receipt = candidate
            break
    checks = receipt.get("checks") if isinstance(receipt, dict) else None
    if not isinstance(checks, dict) or set(checks) != set(SOURCE_IMPORT_CHECKS) or receipt.get("error"):
        raise RuntimeError("Scene Prep source import evidence is invalid: " + json.dumps(receipt))
    checks = {name: dict(value) for name, value in checks.items()}
    blocked = False
    for check in checks.values():
        if check.get("status") == "PASS":
            continue
        no_driver = (receipt.get("cuda_available") is False and check.get("status") == "FAIL" and
                     check.get("error_type") == "RuntimeError" and
                     str(check.get("error", "")).startswith("Found no NVIDIA driver on your system."))
        if no_driver:
            check["status"] = "BLOCKED"
            blocked = True
        else:
            raise RuntimeError("Scene Prep source imports FAIL: " + json.dumps({**receipt, "checks": checks}, sort_keys=True))
    result = {**receipt, "checks": checks, "status": "BLOCKED" if blocked else "PASS",
              "runtime_prepared": False, "gpu_kernels": "NOT_RUN", "inference": "NOT_RUN", "weights": "NOT_DOWNLOADED"}
    if blocked and not allow_no_driver:
        raise RuntimeError("Scene Prep source imports BLOCKED: " + json.dumps(result, sort_keys=True))
    return result


def probe_code(*, require_gpu: bool = True) -> str:
    code = """import importlib.metadata as md,json,sys,torch,torchvision,triton,cv2,numpy,psutil,pycocotools.mask,imageio,win32api
info={'python':list(sys.version_info[:2]),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'triton-windows':md.version('triton-windows'),'torch_file':torch.__file__}
"""
    if require_gpu:
        code += """if not torch.cuda.is_available():
 raise RuntimeError('Windows Scene Prep native GPU probe BLOCKED: CUDA device unavailable')
if torch.cuda.get_device_capability(0)!=(12,0):
 raise RuntimeError('Scene Prep experimental Windows lane requires SM120')
# Pinned SAM3 EDT/connected-components APIs are exercised below; no model weights.
from pixal3d_extension.windows_sam3_probe import run_native_probe
info['native']=run_native_probe()
"""
    return code + "print(json.dumps(info,sort_keys=True))\n"


def validate(root: Path, *, require_gpu: bool = True) -> dict:
    lock = load_lock(root)
    from .scene_prepare_lane import _interpreter_custody
    owned, custody = _interpreter_custody(root)
    if not owned or not _runtime_valid(root, lock["python_runtime"]):
        raise RuntimeError("Windows Scene Prep runtime/interpreter custody mismatch; run Repair: " + json.dumps(custody))
    python = root / "venv-scene-prep/Scripts/python.exe"
    sources = source_metadata(root)
    observed = json.loads(_run([str(python), "-c", probe_code(require_gpu=False)], root, timeout=180, managed_compiler=True).stdout.strip().splitlines()[-1])
    torch_file = Path(observed.get("torch_file", "")).resolve()
    site = root / "venv-scene-prep/Lib/site-packages"
    if not torch_file.is_relative_to(site.resolve()):
        raise RuntimeError("Windows Scene Prep torch import escapes the isolated environment")
    if any(observed.get(name) != value for name, value in EXPECTED.items()):
        raise RuntimeError("Windows Scene Prep Python/torch/CUDA/Triton stack drift")
    pins = {item["name"]: item["version"] for item in lock["dependencies"]}
    versions_code = "import importlib.metadata as md,json; print(json.dumps({name:md.version(name) for name in " + repr(list(pins)) + "}))"
    installed = json.loads(_run([str(python), "-c", versions_code], root, managed_compiler=True).stdout)
    if installed != pins:
        raise RuntimeError("Windows Scene Prep locked dependency versions drifted")
    _run([str(python), "-m", "pip", "check"], root, managed_compiler=True)
    imports = validate_source_imports(root, allow_no_driver=not require_gpu)
    if require_gpu:
        observed = json.loads(_run([str(python), "-c", probe_code(require_gpu=True)], root, timeout=180, managed_compiler=True).stdout.strip().splitlines()[-1])
    return {"status": "ready" if require_gpu else "installed-not-gpu-qualified", "sources": sources,
            "source_imports": imports, **observed}


def provision(root: Path) -> dict:
    """Actual dependency provisioning; this does NOT mark runtime GPU-ready."""
    root = root.resolve(strict=True)
    lock = load_lock(root)
    primary = root / "venv/Scripts/python.exe"
    _reject_alias(primary)
    code = "import json,sys,torch; print(json.dumps([list(sys.version_info[:2]),torch.__version__,torch.version.cuda]))"
    if json.loads(_run([str(primary), "-c", code], root).stdout) != [[3, 11], "2.7.1+cu128", "12.8"]:
        raise RuntimeError("Windows Scene Prep requires the exact experimental primary Blackwell stack")
    python = ensure_python(root)
    target = root / "venv-scene-prep"
    _reject_alias(target)
    from .scene_prepare_lane import _interpreter_custody, _remove_invalid_lane
    if target.exists() and not _interpreter_custody(root)[0]:
        _remove_invalid_lane(root, target)
    if not target.exists():
        _run([str(python), "-m", "venv", "--copies", str(target)], root)
    owned, custody = _interpreter_custody(root)
    if not owned or Path(custody["base_prefix"]).resolve() != python.parent.resolve():
        raise RuntimeError("Windows Scene Prep interpreter does not belong to the extension-owned CPython 3.12 runtime")
    interpreter = target / "Scripts/python.exe"
    try:
        ready = validate(root, require_gpu=False)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        ready = None
    was_installed = ready is not None
    if ready is None:
        # Repair may replace missing/corrupt private bytes, never write through
        # a compiler-directory alias into an unrelated CUDA installation.
        _reject_alias(target / "Lib/site-packages" / PTXAS_MEMBER)
        # Each wheel is obtained and checked before pip sees it; dependency resolution
        # cannot silently replace either primary stack or the locked private stack.
        wheels = [artifact(root, item) for item in lock["dependencies"]] + source_wheels(root)
        _run([str(interpreter), "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", *map(str, wheels)], root, timeout=1800)
        ready = validate(root, require_gpu=False)
    return {"status": "already-installed" if was_installed else "installed", "venv": str(target), "runtime": ready, "runtime_prepared": False, "weights": "NOT_DOWNLOADED", "inference": "NOT_RUN"}
