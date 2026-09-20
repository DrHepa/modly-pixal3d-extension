"""Provision the isolated Python 3.12 SAM3 + DA3 scene-preparation lane."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent.parent
VENV_NAME = "venv-scene-prep"
SAM3_SOURCE_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
DA3_SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
SAM3_REPOSITORY = "https://github.com/facebookresearch/sam3"
DA3_REPOSITORY = "https://github.com/ByteDance-Seed/Depth-Anything-3"
SAM3_SOURCE = f"git+{SAM3_REPOSITORY}.git@{SAM3_SOURCE_REVISION}"
DA3_SOURCE = f"git+{DA3_REPOSITORY}.git@{DA3_SOURCE_REVISION}"
PYPI_INDEX = "https://pypi.org/simple"
NVIDIA_EXTRA_INDEX = "https://pypi.nvidia.com"
TORCH_PACKAGES = ("torch==2.12.0", "torchvision==0.27.0")
PIP_FLAGS = ("--no-cache-dir", "--disable-pip-version-check")
SOURCE_DISTRIBUTIONS = {
    "sam3": (SAM3_REPOSITORY, SAM3_SOURCE_REVISION),
    "depth-anything-3": (DA3_REPOSITORY, DA3_SOURCE_REVISION),
}


def capability() -> dict:
    system = sys.platform
    machine = platform.machine().lower()
    supported = system == "linux" and machine in {"aarch64", "arm64"}
    return {
        "supported": supported,
        "status": "supported" if supported else "unsupported-platform",
        "platform": system,
        "architecture": machine,
        "reason": (
            "Scene preparation supports Linux ARM64 with Python 3.12 and CUDA >=12.6."
            if supported
            else f"Scene preparation is unavailable on {system}/{machine}; the primary Pixal3D extension remains usable."
        ),
    }


def python_path(root: Path = ROOT) -> Path:
    return Path(root) / VENV_NAME / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _python312() -> Path:
    if sys.version_info[:2] == (3, 12):
        return Path(sys.executable)
    candidate = shutil.which("python3.12") or shutil.which("python")
    if not candidate:
        raise RuntimeError("Scene preparation requires Python 3.12")
    probe = subprocess.run([candidate, "-c", "import sys; print('.'.join(map(str,sys.version_info[:2])))"], capture_output=True, text=True)
    if probe.returncode or probe.stdout.strip() != "3.12":
        raise RuntimeError("Scene preparation requires Python 3.12")
    return Path(candidate)


def install_plan(root: Path = ROOT) -> dict:
    root = Path(root)
    interpreter = python_path(root)
    requirements = root / "scene-prep-requirements.txt"
    return {
        "python": "3.12",
        "venv": str(root / VENV_NAME),
        "source_revisions": {"sam3": SAM3_SOURCE_REVISION, "depth-anything-3": DA3_SOURCE_REVISION},
        "weights_managed_by_modly": True,
        "commands": [
            [str(_python312()), "-m", "venv", "--copies", str(root / VENV_NAME)],
            [str(interpreter), "-m", "pip", "install", *PIP_FLAGS, "--upgrade", "pip==25.2"],
            [
                str(interpreter), "-m", "pip", "install", *PIP_FLAGS,
                "--index-url", PYPI_INDEX,
                "--extra-index-url", NVIDIA_EXTRA_INDEX,
                *TORCH_PACKAGES,
            ],
            [str(interpreter), "-m", "pip", "install", *PIP_FLAGS, "--requirement", str(requirements)],
            [str(interpreter), "-m", "pip", "install", *PIP_FLAGS, "--no-build-isolation", "--no-deps", SAM3_SOURCE, DA3_SOURCE],
        ],
    }


def _run(command: list[str], *, cwd: Path) -> dict:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {"ok": result.returncode == 0, "returncode": result.returncode, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}


def _normalize_repository_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if raw.startswith("git+"):
        raw = raw[4:]
    parsed = urlsplit(raw)
    if parsed.scheme.lower() != "https" or parsed.hostname is None or parsed.hostname.lower() != "github.com":
        return None
    if parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment:
        return None
    path = parsed.path.rstrip("/")
    if path.lower().endswith(".git"):
        path = path[:-4]
    parts = [part for part in path.split("/") if part]
    if len(parts) != 2:
        return None
    return f"https://github.com/{parts[0].lower()}/{parts[1].lower()}"


def _source_metadata_probe_code() -> str:
    names = json.dumps(list(SOURCE_DISTRIBUTIONS))
    return (
        "import importlib.metadata as md,json\n"
        f"names={names}\n"
        "result={}\n"
        "for name in names:\n"
        " try:\n"
        "  raw=md.distribution(name).read_text('direct_url.json')\n"
        "  result[name]=json.loads(raw) if raw else None\n"
        " except Exception as exc:\n"
        "  result[name]={'error':type(exc).__name__}\n"
        "print(json.dumps(result,sort_keys=True))\n"
    )


def _source_metadata(root: Path) -> dict | None:
    interpreter = python_path(root)
    if not interpreter.is_file() or interpreter.is_symlink():
        return None
    completed = subprocess.run(
        [str(interpreter), "-c", _source_metadata_probe_code()],
        cwd=Path(root),
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return None
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _interpreter_custody(root: Path) -> tuple[bool, dict]:
    """Prove the existing interpreter and import roots belong to this lane."""
    root = Path(root).resolve(strict=True)
    target = root / VENV_NAME
    interpreter = python_path(root)
    observation = {"status": "unsafe", "venv": str(target), "python": str(interpreter)}
    if target.is_symlink() or not target.is_dir():
        observation["reason"] = "venv is missing, not a directory, or a symlink"
        return False, observation
    current = target
    for part in interpreter.relative_to(target).parts:
        current = current / part
        if current.is_symlink():
            observation["reason"] = "python path uses a symlink"
            return False, observation
    if not interpreter.is_file():
        observation["reason"] = "python is missing or not a regular file"
        return False, observation
    target_resolved = target.resolve(strict=True)
    interpreter_resolved = interpreter.resolve(strict=True)
    if not interpreter_resolved.is_relative_to(target_resolved):
        observation["reason"] = "python escapes the scene-prep venv"
        return False, observation
    probe = (
        "import json,site,sys\n"
        "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix,'site_packages':site.getsitepackages()}))\n"
    )
    try:
        completed = subprocess.run(
            [str(interpreter), "-c", probe], cwd=root, capture_output=True, text=True,
        )
    except OSError as exc:
        observation["reason"] = f"python custody probe failed: {type(exc).__name__}"
        return False, observation
    if completed.returncode:
        observation["reason"] = "python custody probe returned nonzero"
        return False, observation
    try:
        info = json.loads(completed.stdout)
        executable = Path(info["executable"]).resolve(strict=True)
        prefix = Path(info["prefix"]).resolve(strict=True)
        base_prefix = Path(info["base_prefix"]).resolve(strict=True)
        sites = [Path(value).resolve(strict=False) for value in info["site_packages"]]
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        observation["reason"] = f"python custody response is invalid: {type(exc).__name__}"
        return False, observation
    if executable != interpreter_resolved or prefix != target_resolved:
        observation["reason"] = "python executable or sys.prefix is foreign"
        return False, observation
    if base_prefix == target_resolved or not base_prefix.is_dir():
        observation["reason"] = "sys.base_prefix does not identify a base interpreter"
        return False, observation
    if not sites or any(not site_path.is_relative_to(target_resolved) for site_path in sites):
        observation["reason"] = "site-packages escapes the scene-prep venv"
        return False, observation
    observation.update(status="owned", prefix=str(prefix), base_prefix=str(base_prefix), site_packages=[str(path) for path in sites])
    return True, observation


def _remove_invalid_lane(root: Path, target: Path) -> None:
    """Remove only the exact generated lane; never follow or delete symlinks."""
    root = Path(root).resolve(strict=True)
    if target != root / VENV_NAME or target.is_symlink() or not target.is_dir():
        raise RuntimeError("Unsafe scene-prep venv cannot be recreated automatically")
    if target.resolve(strict=True).parent != root:
        raise RuntimeError("Scene-prep venv escapes the extension root")
    shutil.rmtree(target)


def _installed_sources_exact(root: Path) -> tuple[bool, dict]:
    metadata = _source_metadata(Path(root))
    observation = {"status": "mismatch", "distributions": {}}
    if not isinstance(metadata, dict):
        observation["reason"] = "source metadata probe failed"
        return False, observation
    for name, (repository, revision) in SOURCE_DISTRIBUTIONS.items():
        direct = metadata.get(name)
        if not isinstance(direct, dict):
            observation["distributions"][name] = {"exact": False, "reason": "missing direct_url.json"}
            continue
        vcs = direct.get("vcs_info")
        url = _normalize_repository_url(direct.get("url"))
        requested = vcs.get("requested_revision") if isinstance(vcs, dict) else None
        exact = (
            isinstance(vcs, dict)
            and vcs.get("vcs") == "git"
            and url == _normalize_repository_url(repository)
            and vcs.get("commit_id") == revision
            and requested in (None, "", revision)
        )
        observation["distributions"][name] = {
            "exact": exact,
            "repository": url,
            "commit_id": vcs.get("commit_id") if isinstance(vcs, dict) else None,
            "requested_revision": requested,
        }
    exact = len(observation["distributions"]) == len(SOURCE_DISTRIBUTIONS) and all(
        item["exact"] for item in observation["distributions"].values()
    )
    observation["status"] = "exact" if exact else "mismatch"
    return exact, observation


def _runtime_probe_code() -> str:
    return (
        "import importlib.metadata as md,json,sys,torch,cv2,numpy,psutil,pycocotools.mask,imageio\n"
        "from sam3.model_builder import build_sam3_video_predictor\n"
        "from pixal3d_extension.da3_official_adapter import load_depth_anything3\n"
        "load_depth_anything3()\n"
        "def rev(n):\n"
        " d=json.loads(md.distribution(n).read_text('direct_url.json')); return d.get('vcs_info',{}).get('commit_id')\n"
        "print(json.dumps({'python':list(sys.version_info[:2]),'torch':torch.__version__,"
        "'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
        "'sam3_revision':rev('sam3'),'da3_revision':rev('depth-anything-3')}))"
    )


def validate_runtime(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    support = capability()
    if not support["supported"]:
        raise RuntimeError(support["reason"])
    interpreter = python_path(root)
    owned, custody = _interpreter_custody(root)
    if not owned:
        raise RuntimeError("Scene-prep Python is missing or unsafe; run Repair: " + json.dumps(custody, sort_keys=True))
    sources_exact, source_metadata = _installed_sources_exact(root)
    if not sources_exact:
        raise RuntimeError("Scene-prep official source metadata is missing or mismatched; run Repair: " + json.dumps(source_metadata, sort_keys=True))
    completed = subprocess.run([str(interpreter), "-c", _runtime_probe_code()], cwd=root, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("Scene-prep runtime imports failed: " + completed.stderr[-2000:])
    info = json.loads(completed.stdout)
    if info.get("python") != [3, 12]:
        raise RuntimeError("Scene-prep runtime must use Python 3.12")
    try:
        torch_major, torch_minor = (int(part) for part in str(info.get("torch", "0.0")).split("+")[0].split(".")[:2])
    except ValueError as exc:
        raise RuntimeError("Scene-prep torch version is unreadable") from exc
    if (torch_major, torch_minor) < (2, 7) or not info.get("cuda") or not info.get("cuda_available"):
        raise RuntimeError("SAM3 requires PyTorch >=2.7 and a CUDA runtime >=12.6")
    cuda = tuple(int(part) for part in str(info["cuda"]).split(".")[:2])
    if cuda < (12, 6):
        raise RuntimeError("SAM3 requires CUDA >=12.6")
    if info.get("sam3_revision") != SAM3_SOURCE_REVISION or info.get("da3_revision") != DA3_SOURCE_REVISION:
        raise RuntimeError("Scene-prep official source revision drift; run Repair")
    return {"status": "ready", "venv": str(root / VENV_NAME), "source_metadata": source_metadata, **info}


def repair(root: Path = ROOT) -> dict:
    """Create/repair only venv-scene-prep; never mutate either existing lane."""
    root = Path(root).resolve()
    support = capability()
    if not support["supported"]:
        raise RuntimeError(support["reason"])
    target = root / VENV_NAME
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise RuntimeError("Scene-prep venv path conflict")
    plan = install_plan(root)
    commands = plan["commands"]
    interpreter_exists = False
    custody_before = {"status": "missing"}
    if target.exists():
        owned, custody_before = _interpreter_custody(root)
        if owned:
            interpreter_exists = True
        else:
            _remove_invalid_lane(root, target)
    results = []
    if not interpreter_exists:
        create_result = _run(commands[0], cwd=root)
        results.append({"command": commands[0], **create_result})
        if not create_result["ok"]:
            raise RuntimeError("Scene-prep environment creation failed: " + create_result["stderr"])
        owned, custody_after = _interpreter_custody(root)
        if not owned:
            raise RuntimeError("Scene-prep environment failed custody validation before dependency installation: " + json.dumps(custody_after, sort_keys=True))
    sources_exact, source_metadata = _installed_sources_exact(root) if interpreter_exists else (False, {"status": "missing"})
    source_command = commands[-1]
    for command in commands[1:]:
        if sources_exact and command == source_command:
            continue
        result = _run(command, cwd=root)
        results.append({"command": command, **result})
        if not result["ok"]:
            raise RuntimeError("Scene-prep dependency installation failed: " + result["stderr"])
    runtime = validate_runtime(root)
    marker = target / ".modly-scene-prep.json"
    if target.exists():
        marker.write_text(json.dumps({"schema": "modly.scene-prep-lane.v1", **runtime, "sourceRevisions": plan["source_revisions"]}, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "installed" if results else "already-installed",
        "venv": str(target),
        "runtime": runtime,
        "interpreter_custody_before": custody_before,
        "source_install": {
            "status": "skipped-exact" if sources_exact else "installed",
            "metadata_before_install": source_metadata,
        },
        "commands": results,
    }
