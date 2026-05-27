from __future__ import annotations

import argparse
import json
import subprocess
import venv as stdlib_venv
from pathlib import Path
from typing import Any

from pixal3d_extension.pipeline_patch import patch_pipeline, restore_pipeline
from pixal3d_extension.paths import ModlyLayout, resolve_modly_layout, resolve_storage_path
from pixal3d_extension.readiness import check_readiness, check_setup_readiness


EXTENSION_ID = "pixal3d"
VENV_DIR = "venv"
VENV_MARKER = ".modly-prepared"
WHEELHOUSE_DIR = "wheels"

LOCAL_WHEEL_PACKAGES = [
    "utils3d==1.3+modly.headless",
    "pipeline==1.0.0+modly",
    "moge==2.0.0+modly",
    "natten==0.21.0",
    "naf==0.1.0+modly",
    "o-voxel==0.0.1",
    "cumesh==0.0.1",
    "flex-gemm==1.0.0",
    "nvdiffrast==0.4.0",
    "nvdiffrec-render==0.0.0",
    "pixal3d-core==0.1.0+modly",
]

RUNTIME_DIRS = [
    "models/pixal3d/generate",
    "models/pixal3d/aux/dinov3",
    "models/pixal3d/aux/rmbg",
]
READINESS_METADATA = "models/pixal3d/readiness.json"


def _workspace_item(layout: ModlyLayout, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe setup path: {relative_path}")
    return resolve_storage_path(layout, relative_path)


def _readiness_payload() -> dict[str, Any]:
    return {
        "extension_id": EXTENSION_ID,
        "weights_downloaded": False,
        "generation_ready": False,
        "next_step": "Download model assets from Modly UI.",
    }


def _create_prepare_paths(layout: ModlyLayout) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    created: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    venv_path = _workspace_item(layout, VENV_DIR)
    if (venv_path / "pyvenv.cfg").exists():
        skipped.append({"path": VENV_DIR, "reason": "already-exists"})
    else:
        stdlib_venv.EnvBuilder(with_pip=True).create(venv_path)
        created.append({"path": VENV_DIR, "kind": "directory"})
    (venv_path / VENV_MARKER).write_text(f"{EXTENSION_ID}\n", encoding="utf-8")

    for relative_path in RUNTIME_DIRS:
        path = _workspace_item(layout, relative_path)
        if path.exists():
            skipped.append({"path": relative_path, "reason": "already-exists"})
        else:
            path.mkdir(parents=True, exist_ok=True)
            created.append({"path": relative_path, "kind": "directory"})

    readiness_path = _workspace_item(layout, READINESS_METADATA)
    if readiness_path.exists():
        skipped.append({"path": READINESS_METADATA, "reason": "already-exists"})
    else:
        readiness_path.parent.mkdir(parents=True, exist_ok=True)
        created.append({"path": READINESS_METADATA, "kind": "file"})
    readiness_path.write_text(json.dumps(_readiness_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return created, skipped


def _run_setup_command(args: list[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return {
        "args": args,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "ok": completed.returncode == 0,
    }


def _install_prepare_dependencies(workspace_root: Path) -> dict[str, Any]:
    venv_python = workspace_root / VENV_DIR / "bin" / "python"
    wheelhouse = workspace_root / WHEELHOUSE_DIR
    if not venv_python.exists():
        return {"status": "failed", "code": "venv_python_missing", "venv_python": str(venv_python), "commands": []}
    if not wheelhouse.exists():
        return {"status": "failed", "code": "wheelhouse_missing", "wheelhouse": str(wheelhouse), "commands": []}

    commands = [
        [str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"],
        [str(venv_python), "-m", "pip", "install", "--find-links", WHEELHOUSE_DIR, *LOCAL_WHEEL_PACKAGES],
        [str(venv_python), "-m", "pip", "check"],
    ]
    results: list[dict[str, Any]] = []
    for command in commands:
        result = _run_setup_command(command, cwd=workspace_root)
        results.append(result)
        if not result["ok"]:
            return {"status": "failed", "code": "dependency_install_failed", "failed_command": command, "commands": results}
    return {
        "status": "installed",
        "code": "dependencies_installed",
        "venv_python": str(venv_python),
        "wheelhouse": str(wheelhouse),
        "local_wheel_packages": LOCAL_WHEEL_PACKAGES,
        "commands": results,
    }


def _load_payload(raw_payload: str | None) -> dict[str, Any]:
    if not raw_payload:
        return {}
    payload = json.loads(raw_payload)
    return {key: payload[key] for key in {"ext_dir"} if key in payload}


def run_setup(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Prepare the Pixal3D Modly extension.")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--readiness", action="store_true")
    parser.add_argument("--patch-pipeline", action="store_true")
    parser.add_argument("--restore-pipeline", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--payload-json")
    args = parser.parse_args(argv)

    setup_inputs = _load_payload(args.payload_json)
    layout = resolve_modly_layout(args.workspace_root, ext_dir=setup_inputs.get("ext_dir"))
    workspace_root = layout.ext_dir

    result: dict[str, Any] = {
        "extension_id": EXTENSION_ID,
        "entrypoint": "setup.py",
        "workspace_root": str(workspace_root),
        "resolved_paths": layout.as_dict(),
        "downloads_started": False,
        "installs_started": False,
    }

    if args.prepare:
        created, skipped = _create_prepare_paths(layout)
        dependency_install = None if args.skip_install else _install_prepare_dependencies(workspace_root)
        result.update(
            {
                "status": "prepared",
                "created": created,
                "skipped": skipped,
                "dependency_install": dependency_install,
                "installs_started": dependency_install is not None,
                "setup_readiness": check_setup_readiness(workspace_root),
                "next_steps": ["download model assets from Modly UI", "rerun readiness", "run generation"],
            }
        )
        return result

    if args.patch_pipeline:
        return {**result, **patch_pipeline(workspace_root)}
    if args.restore_pipeline:
        return {**result, **restore_pipeline(workspace_root)}
    if args.readiness:
        return {**result, "status": "readiness", "readiness": check_readiness(workspace_root)}

    return {
        **result,
        "status": "dry_run",
        "prepare_command": "python3 setup.py --prepare --json",
        "wheelhouse": WHEELHOUSE_DIR,
        "local_wheel_packages": LOCAL_WHEEL_PACKAGES,
    }


def main() -> None:
    result = run_setup()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
