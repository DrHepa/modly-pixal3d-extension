from __future__ import annotations

import argparse
import json
import subprocess
import sys
import venv as stdlib_venv
from pathlib import Path
from typing import Any

from pixal3d_extension.pipeline_patch import patch_pipeline, restore_pipeline
from pixal3d_extension.paths import ModlyLayout, resolve_modly_layout, resolve_storage_path
from pixal3d_extension.readiness import check_readiness, check_setup_readiness
from modly_wheelhouse import (
    WheelhouseError,
    detect_runtime_lane,
    load_manifest,
    prepare_wheelhouse,
    resolve_verified_fallback,
    validate_manifest,
)


EXTENSION_ID = "pixal3d"
VENV_DIR = "venv"
VENV_MARKER = ".modly-prepared"
WHEELHOUSE_DIR = "wheels"
WHEELHOUSE_MANIFEST = "wheelhouse.manifest.json"

LOCAL_WHEEL_PACKAGES = [
    "utils3d==1.3+modly.headless",
    "pipeline==1.0.0+modly",
    "moge==2.0.0+modly",
    "naf==0.1.0+modly",
    "o-voxel==0.0.1",
    "cumesh==0.0.1",
    "flex-gemm==1.0.0",
    "nvdiffrast==0.4.0",
    "nvdiffrec-render==0.0.0",
    "pixal3d-core==0.1.0+modly",
]

WINDOWS_LOCAL_WHEEL_PACKAGES = [
    "utils3d==1.3+modly.headless",
    "pipeline==1.0.0+modly",
    "moge==2.0.0+modly",
    "naf==0.1.0+modly",
    "o-voxel-vb-ap==0.0.1",
    "cumesh-vb==1.0",
    "flex-gemm-ap==1.0.0",
    "drtk==0.1.0",
    "flash-attn==2.8.3",
    "nvdiffrast==0.4.0",
    "nvdiffrec-render==0.0.1",
    "pixal3d-core==0.1.0+modly",
]

OPTIONAL_NATTEN_PACKAGES = ["natten==0.21.0"]

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


def _install_prepare_dependencies(workspace_root: Path, *, wheelhouse_path: Path | None = None) -> dict[str, Any]:
    venv_python = _venv_python_path(workspace_root)
    wheelhouse = wheelhouse_path or (workspace_root / WHEELHOUSE_DIR)
    if not venv_python.exists():
        return {"status": "failed", "code": "venv_python_missing", "venv_python": str(venv_python), "commands": []}
    if not wheelhouse.exists():
        return {"status": "failed", "code": "wheelhouse_missing", "wheelhouse": str(wheelhouse), "commands": []}

    local_wheel_packages = _local_wheel_packages_for_wheelhouse(wheelhouse)
    commands = [
        [str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"],
        [str(venv_python), "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse), *local_wheel_packages],
    ]
    if _wheelhouse_contains_natten(wheelhouse):
        commands.append([str(venv_python), "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse), *OPTIONAL_NATTEN_PACKAGES])
    commands.append([str(venv_python), "-m", "pip", "check"])
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
        "local_wheel_packages": local_wheel_packages,
        "optional_natten_packages": OPTIONAL_NATTEN_PACKAGES,
        "natten_runtime": _natten_runtime_status(venv_python, workspace_root),
        "commands": results,
    }


def _wheelhouse_contains_natten(wheelhouse: Path) -> bool:
    return any(wheelhouse.glob("natten-*.whl"))


def _local_wheel_packages_for_wheelhouse(wheelhouse: Path) -> list[str]:
    if any(wheelhouse.glob("*win_amd64.whl")):
        return WINDOWS_LOCAL_WHEEL_PACKAGES
    return LOCAL_WHEEL_PACKAGES


def _natten_runtime_status(venv_python: Path, workspace_root: Path) -> dict[str, Any]:
    result = _run_setup_command(
        [
            str(venv_python),
            "-c",
            (
                "import json\n"
                "try:\n"
                "    import natten\n"
                "    print(json.dumps({'importable': True, 'version': getattr(natten, '__version__', None), "
                "'HAS_LIBNATTEN': bool(getattr(natten, 'HAS_LIBNATTEN', False))}, sort_keys=True))\n"
                "except Exception as exc:\n"
                "    print(json.dumps({'importable': False, 'HAS_LIBNATTEN': False, 'error': f'{type(exc).__name__}: {exc}'}, sort_keys=True))\n"
            ),
        ],
        cwd=workspace_root,
    )
    try:
        payload = json.loads(result.get("stdout_tail", "").strip().splitlines()[-1])
    except Exception:
        payload = {"importable": False, "HAS_LIBNATTEN": False, "error": "natten runtime probe did not return JSON"}
    payload["strict_naf_available"] = bool(payload.get("importable") and payload.get("HAS_LIBNATTEN"))
    payload["fallback_required"] = not payload["strict_naf_available"]
    return payload


def _venv_python_path(workspace_root: Path) -> Path:
    venv_root = workspace_root / VENV_DIR
    candidates = []
    if sys.platform.startswith("win"):
        candidates.extend([venv_root / "Scripts" / "python.exe", venv_root / "bin" / "python"])
    else:
        candidates.extend([venv_root / "bin" / "python", venv_root / "Scripts" / "python.exe"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _prepare_wheelhouse_for_setup(workspace_root: Path) -> dict[str, Any]:
    manifest = load_manifest(workspace_root / WHEELHOUSE_MANIFEST)
    runtime_evidence = detect_runtime_lane()
    try:
        return prepare_wheelhouse(manifest, runtime_evidence, workspace_root)
    except WheelhouseError as exc:
        if exc.code not in {"network", "auth_required"}:
            raise
        fallback = resolve_verified_fallback(manifest, workspace_root, runtime_evidence)
        return {
            **fallback,
            "fallback_mode": manifest.get("fallback", {}).get("mode", "migration_only"),
            "fallback_reason": exc.code,
            "release_failure": exc.observation,
        }


def _prepared_wheelhouse_path(observation: dict[str, Any]) -> Path:
    path = observation.get("wheelhouse_path") or observation.get("fallback_path")
    if not isinstance(path, str) or not path:
        raise WheelhouseError("wheelhouse_path_missing", "Wheelhouse preparation did not return a local install path")
    return Path(path)


def _wheelhouse_manifest_observation(workspace_root: Path) -> dict[str, Any]:
    manifest_path = workspace_root / WHEELHOUSE_MANIFEST
    if not manifest_path.exists():
        return {"status": "missing", "path": WHEELHOUSE_MANIFEST}
    try:
        manifest = load_manifest(manifest_path)
        validate_manifest(manifest)
    except (OSError, json.JSONDecodeError, WheelhouseError) as exc:
        code = getattr(exc, "code", "invalid_manifest")
        return {"status": "failed", "failure_code": code, "path": WHEELHOUSE_MANIFEST}
    return {
        "status": "available",
        "release_tag": manifest["release"]["tag"],
        "wheelhouse_version": manifest["wheelhouse_version"],
        "asset_count": len(manifest["assets"]),
    }


def _load_payload(raw_payload: str | None) -> dict[str, Any]:
    if not raw_payload:
        return {}
    payload = json.loads(raw_payload)
    return {key: payload[key] for key in {"ext_dir"} if key in payload}


def _coerce_payload_arg(explicit_payload: str | None, positional_payload: str | None, unknown_args: list[str]) -> str | None:
    if explicit_payload:
        return explicit_payload
    if positional_payload:
        return positional_payload
    if unknown_args and unknown_args[0].lstrip().startswith("{"):
        return " ".join(unknown_args)
    return None


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
    parser.add_argument("positional_payload_json", nargs="?", help="Modly install payload JSON. Modly may pass this as a positional argument.")
    args, unknown_args = parser.parse_known_args(argv)

    raw_payload = _coerce_payload_arg(args.payload_json, args.positional_payload_json, unknown_args)
    setup_inputs = _load_payload(raw_payload)
    layout = resolve_modly_layout(args.workspace_root, ext_dir=setup_inputs.get("ext_dir"))
    workspace_root = layout.ext_dir
    prepare_requested = args.prepare or bool(raw_payload)

    result: dict[str, Any] = {
        "extension_id": EXTENSION_ID,
        "entrypoint": "setup.py",
        "workspace_root": str(workspace_root),
        "resolved_paths": layout.as_dict(),
        "downloads_started": False,
        "installs_started": False,
        "wheelhouse_manifest": _wheelhouse_manifest_observation(workspace_root),
    }

    if prepare_requested:
        created, skipped = _create_prepare_paths(layout)
        wheelhouse_prepare = None
        dependency_install = None
        if not args.skip_install:
            try:
                wheelhouse_prepare = _prepare_wheelhouse_for_setup(workspace_root)
                dependency_install = _install_prepare_dependencies(
                    workspace_root,
                    wheelhouse_path=_prepared_wheelhouse_path(wheelhouse_prepare),
                )
            except (OSError, json.JSONDecodeError, WheelhouseError) as exc:
                code = getattr(exc, "code", "wheelhouse_prepare_failed")
                wheelhouse_prepare = getattr(
                    exc,
                    "observation",
                    {"status": "failed", "failure_code": code, "downloads_started": False, "installs_started": False},
                )
                return {
                    **result,
                    "status": "failed",
                    "created": created,
                    "skipped": skipped,
                    "wheelhouse_prepare": wheelhouse_prepare,
                    "dependency_install": None,
                    "installs_started": False,
                    "setup_readiness": check_setup_readiness(workspace_root),
                    "next_steps": ["preseed a verified wheelhouse release asset or fix the wheelhouse manifest"],
                }
        result.update(
            {
                "status": "prepared",
                "created": created,
                "skipped": skipped,
                "wheelhouse_prepare": wheelhouse_prepare,
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
        "wheelhouse_manifest_path": WHEELHOUSE_MANIFEST,
        "local_wheel_packages": LOCAL_WHEEL_PACKAGES,
    }


def main() -> None:
    result = run_setup()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
