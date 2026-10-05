"""Exercise real, weights-free Windows setup; hosted runners cannot qualify a GPU."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED = {"torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124",
            "transformers": "4.57.3", "pixal3d-core": "0.1.0+modly", "natten": "0.21.0",
            "utils3d": "1.3+modly.headless", "pipeline": "1.0.0+modly", "moge": "2.0.0+modly",
            "naf": "0.1.0+modly", "o-voxel-vb-ap": "0.0.1", "cumesh-vb": "1.0",
            "flex-gemm-ap": "1.0.0", "drtk": "0.1.0", "flash-attn": "2.8.3",
            "nvdiffrast": "0.4.0", "nvdiffrec-render": "0.0.1"}
NATIVE = {"cumesh_vb", "flex_gemm_ap", "o_voxel_vb_ap", "nvdiffrast", "nvdiffrec_render", "natten"}


def classify_probe(probe):
    for key, expected in (("torch_version", EXPECTED["torch"]),
                          ("torchvision_version", EXPECTED["torchvision"]), ("torch_cuda_version", "12.4")):
        assert probe.get(key) == expected, (key, probe)
    assert probe.get("torch_cuda_available") is False, "Hosted runner must not be treated as GPU-qualified"
    if probe.get("error"):
        assert "Found no NVIDIA driver" in probe["error"], probe
        return "BLOCKED"
    assert set(probe.get("imports", [])) == NATIVE, probe
    assert set(probe.get("upstream_imports", [])) == {"cumesh", "flex_gemm", "o_voxel"}, probe
    assert probe.get("natten_version") == EXPECTED["natten"] and probe.get("natten_has_libnatten") is True, probe
    return "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    summary = {"package_installation": "NOT_RUN", "dependency_graph": "NOT_RUN",
               "native_imports": "NOT_RUN", "full_setup": "NOT_RUN", "hardware": "NOT_QUALIFIED",
               "weights": "NOT_RUN", "inference": "NOT_RUN"}

    def run(command, cwd, label, env=None):
        result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
        (evidence / (label + ".json")).write_text(json.dumps({"args": command, "returncode": result.returncode,
                                                            "stdout": result.stdout, "stderr": result.stderr}, indent=2))
        return result

    try:
        assert os.name == "nt" and sys.version_info[:2] == (3, 11), "Real Windows CPython 3.11 required"
        root = Path(__file__).resolve().parents[2]
        extension = Path(tempfile.mkdtemp(prefix="pixal3d-clean-install-", dir=os.environ["RUNNER_TEMP"]))
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        for name in filter(None, tracked):
            assert Path(name).parts[0] not in {"models", "venv"}, "Do not copy private weights or environments"
            target = extension / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
        payload = {"python_exe": sys.executable, "ext_dir": str(extension), "cuda_version": 128, "gpu_sm": 86}
        summary["routing_payload_not_hardware_evidence"] = payload
        env = {**os.environ, "PIP_LOG": str(evidence / "pip-complete.log"),
               "HF_HOME": str(extension / "hf-cache"), "TORCH_HOME": str(extension / "torch-cache")}
        completed = run([sys.executable, str(extension / "setup.py"), json.dumps(payload)], extension, "setup", env)
        result = json.loads(completed.stdout)
        (evidence / "setup-result.json").write_text(json.dumps(result, indent=2))
        summary["full_setup"] = "FAIL" if completed.returncode else "PASS"
        assert completed.returncode == 1 and result["status"] == "failed", "Full setup must retain no-GPU failure"
        prepared = result["wheelhouse_prepare"]
        assert prepared["selected_asset"] == "windows-x64-cp311-cuda124"
        assert prepared["downloaded"] is True and prepared["sha256_verified"] is True
        manifest = json.loads((extension / "wheelhouse.manifest.json").read_text())
        asset = next(item for item in manifest["assets"] if item["id"] == prepared["selected_asset"])
        archives = list((extension / manifest["cache"]["root"]).rglob("archive.zip"))
        assert len(archives) == 1 and hashlib.sha256(archives[0].read_bytes()).hexdigest() == asset["sha256"]
        summary["archive_sha256"] = asset["sha256"]
        install = result["dependency_install"]
        assert len(install["commands"]) == 7 and all(command["ok"] for command in install["commands"]), install
        assert install["code"] == "dependency_runtime_check_failed", install
        assert install["pip_check"]["ok"] is True, install
        python = str(extension / "venv/Scripts/python.exe")
        versions = run([python, "-c", "import json; from importlib.metadata import version; "
                        f"print(json.dumps({{name: version(name) for name in {list(EXPECTED)!r}}}))"], extension, "versions", env)
        assert versions.returncode == 0, versions.stderr
        installed = json.loads(versions.stdout)
        for name, expected in EXPECTED.items():
            actual = installed[name]
            assert actual == expected or ("+" not in expected and actual.split("+")[0] == expected), (name, actual, expected)
        check = run([python, "-m", "pip", "check"], extension, "pip-check", env)
        assert check.returncode == 0, check.stdout + check.stderr
        summary.update(package_installation="PASS", dependency_graph="PASS", installed_versions=installed)
        # Re-execute the unchanged production probe, preserving complete stdout/stderr.
        probe_result = run(install["runtime_check"]["args"], extension, "native-probe", env)
        assert probe_result.returncode == 0, probe_result.stderr
        probe = json.loads(probe_result.stdout.strip().splitlines()[-1])
        summary["native_imports"] = classify_probe(probe)
        assert summary["native_imports"] == classify_probe(install["runtime_check"]), "Probe evidence must agree"
        summary["dependency_installation"] = "PASS" if summary["native_imports"] == "PASS" else "NOT_QUALIFIED"
        model_files = [path.relative_to(extension / "models").as_posix()
                       for path in (extension / "models").rglob("*") if path.is_file()]
        assert model_files == ["pixal3d/readiness.json"], model_files
        assert not any(path.is_file() for cache in ("hf-cache", "torch-cache")
                       for path in (extension / cache).rglob("*")), "Runtime must not fetch model weights"
        summary["weights"] = "NOT_DOWNLOADED"
    except Exception as exc:
        summary["unexpected_failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (evidence / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
