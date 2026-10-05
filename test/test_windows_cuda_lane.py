import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import setup
from modly_wheelhouse import WheelhouseError, detect_runtime_lane, load_manifest, select_asset


ROOT = Path(__file__).resolve().parents[1]


def windows_runtime(payload, python_tag="cp311", machine="AMD64"):
    return detect_runtime_lane(payload, system="Windows", machine=machine, python_tag=python_tag)


class WindowsCudaLaneTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(ROOT / "wheelhouse.manifest.json")

    def test_reporter_and_laptop_new_driver_payloads_select_published_abi(self):
        for sm in (86, 75):
            with self.subTest(sm=sm):
                runtime = windows_runtime({"cuda_version": 128, "gpu_sm": sm})
                self.assertEqual(runtime["accelerator_lane"], "cuda124")
                self.assertEqual(runtime["cuda_version"], "12.8")
                self.assertEqual(runtime["gpu_sm"], str(sm))
                self.assertEqual(select_asset(self.manifest, runtime)["id"], "windows-x64-cp311-cuda124")

    def test_driver_versions_and_encodings_preserve_evidence_for_both_python_abis(self):
        versions = ((124, "12.4"), (125, "12.5"), (126, "12.6"), (128, "12.8"), (130, "13.0"))
        for python_tag in ("cp311", "cp312"):
            for compact, dotted in versions:
                for encoded in (compact, str(compact), dotted, float(dotted), dotted + ".1"):
                    with self.subTest(python_tag=python_tag, encoded=encoded):
                        runtime = windows_runtime({"cuda_version": encoded, "gpu_sm": "sm_86"}, python_tag)
                        self.assertEqual(runtime["accelerator_lane"], "cuda124")
                        self.assertEqual(runtime["cuda_version"], dotted)
                        self.assertEqual(runtime["gpu_sm"], "86")
                        self.assertEqual(select_asset(self.manifest, runtime)["id"], f"windows-x64-{python_tag}-cuda124")

    def test_pre_blackwell_sm_encodings_do_not_claim_kernel_qualification(self):
        for sm, normalized in ((61, "61"), (70, "70"), (75, "75"), ("8.6", "86"), ("compute_89", "89"), (90, "90")):
            with self.subTest(sm=sm):
                runtime = windows_runtime({"cuda_version": 128, "gpu_sm": sm})
                self.assertEqual(runtime["accelerator_lane"], "cuda124")
                self.assertEqual(runtime["gpu_sm"], normalized)

    def test_insufficient_driver_capability_cannot_match_published_windows_assets(self):
        for version in (118, 120, 123, "12.3.1"):
            with self.subTest(version=version):
                runtime = windows_runtime({"cuda_version": version, "gpu_sm": 86})
                with self.assertRaises(WheelhouseError) as failure:
                    select_asset(self.manifest, runtime)
                self.assertEqual(failure.exception.code, "unsupported_lane")

    def test_blackwell_never_selects_cuda124_even_with_older_driver_metadata(self):
        for sm in (100, 101, 103, 110, 120, 121):
            for version in (124, 126, 128, 130):
                with self.subTest(sm=sm, version=version):
                    with self.assertRaises(WheelhouseError) as failure:
                        select_asset(self.manifest, windows_runtime({"cuda_version": version, "gpu_sm": sm}))
                    self.assertEqual(failure.exception.code, "unsupported_lane")
        candidate = windows_runtime({"cuda_version": 128, "gpu_sm": 120})
        self.assertEqual(candidate["accelerator_lane"], "cuda128-blackwell")

    def test_invalid_and_partial_metadata_fail_closed(self):
        cases = [
            ({"cuda_version": 128}, "incomplete_runtime_evidence"),
            ({"gpu_sm": 86}, "incomplete_runtime_evidence"),
            ({"cuda_version": None, "gpu_sm": 86}, "incomplete_runtime_evidence"),
            ({"cuda_version": None, "gpu_sm": None}, "invalid_runtime_evidence"),
            ({"cuda_version": "12.x", "gpu_sm": 86}, "invalid_runtime_evidence"),
            ({"cuda_version": True, "gpu_sm": 86}, "invalid_runtime_evidence"),
            ({"cuda_version": 128, "gpu_sm": False}, "invalid_runtime_evidence"),
            ({"cuda_version": 128, "gpu_sm": 0}, "invalid_runtime_evidence"),
            ({"cuda_version": 128, "gpu_sm": "8.60"}, "invalid_runtime_evidence"),
        ]
        for payload, code in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(WheelhouseError) as failure:
                    windows_runtime(payload)
                self.assertEqual(failure.exception.code, code)

    def test_unsupported_os_arch_and_python_do_not_gain_compatibility_mapping(self):
        tuples = (("Windows", "ARM64", "cp311"), ("Windows", "AMD64", "cp310"),
                  ("Windows", "AMD64", "cp313"), ("Darwin", "AMD64", "cp311"))
        for system, machine, python_tag in tuples:
            with self.subTest(system=system, machine=machine, python_tag=python_tag):
                runtime = detect_runtime_lane({"cuda_version": 128, "gpu_sm": 86}, system=system, machine=machine, python_tag=python_tag)
                self.assertEqual(runtime["accelerator_lane"], "cuda128")
                with self.assertRaises(WheelhouseError) as failure:
                    select_asset(self.manifest, runtime)
                self.assertEqual(failure.exception.code, "unsupported_lane")

    def test_linux_exact_driver_mapping_and_no_metadata_behavior_are_unchanged(self):
        for machine in ("x86_64", "aarch64"):
            for version, lane in ((124, "cuda124"), (126, "cuda126"), (128, "cuda128"), (130, "cuda130")):
                with self.subTest(machine=machine, version=version):
                    runtime = detect_runtime_lane({"cuda_version": version, "gpu_sm": 86}, system="Linux", machine=machine, python_tag="cp312")
                    self.assertEqual(runtime["accelerator_lane"], lane)
        for payload in ({}, {"ext_dir": "/extension"}):
            self.assertEqual(windows_runtime(payload), {"os": "windows", "arch": "x64", "python_tag": "cp311", "accelerator_lane": "cuda124"})

    def test_setup_failures_precede_path_creation_download_and_install(self):
        cases = [({"cuda_version": 123, "gpu_sm": 86}, "cp311", "AMD64"),
                 ({"cuda_version": 128, "gpu_sm": 100}, "cp311", "AMD64"),
                 ({"cuda_version": 124, "gpu_sm": 120}, "cp311", "AMD64"),
                 ({"cuda_version": 128, "gpu_sm": 120}, "cp311", "AMD64"),
                 ({"cuda_version": 128}, "cp311", "AMD64"),
                 ({"cuda_version": "invalid", "gpu_sm": 86}, "cp311", "AMD64"),
                 ({"cuda_version": None, "gpu_sm": None}, "cp311", "AMD64"),
                 ({"cuda_version": 128, "gpu_sm": 86}, "cp310", "AMD64"),
                 ({"cuda_version": 128, "gpu_sm": 86}, "cp311", "ARM64")]
        for payload, python_tag, machine in cases:
            with self.subTest(payload=payload, python_tag=python_tag, machine=machine), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "extension"
                root.mkdir()
                shutil.copy2(ROOT / "wheelhouse.manifest.json", root / "wheelhouse.manifest.json")
                with patch.object(setup, "detect_runtime_lane", side_effect=lambda value: windows_runtime(value, python_tag, machine)), \
                     patch.object(setup, "_create_prepare_paths") as create, \
                     patch.object(setup, "prepare_wheelhouse") as download, \
                     patch.object(setup, "_install_prepare_dependencies") as install:
                    result = setup.run_setup([json.dumps({"ext_dir": str(root), **payload})])
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["downloads_started"])
                self.assertFalse(result["installs_started"])
                create.assert_not_called()
                download.assert_not_called()
                install.assert_not_called()
                self.assertEqual(sorted(path.name for path in root.iterdir()), ["wheelhouse.manifest.json"])

    def test_setup_json_and_legacy_forms_choose_same_windows_published_lane(self):
        for sm in (75, 86):
            for legacy in (False, True):
                with self.subTest(sm=sm, legacy=legacy), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp) / "extension"
                    root.mkdir()
                    shutil.copy2(ROOT / "wheelhouse.manifest.json", root / "wheelhouse.manifest.json")
                    arguments = ["/host/python.exe", str(root), str(sm), "128"] if legacy else [json.dumps({"ext_dir": str(root), "gpu_sm": sm, "cuda_version": 128})]
                    with patch.object(setup, "detect_runtime_lane", side_effect=windows_runtime), \
                         patch.object(setup, "_create_prepare_paths", return_value=([], [])):
                        result = setup.run_setup([*arguments, "--skip-install"])
                    self.assertEqual(result["status"], "prepared")
                    self.assertEqual(result["runtime_lane_preflight"]["selected_asset"], "windows-x64-cp311-cuda124")
                    self.assertEqual(result["runtime_evidence"]["cuda_version"], "12.8")
                    self.assertEqual(result["runtime_evidence"]["gpu_sm"], str(sm))


class WindowsTorchAbiTests(unittest.TestCase):
    def test_windows_cu124_probe_requires_pinned_torch_torchvision_and_cuda_only(self):
        base = {"ok": True, "torch_version": "2.6.0+cu124", "torchvision_version": "0.21.0+cu124", "torch_cuda_version": "12.4"}
        for python_tag in ("cp311", "cp312"):
            policy = setup._dependency_policy({"os": "windows", "arch": "x64", "python_tag": python_tag, "accelerator_lane": "cuda124"})
            self.assertFalse(policy["strict_validation"])
            self.assertTrue(setup._validate_runtime_probe(base, policy)["ok"])
            for key, wrong in (("torch_version", "2.7.1+cu128"), ("torchvision_version", "0.21.0"), ("torch_cuda_version", "12.8")):
                for value in (wrong, None):
                    with self.subTest(python_tag=python_tag, key=key, value=value):
                        result = setup._validate_runtime_probe({**base, key: value}, policy)
                        self.assertFalse(result["ok"])
                        self.assertTrue(any(key in error for error in result["validation_errors"]))

    def test_other_platform_probes_retain_existing_non_strict_validation(self):
        for arch in ("x64", "aarch64"):
            policy = setup._dependency_policy({"os": "linux", "arch": arch, "python_tag": "cp312", "accelerator_lane": "cuda124"})
            self.assertTrue(setup._validate_runtime_probe({"ok": True, "torch_cuda_version": "13.0"}, policy)["ok"])
        policy = setup._dependency_policy({"os": "windows", "arch": "x64", "python_tag": "cp310", "accelerator_lane": "cuda124"})
        self.assertTrue(setup._validate_runtime_probe({"ok": True}, policy)["ok"])

    def test_existing_runtime_cuda_check_applies_windows_abi_validation(self):
        runtime = {"os": "windows", "arch": "x64", "python_tag": "cp311", "accelerator_lane": "cuda124"}
        base = {"ok": True, "transformers_version": setup.TRANSFORMERS_VERSION, "torch_version": "2.6.0+cu124", "torchvision_version": "0.21.0+cu124", "torch_cuda_version": "12.4"}
        for cuda, valid in (("12.4", True), ("12.8", False)):
            with self.subTest(cuda=cuda), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                response = {"ok": True, "stdout_tail": json.dumps({**base, "torch_cuda_version": cuda}), "stderr_tail": "", "returncode": 0}
                with patch.object(setup, "_run_setup_command", return_value=response) as command:
                    result = setup._runtime_cuda_check(Path("/venv/python.exe"), root, root, runtime)
                self.assertEqual(result["ok"], valid)
                compile(command.call_args.args[0][-1], "<runtime-probe>", "exec")


if __name__ == "__main__":
    unittest.main()
