import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pixal3d_extension import scene_prepare_lane


def _exact_source_metadata():
    return {
        "sam3": {
            "url": "https://github.com/facebookresearch/sam3.git",
            "vcs_info": {
                "vcs": "git",
                "commit_id": scene_prepare_lane.SAM3_SOURCE_REVISION,
                "requested_revision": scene_prepare_lane.SAM3_SOURCE_REVISION,
            },
        },
        "depth-anything-3": {
            "url": "https://github.com/ByteDance-Seed/Depth-Anything-3.git/",
            "vcs_info": {
                "vcs": "git",
                "commit_id": scene_prepare_lane.DA3_SOURCE_REVISION,
                "requested_revision": scene_prepare_lane.DA3_SOURCE_REVISION,
            },
        },
    }


class ScenePrepareLaneTests(unittest.TestCase):
    def test_plan_is_python312_isolated_pinned_and_weight_free(self):
        plan = scene_prepare_lane.install_plan(Path("/extension"))
        self.assertEqual(plan["python"], "3.12")
        self.assertEqual(plan["venv"], "/extension/venv-scene-prep")
        joined = " ".join(" ".join(command) for command in plan["commands"])
        self.assertIn(scene_prepare_lane.SAM3_SOURCE_REVISION, joined)
        self.assertIn(scene_prepare_lane.DA3_SOURCE_REVISION, joined)
        self.assertNotIn("facebook/sam3", joined)
        self.assertNotIn("depth-anything/DA3-BASE", joined)
        self.assertNotIn("hf_hub_download", joined)
        self.assertIn("--no-build-isolation", joined)
        torch_command = next(command for command in plan["commands"] if "torch==2.12.0" in command)
        self.assertEqual(torch_command[torch_command.index("--index-url") + 1], "https://pypi.org/simple")
        self.assertEqual(torch_command[torch_command.index("--extra-index-url") + 1], "https://pypi.nvidia.com")
        self.assertLess(torch_command.index("--index-url"), torch_command.index("--extra-index-url"))
        probe = scene_prepare_lane._runtime_probe_code()
        compile(probe, "<scene-prep-probe>", "exec")
        self.assertIn("psutil", probe)
        self.assertIn("pycocotools.mask", probe)
        self.assertIn("imageio", probe)
        requirements = (scene_prepare_lane.ROOT / "scene-prep-requirements.txt").read_text(encoding="utf-8")
        self.assertIn("psutil==7.2.2", requirements)
        self.assertIn("pycocotools==2.0.10", requirements)
        self.assertIn("imageio==2.37.3", requirements)
        metadata_probe = scene_prepare_lane._source_metadata_probe_code()
        compile(metadata_probe, "<scene-prep-source-probe>", "exec")
        self.assertIn("direct_url.json", metadata_probe)

    def test_mv_da3_readiness_does_not_require_sam_import_or_source_metadata(self):
        metadata = _exact_source_metadata()
        metadata.pop("sam3")
        response = json.dumps({
            "python": [3, 12], "torch": "2.12.0", "cuda": "13.0",
            "cuda_available": True, "da3_revision": scene_prepare_lane.DA3_SOURCE_REVISION,
        })
        with patch.object(scene_prepare_lane, "da3_capability", return_value={"supported": True}), \
             patch.object(scene_prepare_lane, "_interpreter_custody", return_value=(True, {"status": "owned"})), \
             patch.object(scene_prepare_lane, "_source_metadata", return_value=metadata), \
             patch.object(scene_prepare_lane.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=response, stderr="")) as run:
            result = scene_prepare_lane.validate_da3_runtime(Path("/extension"))
        self.assertEqual(result["status"], "ready")
        probe = run.call_args.args[0][-1]
        self.assertIn("load_depth_anything3", probe)
        self.assertNotIn("sam3", probe)

    def test_repair_skips_vcs_install_only_when_both_sources_are_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            interpreter = scene_prepare_lane.python_path(root)
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            run_result = {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}
            with patch.object(scene_prepare_lane, "capability", return_value={"supported": True}), \
                 patch.object(scene_prepare_lane, "_interpreter_custody", return_value=(True, {"status": "owned"})), \
                 patch.object(scene_prepare_lane, "_source_metadata", return_value=_exact_source_metadata()), \
                 patch.object(scene_prepare_lane, "_run", return_value=run_result) as run, \
                 patch.object(scene_prepare_lane, "validate_runtime", return_value={"status": "ready"}) as validate:
                result = scene_prepare_lane.repair(root)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertFalse(any(any(part.startswith("git+https://") for part in command) for command in commands))
            self.assertEqual(result["source_install"]["status"], "skipped-exact")
            validate.assert_called_once_with(root.resolve())

    def test_repair_reinstalls_vcs_sources_when_metadata_mismatches(self):
        metadata = _exact_source_metadata()
        metadata["depth-anything-3"]["vcs_info"]["commit_id"] = "0" * 40
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            interpreter = scene_prepare_lane.python_path(root)
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            run_result = {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}
            with patch.object(scene_prepare_lane, "capability", return_value={"supported": True}), \
                 patch.object(scene_prepare_lane, "_interpreter_custody", return_value=(True, {"status": "owned"})), \
                 patch.object(scene_prepare_lane, "_source_metadata", return_value=metadata), \
                 patch.object(scene_prepare_lane, "_run", return_value=run_result) as run, \
                 patch.object(scene_prepare_lane, "validate_runtime", return_value={"status": "ready"}):
                result = scene_prepare_lane.repair(root)
            commands = [call.args[0] for call in run.call_args_list]
            source_commands = [command for command in commands if any(part.startswith("git+https://") for part in command)]
            self.assertEqual(len(source_commands), 1)
            self.assertIn(scene_prepare_lane.SAM3_SOURCE, source_commands[0])
            self.assertIn(scene_prepare_lane.DA3_SOURCE, source_commands[0])
            self.assertEqual(result["source_install"]["status"], "installed")

    def test_source_metadata_requires_both_official_git_origins(self):
        metadata = _exact_source_metadata()
        metadata["sam3"]["url"] = "https://github.com/example/sam3.git"
        with patch.object(scene_prepare_lane, "_source_metadata", return_value=metadata):
            exact, observation = scene_prepare_lane._installed_sources_exact(Path("/extension"))
        self.assertFalse(exact)
        self.assertEqual(observation["status"], "mismatch")
        self.assertFalse(observation["distributions"]["sam3"]["exact"])

    def test_repair_never_targets_primary_or_worldsculpt_venv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            primary = root / "venv"
            worldsculpt = root / "venv-worldsculpt"
            primary.mkdir()
            worldsculpt.mkdir()
            with patch.object(scene_prepare_lane, "_python312", return_value=Path("/usr/bin/python3.12")), \
                 patch.object(scene_prepare_lane, "capability", return_value={"supported": True}), \
                 patch.object(scene_prepare_lane, "_interpreter_custody", return_value=(True, {"status": "owned"})), \
                 patch.object(scene_prepare_lane, "_run") as run, \
                 patch.object(scene_prepare_lane, "validate_runtime", return_value={"status": "ready"}):
                run.return_value = {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}
                result = scene_prepare_lane.repair(root)
            self.assertEqual(result["venv"], str(root / "venv-scene-prep"))
            commands = [call.args[0] for call in run.call_args_list]
            self.assertTrue(all(str(primary) not in command for command in commands))
            self.assertTrue(all(str(worldsculpt) not in command for command in commands))

    def test_setup_exposes_explicit_repair_without_touching_other_lanes(self):
        import setup

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(setup, "_wheelhouse_manifest_observation", return_value={}), \
                 patch.object(scene_prepare_lane, "capability", return_value={"supported": True}), \
                 patch.object(scene_prepare_lane, "repair", return_value={"status": "installed", "venv": str(root / "venv-scene-prep")}) as repair:
                result = setup.run_setup(["--repair-scene-prep", "--workspace-root", str(root)])
            self.assertEqual(result["status"], "prepared")
            self.assertEqual(result["scene_prep_lane"]["status"], "installed")
            repair.assert_called_once_with(root.resolve())

    def test_explicit_scene_repair_honors_skip_install(self):
        import setup

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(setup, "_wheelhouse_manifest_observation", return_value={}), \
                 patch.object(scene_prepare_lane, "capability") as capability, \
                 patch.object(scene_prepare_lane, "repair") as repair:
                result = setup.run_setup([
                    "--repair-scene-prep", "--skip-install", "--workspace-root", str(root),
                ])
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["scene_prep_lane"]["status"], "skipped")
            self.assertFalse(result["installs_started"])
            capability.assert_not_called()
            repair.assert_not_called()

    def test_foreign_interpreter_is_rejected_before_any_pip_command(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / scene_prepare_lane.VENV_NAME
            target.mkdir()
            interpreter = scene_prepare_lane.python_path(root)
            interpreter.parent.mkdir()
            interpreter.symlink_to(Path("/usr/bin/python3"))
            owned, observation = scene_prepare_lane._interpreter_custody(root)
            self.assertFalse(owned)
            self.assertIn("symlink", observation["reason"])

            calls = []
            run_result = {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}
            with patch.object(scene_prepare_lane, "capability", return_value={"supported": True}), \
                 patch.object(scene_prepare_lane, "_python312", return_value=Path("/usr/bin/python3.12")), \
                 patch.object(scene_prepare_lane, "_interpreter_custody", side_effect=[
                     (False, {"status": "unsafe", "reason": "foreign"}),
                     (False, {"status": "unsafe", "reason": "still foreign"}),
                 ]), \
                 patch.object(scene_prepare_lane, "_run", side_effect=lambda command, **_kwargs: (calls.append(command) or run_result)):
                with self.assertRaisesRegex(RuntimeError, "custody validation"):
                    scene_prepare_lane.repair(root)
            self.assertEqual(len(calls), 1)
            self.assertIn("venv", calls[0])
            self.assertFalse(any("pip" in command for command in calls))

    def test_general_setup_provisions_scene_lane_unless_explicitly_skipped(self):
        import setup

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            common = ["--prepare", "--workspace-root", str(root)]
            with patch.object(setup, "_create_prepare_paths", return_value=([], [])), \
                 patch.object(setup, "_prepare_wheelhouse_for_setup", return_value={"wheelhouse_path": str(root / "wheels")}), \
                 patch.object(setup, "_prepared_wheelhouse_path", return_value=root / "wheels"), \
                 patch.object(setup, "_install_prepare_dependencies", return_value={"status": "installed"}), \
                 patch.object(setup, "check_setup_readiness", return_value={}), \
                 patch.object(setup, "_wheelhouse_manifest_observation", return_value={}), \
                 patch.object(scene_prepare_lane, "capability", return_value={"supported": True}), \
                 patch.object(scene_prepare_lane, "repair", return_value={"status": "installed"}) as repair:
                result = setup.run_setup(common)
                skipped = setup.run_setup([*common, "--skip-scene-prep"])
            self.assertEqual(result["scene_prep_lane"]["status"], "installed")
            self.assertNotIn("scene_prep_lane", skipped)
            repair.assert_called_once_with(root.resolve())

    def test_general_setup_keeps_primary_success_when_scene_lane_is_unsupported(self):
        import setup

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            support = {"supported": False, "status": "unsupported-platform", "reason": "not Linux ARM64"}
            with patch.object(setup, "_create_prepare_paths", return_value=([], [])), \
                 patch.object(setup, "_prepare_wheelhouse_for_setup", return_value={"wheelhouse_path": str(root / "wheels")}), \
                 patch.object(setup, "_prepared_wheelhouse_path", return_value=root / "wheels"), \
                 patch.object(setup, "_install_prepare_dependencies", return_value={"status": "installed"}), \
                 patch.object(setup, "check_setup_readiness", return_value={}), \
                 patch.object(setup, "_wheelhouse_manifest_observation", return_value={}), \
                 patch.object(scene_prepare_lane, "capability", return_value=support), \
                 patch.object(scene_prepare_lane, "repair") as repair:
                result = setup.run_setup(["--prepare", "--workspace-root", str(root)])
            self.assertEqual(result["status"], "prepared")
            self.assertEqual(result["scene_prep_lane"]["status"], "unsupported-platform")
            repair.assert_not_called()

    def test_general_setup_keeps_primary_success_when_optional_scene_repair_fails(self):
        import setup

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(setup, "_create_prepare_paths", return_value=([], [])), \
                 patch.object(setup, "_prepare_wheelhouse_for_setup", return_value={"wheelhouse_path": str(root / "wheels")}), \
                 patch.object(setup, "_prepared_wheelhouse_path", return_value=root / "wheels"), \
                 patch.object(setup, "_install_prepare_dependencies", return_value={"status": "installed"}), \
                 patch.object(setup, "check_setup_readiness", return_value={}), \
                 patch.object(setup, "_wheelhouse_manifest_observation", return_value={}), \
                 patch.object(scene_prepare_lane, "capability", return_value={"supported": True}), \
                 patch.object(scene_prepare_lane, "repair", side_effect=RuntimeError("lane failed")):
                result = setup.run_setup(["--prepare", "--workspace-root", str(root)])
            self.assertEqual(result["status"], "prepared")
            self.assertEqual(result["scene_prep_lane"]["status"], "unprepared")
            self.assertIn("lane failed", result["scene_prep_lane"]["reason"])

    def test_explicit_scene_repair_fails_cleanly_on_unsupported_platform(self):
        import setup

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            support = {"supported": False, "status": "unsupported-platform", "reason": "not Linux ARM64"}
            with patch.object(setup, "_wheelhouse_manifest_observation", return_value={}), \
                 patch.object(scene_prepare_lane, "capability", return_value=support), \
                 patch.object(scene_prepare_lane, "repair") as repair:
                result = setup.run_setup(["--repair-scene-prep", "--workspace-root", str(root)])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure_code"], "scene_prep_unsupported_platform")
            self.assertFalse(result["installs_started"])
            repair.assert_not_called()


if __name__ == "__main__":
    unittest.main()
