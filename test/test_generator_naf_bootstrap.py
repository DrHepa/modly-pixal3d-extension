import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import generator as generator_module
from generator import Pixal3DGenerator


class GeneratorNafBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.base = self.root / "weights" / "base"
        self.mv = self.root / "weights" / "mv"
        self.adapters = self.root / "weights" / "adapters"
        for path in (self.base, self.mv, self.adapters):
            path.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def _generator(self, node_id: str) -> Pixal3DGenerator:
        with patch("pixal3d_extension.multiview.configure_mv_hf_cache"):
            generator = Pixal3DGenerator(
                self.root / "models" / "pixal3d" / node_id,
                self.workspace,
            )
        generator.MODEL_NODE_ID = node_id
        generator.shared_model_dirs = {
            "pixal3d-base": self.base,
            "pixal3d-mv": self.mv,
            "worldsculpt-adapters": self.adapters,
        }
        return generator

    @property
    def naf_path(self) -> Path:
        return self.root / "models/pixal3d/auxiliary/naf/naf_release.pth"

    def _successful_bootstrap(self, workspace_root, **_kwargs):
        self.assertEqual(Path(workspace_root), self.root)
        self.naf_path.parent.mkdir(parents=True, exist_ok=True)
        self.naf_path.write_bytes(b"valid-naf")
        return {"status": "ready", "code": "auxiliary_assets_bootstrapped"}

    def test_mv_and_worldsculpt_load_bootstrap_missing_naf_before_readiness(self):
        for node_id in ("generate-mv", "worldsculpt"):
            with self.subTest(node_id=node_id):
                self.naf_path.unlink(missing_ok=True)
                generator = self._generator(node_id)
                events = []

                def readiness():
                    events.append("readiness")
                    self.assertTrue(self.naf_path.is_file())
                    return {"ok": True, "machine_code": "ready", "reason": "ready"}

                with patch.object(generator, "_preflight_non_naf_shared_assets") as preflight, \
                     patch.object(generator_module, "bootstrap_auxiliary_assets", side_effect=self._successful_bootstrap) as bootstrap, \
                     patch.object(generator_module, "verify_naf_checkpoint", side_effect=lambda path: events.append(("verify", Path(path)))), \
                     patch.object(generator, "readiness_status", side_effect=readiness):
                    self.assertIs(generator.load(), generator)

                bootstrap.assert_called_once_with(self.root)
                preflight.assert_called_once_with()
                self.assertEqual(events, [("verify", self.naf_path), "readiness"])

    def test_valid_naf_never_downloads_for_either_load_path(self):
        self.naf_path.parent.mkdir(parents=True)
        self.naf_path.write_bytes(b"valid-naf")
        for node_id in ("generate-mv", "worldsculpt"):
            with self.subTest(node_id=node_id):
                generator = self._generator(node_id)
                with patch.object(generator, "_preflight_non_naf_shared_assets"), \
                     patch.object(generator_module, "bootstrap_auxiliary_assets") as bootstrap, \
                     patch.object(generator_module, "verify_naf_checkpoint"), \
                     patch.object(generator, "readiness_status", return_value={"ok": True, "machine_code": "ready", "reason": "ready"}):
                    generator.load()
                bootstrap.assert_not_called()

    def test_corrupt_naf_fails_closed_without_automatic_replacement(self):
        self.naf_path.parent.mkdir(parents=True)
        self.naf_path.write_bytes(b"corrupt")
        generator = self._generator("generate-mv")
        expected_command = (
            f"setup.py --bootstrap-auxiliary-assets --force-auxiliary-assets "
            f"--workspace-root {Path(generator_module.__file__).resolve().parent} --json"
        )
        with patch.object(generator, "_preflight_non_naf_shared_assets"), \
             patch.object(generator_module, "verify_naf_checkpoint", side_effect=RuntimeError("SHA256 mismatch")), \
             patch.object(generator_module, "bootstrap_auxiliary_assets") as bootstrap, \
             patch.object(generator, "readiness_status") as readiness, \
             self.assertRaisesRegex(RuntimeError, "naf_bootstrap_failed") as raised:
            generator.load()
        bootstrap.assert_not_called()
        readiness.assert_not_called()
        self.assertIn(expected_command, str(raised.exception))

    def test_network_bootstrap_failure_is_actionable_and_prevents_readiness(self):
        generator = self._generator("worldsculpt")
        with patch.object(generator, "_preflight_non_naf_shared_assets"), \
             patch.object(generator_module, "bootstrap_auxiliary_assets", return_value={
            "status": "failed", "code": "auxiliary_bootstrap_failed", "error": "network unreachable"
        }), patch.object(generator_module, "verify_naf_checkpoint") as verify, \
             patch.object(generator, "readiness_status") as readiness, \
             self.assertRaisesRegex(RuntimeError, "naf_bootstrap_failed.*bootstrap-auxiliary-assets") as raised:
            generator.load()
        verify.assert_not_called()
        readiness.assert_not_called()
        self.assertIn("network unreachable", str(raised.exception))

    def test_mv_direct_generate_defensively_bootstraps_missing_naf(self):
        generator = self._generator("generate-mv")
        capture = SimpleNamespace(kind="capture", path=self.workspace / "capture-manifest.json")
        capture.path.write_text("{}")
        output = self.workspace / "mv.glb"
        with patch.object(generator, "_preflight_non_naf_shared_assets"), \
             patch("pixal3d_extension.multiview_capture.validate_mv_capture") as input_preflight, \
             patch.object(generator_module, "bootstrap_auxiliary_assets", side_effect=self._successful_bootstrap) as bootstrap, \
             patch.object(generator_module, "verify_naf_checkpoint"), \
             patch("pixal3d_extension.multiview.run_multiview", return_value=output) as run:
            self.assertEqual(generator.generate(capture, {"num_views": "4"}), output)
        input_preflight.assert_called_once_with(capture.path, self.workspace, 4)
        bootstrap.assert_called_once_with(self.root)
        self.assertEqual(run.call_args.kwargs["naf_path"], self.naf_path)

    def test_worldsculpt_direct_generate_defensively_bootstraps_missing_naf(self):
        generator = self._generator("worldsculpt")
        scene_manifest = self.workspace / "scene-manifest.json"
        scene_manifest.write_text("{}")
        scene = self.workspace / "scene"
        scene.mkdir()
        output = self.workspace / "world.glb"
        with patch.object(generator, "_preflight_non_naf_shared_assets"), \
             patch.object(generator_module, "bootstrap_auxiliary_assets", side_effect=self._successful_bootstrap) as bootstrap, \
             patch.object(generator_module, "verify_naf_checkpoint"), \
             patch("pixal3d_extension.worldsculpt.resolve_scene_manifest", return_value=scene), \
             patch("pixal3d_extension.worldsculpt.run_worldsculpt", return_value=output) as run:
            self.assertEqual(generator.generate(None, {"scene_manifest_path": str(scene_manifest)}), output)
        bootstrap.assert_called_once_with(self.root)
        self.assertEqual(run.call_args.kwargs["naf_path"], self.naf_path)

    def test_direct_generate_surfaces_bootstrap_network_failure_before_runner(self):
        generator = self._generator("generate-mv")
        capture = SimpleNamespace(kind="capture", path=self.workspace / "capture-manifest.json")
        capture.path.write_text("{}")
        with patch.object(generator, "_preflight_non_naf_shared_assets"), \
             patch("pixal3d_extension.multiview_capture.validate_mv_capture"), \
             patch.object(generator_module, "bootstrap_auxiliary_assets", return_value={
            "status": "failed", "code": "auxiliary_bootstrap_failed", "error": "connection reset"
        }), patch("pixal3d_extension.multiview.run_multiview") as run, \
             self.assertRaisesRegex(RuntimeError, "naf_bootstrap_failed"):
            generator.generate(capture)
        run.assert_not_called()

    def test_load_does_not_bootstrap_when_non_naf_shared_assets_are_missing(self):
        for node_id, error in (
            ("generate-mv", "mv_assets_missing"),
            ("worldsculpt", "worldsculpt_assets_missing"),
        ):
            with self.subTest(node_id=node_id):
                generator = self._generator(node_id)
                with patch.object(generator_module, "bootstrap_auxiliary_assets") as bootstrap, \
                     self.assertRaisesRegex(RuntimeError, error):
                    generator.load()
                bootstrap.assert_not_called()
                self.assertFalse(self.naf_path.exists())

    def test_mv_invalid_pipeline_config_never_bootstraps_or_mutates_naf(self):
        from pixal3d_extension.multiview import (
            AUXILIARY_FILES,
            BASE_DECODER_FILES,
            MV_WEIGHT_FILES,
            prepare_mv_pipeline_config,
        )

        for relative in MV_WEIGHT_FILES:
            path = self.mv / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-json" if relative == "pipeline_mv.json" else b"weight")
        base_files = tuple(
            f"ckpts/{model}.{extension}"
            for model in BASE_DECODER_FILES
            for extension in ("json", "safetensors")
        )
        for relative in (*base_files, *AUXILIARY_FILES):
            path = self.base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"weight")

        generator = self._generator("generate-mv")
        with patch.object(generator_module, "bootstrap_auxiliary_assets") as bootstrap, \
             self.assertRaisesRegex(RuntimeError, "mv_assets_missing.*pipeline_mv.json"):
            generator.load()

        bootstrap.assert_not_called()
        self.assertFalse(self.naf_path.exists())

        private_config = self.root / "private" / "pipeline_mv.local.json"
        private_config.parent.mkdir()
        private_config.write_bytes(b"sentinel")
        with self.assertRaisesRegex(ValueError, "pipeline_mv.json"):
            prepare_mv_pipeline_config(self.mv, self.base, private_config)
        self.assertEqual(private_config.read_bytes(), b"sentinel")

    def test_worldsculpt_valid_adapters_with_missing_base_still_never_bootstrap(self):
        from pixal3d_extension.worldsculpt_contract import ADAPTER_DIRS, STEP

        for directory in ADAPTER_DIRS:
            stage = self.adapters / directory
            (stage / "ckpts").mkdir(parents=True)
            (stage / "config.json").write_text('{"models": {}, "trainer": {}}')
            (stage / "ckpts" / f"denoiser_step{STEP:07d}.pt").write_bytes(b"x")
            (stage / "ckpts" / f"mv_aggregator_step{STEP:07d}.pt").write_bytes(b"x")
        generator = self._generator("worldsculpt")
        with patch.object(generator_module, "bootstrap_auxiliary_assets") as bootstrap, \
             self.assertRaisesRegex(RuntimeError, "worldsculpt_assets_missing.*pipeline.json"):
            generator.load()
        bootstrap.assert_not_called()
        self.assertFalse(self.naf_path.exists())

    def test_direct_generation_pre_cancel_never_bootstraps_or_runs(self):
        cancelled = threading.Event()
        cancelled.set()
        capture = SimpleNamespace(kind="capture", path=self.workspace / "capture-manifest.json")
        capture.path.write_text("{}")
        scene_manifest = self.workspace / "scene-manifest.json"
        scene_manifest.write_text("{}")
        for node_id, input_value, params, runner_name in (
            ("generate-mv", capture, None, "pixal3d_extension.multiview.run_multiview"),
            ("worldsculpt", None, {"scene_manifest_path": str(scene_manifest)}, "pixal3d_extension.worldsculpt.run_worldsculpt"),
        ):
            with self.subTest(node_id=node_id):
                generator = self._generator(node_id)
                with patch.object(generator, "_preflight_non_naf_shared_assets") as preflight, \
                     patch.object(generator_module, "bootstrap_auxiliary_assets") as bootstrap, \
                     patch(runner_name) as runner, \
                     self.assertRaisesRegex(RuntimeError, "cancelled"):
                    generator.generate(input_value, params, cancel_evt=cancelled)
                preflight.assert_not_called()
                bootstrap.assert_not_called()
                runner.assert_not_called()
                self.assertFalse(self.naf_path.exists())

    def test_worldsculpt_missing_or_invalid_scene_never_bootstraps_or_creates_naf(self):
        for label, scene_manifest in (
            ("missing", self.workspace / "missing-scene-manifest.json"),
            ("invalid", self.workspace / "invalid-scene-manifest.json"),
        ):
            with self.subTest(label=label):
                self.naf_path.unlink(missing_ok=True)
                if label == "invalid":
                    scene_manifest.write_text("{}")
                generator = self._generator("worldsculpt")
                with patch.object(generator, "_preflight_non_naf_shared_assets"), \
                     patch.object(generator_module, "bootstrap_auxiliary_assets", side_effect=self._successful_bootstrap) as bootstrap, \
                     patch.object(generator_module, "verify_naf_checkpoint"), \
                     patch("pixal3d_extension.worldsculpt.run_worldsculpt") as run, \
                     self.assertRaises((FileNotFoundError, ValueError)):
                    generator.generate(None, {"scene_manifest_path": str(scene_manifest)})
                bootstrap.assert_not_called()
                run.assert_not_called()
                self.assertFalse(self.naf_path.exists())

    def test_mv_missing_or_invalid_capture_never_bootstraps_or_creates_naf(self):
        for label, capture_path in (
            ("missing", self.workspace / "missing" / "capture-manifest.json"),
            ("invalid", self.workspace / "invalid" / "capture-manifest.json"),
        ):
            with self.subTest(label=label):
                self.naf_path.unlink(missing_ok=True)
                if label == "invalid":
                    capture_path.parent.mkdir()
                    capture_path.write_text("{}")
                generator = self._generator("generate-mv")
                capture = SimpleNamespace(kind="capture", path=capture_path)
                with patch.object(generator, "_preflight_non_naf_shared_assets"), \
                     patch.object(generator_module, "bootstrap_auxiliary_assets", side_effect=self._successful_bootstrap) as bootstrap, \
                     patch.object(generator_module, "verify_naf_checkpoint"), \
                     patch("pixal3d_extension.multiview.run_multiview", wraps=__import__("pixal3d_extension.multiview", fromlist=["run_multiview"]).run_multiview) as run, \
                     self.assertRaises((FileNotFoundError, ValueError)):
                    generator.generate(capture, {"num_views": "4"})
                bootstrap.assert_not_called()
                run.assert_not_called()
                self.assertFalse(self.naf_path.exists())

    def test_direct_generation_rechecks_cancellation_immediately_after_bootstrap(self):
        capture = SimpleNamespace(kind="capture", path=self.workspace / "capture-manifest.json")
        capture.path.write_text("{}")
        scene_manifest = self.workspace / "scene-manifest.json"
        scene_manifest.write_text("{}")
        for node_id, input_value, params, runner_name in (
            ("generate-mv", capture, None, "pixal3d_extension.multiview.run_multiview"),
            ("worldsculpt", None, {"scene_manifest_path": str(scene_manifest)}, "pixal3d_extension.worldsculpt.run_worldsculpt"),
        ):
            with self.subTest(node_id=node_id):
                self.naf_path.unlink(missing_ok=True)
                cancelled = threading.Event()
                generator = self._generator(node_id)

                def bootstrap_then_cancel(root):
                    result = self._successful_bootstrap(root)
                    cancelled.set()
                    return result

                with patch.object(generator, "_preflight_non_naf_shared_assets"), \
                     patch("pixal3d_extension.multiview_capture.validate_mv_capture"), \
                     patch("pixal3d_extension.worldsculpt.resolve_scene_manifest", return_value=self.workspace), \
                     patch.object(generator_module, "bootstrap_auxiliary_assets", side_effect=bootstrap_then_cancel) as bootstrap, \
                     patch.object(generator_module, "verify_naf_checkpoint"), \
                     patch(runner_name) as runner, \
                     self.assertRaisesRegex(RuntimeError, "cancelled"):
                    generator.generate(input_value, params, cancel_evt=cancelled)
                bootstrap.assert_called_once_with(self.root)
                runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
