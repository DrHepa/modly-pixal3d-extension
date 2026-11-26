import subprocess
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import test_generator_naf_bootstrap as fixtures
from pixal3d_extension import scene_prepare_lane


class MvCalibrationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.GeneratorNafBootstrapTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.generator = self.fixture._generator("generate-mv")
        for name in ("config.json", "model.safetensors"):
            (self.fixture.da3 / name).write_bytes(b"weight")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict("sys.modules", {"torch": None}))
        self.core = self.stack.enter_context(patch("pixal3d_extension.multiview.validate_mv_pipeline_config"))
        self.missing = self.stack.enter_context(patch("pixal3d_extension.multiview.missing_mv_assets", return_value=[]))
        self.mv_runtime = self.stack.enter_context(patch("pixal3d_extension.multiview.mv_runtime_available", return_value=True))
        self.da3_capability = scene_prepare_lane.da3_capability
        self.support = self.stack.enter_context(patch.object(scene_prepare_lane, "da3_capability", return_value={"supported": True}))
        self.runtime = self.stack.enter_context(patch.object(scene_prepare_lane, "validate_da3_runtime"))
        self.naf = self.stack.enter_context(patch.object(self.generator, "_ensure_naf_checkpoint", return_value=self.fixture.naf_path))

    def _status(self, mode="auto"):
        return self.generator.readiness_status({"view_layout": mode})

    def _assert_public_keys(self, status, mode="auto"):
        self.assertEqual(status["camera_mode"], mode)
        self.assertIn("missing", status)
        self.assertEqual(status["auto_calibration_requires"],
                         ["DA3 Base weights", "validated Scene Prep Python 3.12 runtime"])

    def test_load_unselected_and_declared_roles_do_not_touch_da3(self):
        self.generator.shared_model_dirs.pop("da3-base")
        with patch("pixal3d_extension.scene_prepare.validate_da3_weights") as weights:
            self.assertIs(self.generator.load(), self.generator)
            for params, mode in ((None, "not-selected"), ({"view_layout": "declared_roles"}, "declared_roles")):
                status = self.generator.readiness_status(params)
                self.assertTrue(status["ok"])
                self._assert_public_keys(status, mode)
                self.generator._preflight_non_naf_shared_assets(params)
            weights.assert_not_called()
        self.support.assert_not_called()
        self.runtime.assert_not_called()

    def test_da3_group_missing_is_an_asset_failure_before_naf(self):
        self.generator.shared_model_dirs.pop("da3-base")
        status = self._status()
        self.assertEqual(status["machine_code"], "mv_assets_missing")
        self.assertIn("da3-base", status["reason"])
        self.assertTrue(status["missing"])
        self._assert_public_keys(status)
        with self.assertRaisesRegex(RuntimeError, "^mv_assets_missing:.*da3-base") as raised:
            self.generator._prepare_generation_assets(params={"view_layout": "auto"})
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.runtime.assert_not_called()
        self.naf.assert_not_called()

    def test_missing_or_nonfile_da3_weights_never_probe_runtime(self):
        config = self.fixture.da3 / "config.json"
        config.unlink()
        for nonfile in (False, True):
            with self.subTest(nonfile=nonfile):
                if nonfile:
                    config.mkdir()
                status = self._status()
                self.assertEqual(status["machine_code"], "mv_assets_missing")
                self.assertIn("config.json", status["reason"])
                with self.assertRaisesRegex(RuntimeError, "^mv_assets_missing:"):
                    self.generator._prepare_generation_assets(params={"view_layout": "auto"})
        self.runtime.assert_not_called()
        self.naf.assert_not_called()

    def test_runtime_failures_are_not_weights_and_preserve_original_cause(self):
        failures = (RuntimeError("SM120 policy diagnostic"), OSError("private interpreter unavailable"),
                    ValueError("invalid probe JSON"), subprocess.SubprocessError("probe failed"),
                    subprocess.CalledProcessError(1, ["private-python", "probe"], stderr="import failed"),
                    subprocess.TimeoutExpired(["private-python", "probe"], 5))
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.runtime.side_effect = failure
                status = self._status()
                self.assertFalse(status["ok"])
                self.assertEqual(status["machine_code"], "mv_calibration_runtime_unavailable")
                self.assertEqual(status["missing"], [])
                self._assert_public_keys(status)
                self.assertIn(type(failure).__name__, status["reason"])
                self.assertIn(str(failure), status["reason"])
                for misleading in ("Models", "download", "mv_assets_missing"):
                    self.assertNotIn(misleading, status["reason"])
                with self.assertRaisesRegex(RuntimeError, "^mv_calibration_runtime_unavailable:") as raised:
                    self.generator._prepare_generation_assets(params={"view_layout": "auto"})
                self.assertIs(raised.exception.__cause__, failure)
                self.assertEqual(str(raised.exception), f"{status['machine_code']}: {status['reason']}")
        self.naf.assert_not_called()

    def test_unsupported_platform_keeps_its_code_without_runtime_probe(self):
        self.support.return_value = {"supported": False, "reason": "unsupported test platform"}
        status = self._status()
        self.assertEqual(status["machine_code"], "mv_calibration_unsupported_platform")
        self._assert_public_keys(status)
        self.assertEqual(status["missing"], [])
        with self.assertRaisesRegex(RuntimeError, "^mv_calibration_unsupported_platform:"):
            self.generator._prepare_generation_assets(params={"view_layout": "auto"})
        self.runtime.assert_not_called()
        self.naf.assert_not_called()

    def test_core_asset_failures_keep_priority_and_existing_code(self):
        for missing in ("pixal3d-base/ckpts/decoder.safetensors", "pixal3d-mv/pipeline_mv.json"):
            with self.subTest(missing=missing):
                self.missing.return_value = [missing]
                status = self._status()
                self.assertEqual(status["machine_code"], "mv_assets_missing")
                self.assertEqual(status["missing"], [missing])
                failure = ValueError(missing)
                self.core.side_effect = failure
                with self.assertRaisesRegex(RuntimeError, "^mv_assets_missing:") as raised:
                    self.generator._prepare_generation_assets(params={"view_layout": "auto"})
                self.assertIs(raised.exception.__cause__, failure)
        self.support.assert_not_called()
        self.runtime.assert_not_called()
        self.naf.assert_not_called()

    def test_single_view_asset_code_is_unchanged(self):
        generator = self.fixture._generator("generate")
        with self.assertRaisesRegex(RuntimeError, "^missing_assets:"):
            generator._preflight_non_naf_shared_assets()
        self.runtime.assert_not_called()

    def test_primary_mv_runtime_missing_is_not_a_calibration_failure(self):
        self.mv_runtime.return_value = False
        status = self._status()
        self.assertEqual(status["machine_code"], "mv_runtime_missing")
        self.assertEqual(status["missing"], [])
        self._assert_public_keys(status)
        self.support.assert_not_called()
        self.runtime.assert_not_called()

    def test_generate_classifies_absent_da3_before_naf_or_inference(self):
        self.generator.shared_model_dirs.pop("da3-base")
        with patch("pixal3d_extension.multiview_images.validate_ordered_images"), \
             patch("pixal3d_extension.multiview_images.run_multiview_from_images") as run:
            with self.assertRaisesRegex(RuntimeError, "^mv_assets_missing:.*da3-base"):
                self.generator.generate(b"png", {"extra_image_paths": ["secondary.png"], "view_layout": "auto"})
        self.runtime.assert_not_called()
        self.naf.assert_not_called()
        run.assert_not_called()

    def test_inference_error_is_not_reclassified_as_preflight_failure(self):
        failure = ValueError("inference stage failed")
        with patch("pixal3d_extension.multiview_images.validate_ordered_images"), \
             patch("pixal3d_extension.multiview_images.run_multiview_from_images", side_effect=failure) as run:
            with self.assertRaises(ValueError) as raised:
                self.generator.generate(b"png", {"extra_image_paths": ["secondary.png"], "view_layout": "auto"})
        self.assertIs(raised.exception, failure)
        self.runtime.assert_called_once()
        self.naf.assert_called_once()
        self.assertEqual(run.call_args.kwargs["da3_root"], self.fixture.da3)

    def test_ready_modes_keep_public_shape_and_only_auto_validates_runtime(self):
        statuses = [self.generator.readiness_status(), self._status("declared_roles"), self._status()]
        for status, mode in zip(statuses, ("not-selected", "declared_roles", "auto")):
            self.assertTrue(status["ok"])
            self.assertEqual(status["machine_code"], "ready")
            self.assertEqual(status["missing"], [])
            self._assert_public_keys(status, mode)
            self.assertEqual(set(status), set(statuses[0]))
        self.runtime.assert_called_once()

    def test_invalid_camera_mode_never_checks_calibration(self):
        self.assertEqual(self._status("invalid")["machine_code"], "mv_invalid_camera_mode")
        self.generator._preflight_non_naf_shared_assets({"view_layout": "invalid"})
        self.runtime.assert_not_called()
        self.naf.assert_not_called()

    def test_windows_metadata_distinguishes_interpreters_from_runtime_policy(self):
        with patch.object(scene_prepare_lane.sys, "platform", "win32"), \
             patch.object(scene_prepare_lane.platform, "machine", return_value="AMD64"):
            support = self.da3_capability()
        self.assertTrue(support["supported"])
        self.assertEqual(support["primary_python"], "3.11")
        self.assertEqual(support["calibration_python"], "3.12")
        self.assertIn("cu128; SM120", support["runtime_policy"])
        self.assertIn("not an intrinsic DA3 requirement", support["reason"])


if __name__ == "__main__":
    unittest.main()
