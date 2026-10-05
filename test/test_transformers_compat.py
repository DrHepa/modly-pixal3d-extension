import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from generator import Pixal3DGenerator
import setup
from pixal3d_extension import readiness


class TransformersCompatTests(unittest.TestCase):
    def test_single_view_generator_rejects_drift_before_load_or_generation(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            (base / "pipeline.json").write_text("{}")
            g = Pixal3DGenerator(model_dir=Path(tmp) / "models/pixal3d/generate")
            g.MODEL_NODE_ID = "generate"
            g.shared_model_dirs = {"pixal3d-base": base}
            with patch.object(readiness, "_private_transformers_version", return_value="5.9.0"), patch(
                "pixal3d_extension.pipeline_patch.patch_pipeline"
            ) as pipeline_patch, patch("pixal3d_extension.runtime.run_job") as run_job:
                status = g.readiness_status()
                self.assertFalse(status["ok"])
                self.assertEqual(status["machine_code"], "transformers_version_mismatch")
                with self.assertRaisesRegex(RuntimeError, "transformers_version_mismatch"):
                    g.load()
                with self.assertRaisesRegex(RuntimeError, "transformers_version_mismatch"):
                    g.generate({"input_image": "unused", "output_dir": tmp})
                pipeline_patch.assert_not_called()
                run_job.assert_not_called()

    def test_readiness_rejects_major_version_drift(self):
        with patch.object(readiness, "check_asset_sentinels", return_value={"status": "ready"}), patch.object(
            readiness, "resolve_auxiliary_sources", return_value={"status": "ready", "unlocalized_runtime_dependencies": []}
        ), patch.object(readiness, "validate_pipeline_patch", return_value={"status": "ready"}), patch.object(
            readiness, "_private_transformers_version", return_value="5.9.0"
        ):
            result = readiness.check_readiness("/tmp/unused", runtime_validated=True)
        self.assertEqual(result["code"], "transformers_version_mismatch")
        self.assertFalse(result["generation_allowed"])

    def test_setup_cuda_probe_checks_exact_transformers_version(self):
        for payload in ('{"ok": true, "transformers_version": "5.9.0"}', '{"ok": true}'):
            with self.subTest(payload=payload), patch.object(
                setup, "_run_setup_command", return_value={"ok": True, "stdout_tail": payload, "args": []}
            ) as run:
                result = setup._runtime_cuda_check(Path("/tmp/python"), Path("/tmp"), Path("/tmp/wheels"))
                self.assertFalse(result["ok"])
                self.assertIn("4.57.3", run.call_args.args[0][-1])


if __name__ == "__main__":
    unittest.main()
