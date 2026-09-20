import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pixal3d_extension.scene_prepare import (
    DA3_FILES,
    SAM3_FILES,
    _run_worker,
    offline_environment,
    run_scene_from_estimates,
    validate_scene_prepare_weights,
    worker_command,
)
from pixal3d_extension.da3_official_adapter import load_depth_anything3


class ScenePrepareRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sam = self.root / "sam3"
        self.da3 = self.root / "da3"
        self.sam.mkdir()
        self.da3.mkdir()
        for relative in SAM3_FILES:
            path = self.sam / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"{}" if path.suffix == ".json" else b"weights")
        for relative in DA3_FILES:
            path = self.da3 / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"{}" if path.suffix == ".json" else b"weights")

    def tearDown(self):
        self.temp.cleanup()

    def test_requires_exact_local_ui_managed_files(self):
        validate_scene_prepare_weights(self.sam, self.da3)
        (self.sam / SAM3_FILES[-1]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Models UI"):
            validate_scene_prepare_weights(self.sam, self.da3)

    def test_worker_uses_only_isolated_lane_and_offline_hf_environment(self):
        command = worker_command(self.root, self.root / "job.json")
        self.assertEqual(command[0], str(self.root / "venv-scene-prep/bin/python"))
        self.assertEqual(command[1:3], ["-m", "pixal3d_extension.scene_prepare_worker"])
        env = offline_environment(self.root / "cache", self.root)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(env["HF_HOME"], str(self.root / "cache"))
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(self.root.resolve()))
        self.assertNotIn("HF_TOKEN", env)

    def test_worker_module_launches_from_explicit_extension_root(self):
        interpreter = self.root / "venv-scene-prep/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        package = self.root / "pixal3d_extension"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        expected = self.root / "scene.json"
        (package / "scene_prepare_worker.py").write_text(
            "import json\nprint(json.dumps({'type':'done','scene_manifest_path':%r}), flush=True)\n" % str(expected),
            encoding="utf-8",
        )
        result = _run_worker(
            worker_command(self.root, self.root / "job.json"),
            cwd=self.root,
            env=offline_environment(self.root / "cache", self.root),
        )
        self.assertEqual(result, expected)

    def test_cancellation_is_responsive_when_worker_writes_partial_line(self):
        cancelled = threading.Event()
        timer = threading.Timer(0.15, cancelled.set)
        timer.start()
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                _run_worker(
                    [sys.executable, "-c", "import sys,time;sys.stdout.write('{');sys.stdout.flush();time.sleep(10)"],
                    cwd=self.root,
                    env=os.environ.copy(),
                    cancel_event=cancelled,
                )
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 2.0)

    def test_cancellation_reaps_worker_descendants(self):
        marker = self.root / "descendant-survived"
        child = (
            "import pathlib,time;time.sleep(1.0);"
            f"pathlib.Path({str(marker)!r}).write_text('survived')"
        )
        parent = (
            "import subprocess,sys,time;"
            f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
            "time.sleep(10)"
        )
        cancelled = threading.Event()
        timer = threading.Timer(0.15, cancelled.set)
        timer.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                _run_worker(
                    [sys.executable, "-c", parent], cwd=self.root,
                    env=os.environ.copy(), cancel_event=cancelled,
                )
        finally:
            timer.cancel()
        time.sleep(1.2)
        self.assertFalse(marker.exists())

    def test_unsupported_platform_is_scoped_to_estimate_node_readiness(self):
        from generator import Pixal3DGenerator, SCENE_ESTIMATE_NODE, SCENE_NORMALIZE_NODE

        estimate = Pixal3DGenerator()
        estimate.MODEL_NODE_ID = SCENE_ESTIMATE_NODE
        normalize = Pixal3DGenerator()
        normalize.MODEL_NODE_ID = SCENE_NORMALIZE_NODE
        support = {"supported": False, "reason": "unsupported test platform"}
        with patch("pixal3d_extension.scene_prepare_lane.capability", return_value=support):
            status = estimate.readiness_status()
        self.assertFalse(status["ok"])
        self.assertEqual(status["machine_code"], "scene_prep_unsupported_platform")
        self.assertTrue(normalize.readiness_status()["ok"])

    def test_da3_adapter_keeps_official_api_and_fails_closed_on_unused_boundaries(self):
        package = types.ModuleType("depth_anything_3")
        package.__path__ = []
        api = types.ModuleType("depth_anything_3.api")
        official_class = type("DepthAnything3", (), {})
        api.DepthAnything3 = official_class
        modules = {"depth_anything_3": package, "depth_anything_3.api": api}
        with patch.dict(sys.modules, modules, clear=False):
            sys.modules.pop("depth_anything_3.utils.export", None)
            sys.modules.pop("depth_anything_3.utils.pose_align", None)
            self.assertIs(load_depth_anything3(), official_class)
            with self.assertRaisesRegex(RuntimeError, "does not enable"):
                sys.modules["depth_anything_3.utils.export"].export()

    def test_estimator_rejects_unsafe_output_parents_before_mkdir(self):
        workspace = self.root / "workspace"
        capture = workspace / "Captures" / "capture"
        capture.mkdir(parents=True)
        frame = capture / "frame.png"
        frame.write_bytes(b"frame")
        manifest = capture / "capture-manifest.json"
        manifest.write_text(json.dumps({
            "schema": "modly.capture-manifest.v1",
            "kind": "frames",
            "captureRoot": ".",
            "frames": [{"index": 0, "path": "frame.png", "byteSize": 5, "width": 1, "height": 1}],
            "provenance": {"source": "test", "ordering": "manifest-index"},
        }), encoding="utf-8")
        outside = self.root / "outside"
        outside.mkdir()
        link = workspace / "linked-output"
        link.symlink_to(outside, target_is_directory=True)
        cases = (
            workspace,
            Path("/"),
            outside / "new-output",
            link / "new-output",
            workspace / "Workflows" / ".." / ".." / "escape",
        )
        before = {path.relative_to(workspace) for path in workspace.rglob("*")}
        with patch("pixal3d_extension.scene_prepare.validate_scene_prepare_weights"), \
             patch("pixal3d_extension.scene_prepare.validate_runtime"):
            for output in cases:
                with self.subTest(output=output), self.assertRaises(ValueError):
                    run_scene_from_estimates(
                        capture_input=manifest,
                        workspace_dir=workspace,
                        output_dir=output,
                        sam_root=self.sam,
                        da3_root=self.da3,
                        params={},
                    )
        after = {path.relative_to(workspace) for path in workspace.rglob("*")}
        self.assertEqual(before, after)
        self.assertFalse((outside / "new-output").exists())


if __name__ == "__main__":
    unittest.main()
