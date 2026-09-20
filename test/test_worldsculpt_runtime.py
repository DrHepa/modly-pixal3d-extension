import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from pixal3d_extension import naf_checkpoint, worldsculpt as ws


class WorldSculptRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_manifest_rejects_escape(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        manifest = workspace / "scene.json"
        manifest.write_text(json.dumps({"schema": "modly.scene-manifest.v1", "sceneRoot": "../outside"}))
        with self.assertRaisesRegex(ValueError, "sceneRoot"):
            ws.resolve_scene_manifest(manifest, workspace)

    def test_manifest_accepts_workspace_scene(self):
        workspace = self.root / "workspace"
        scene = workspace / "scene"
        scene.mkdir(parents=True)
        manifest = workspace / "scene.json"
        manifest.write_text(json.dumps({"schema": "modly.scene-manifest.v1", "sceneRoot": "scene"}))
        self.assertEqual(ws.resolve_scene_manifest(manifest, workspace), scene)

    def test_nested_manifest_dot_resolves_to_its_directory(self):
        workspace = self.root / "workspace"
        scene = workspace / "nested" / "scene"
        scene.mkdir(parents=True)
        manifest = scene / "scene.json"
        manifest.write_text(json.dumps({"schema": "modly.scene-manifest.v1", "sceneRoot": "."}))
        self.assertEqual(ws.resolve_scene_manifest(manifest, workspace), scene)

    def test_official_trellis2_base_config_passes_without_weakening_model_checks(self):
        base = self.root / "base"
        base.mkdir()
        models = {f"model_{index}": f"ckpts/{name}" for index, name in enumerate(ws._BASE_MODELS)}
        pipeline = {"name": "Trellis2ImageTo3DPipeline", "args": {"models": models}}
        (base / "pipeline.json").write_text(json.dumps(pipeline))
        for name in ws._BASE_MODELS:
            for extension in ("json", "safetensors"):
                checkpoint = base / "ckpts" / f"{name}.{extension}"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"model")
        for auxiliary, files in ws._AUXILIARY_FILES.items():
            for relative in files:
                asset = base / "auxiliary" / auxiliary / relative
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.write_bytes(b"model")
        naf = self.root / "naf_release.pth"
        naf.write_bytes(b"model")
        size_patch = patch.object(naf_checkpoint, "NAF_SIZE", 5)
        hash_patch = patch.object(naf_checkpoint, "NAF_SHA256", hashlib.sha256(b"model").hexdigest())
        size_patch.start()
        hash_patch.start()
        self.addCleanup(size_patch.stop)
        self.addCleanup(hash_patch.stop)
        ws.validate_base(base, naf)
        pipeline["name"] = "Pixal3DImageTo3DPipeline"
        (base / "pipeline.json").write_text(json.dumps(pipeline))
        ws.validate_base(base, naf)
        pipeline["name"] = "UnknownPipeline"
        (base / "pipeline.json").write_text(json.dumps(pipeline))
        with self.assertRaisesRegex(ValueError, "Pixal3D base pipeline"):
            ws.validate_base(base, naf)
        pipeline["name"] = "Trellis2ImageTo3DPipeline"
        pipeline["args"]["models"]["model_0"] = "../remote"
        (base / "pipeline.json").write_text(json.dumps(pipeline))
        with self.assertRaisesRegex(ValueError, "nonlocal model reference"):
            ws.validate_base(base, naf)

    def test_private_overlay_localizes_dino_and_naf_without_mutating_base(self):
        base = self.root / "base"
        base.mkdir()
        source_json = {"name": "Pixal3DImageTo3DPipeline", "args": {
            "models": {},
            "image_cond_model": {"args": {"model_name": ws._DINO}},
            "rembg_model": {"args": {"model_name": ws._RMBG}},
        }}
        (base / "pipeline.json").write_text(json.dumps(source_json))
        for auxiliary, files in ws._AUXILIARY_FILES.items():
            for relative in files:
                asset = base / "auxiliary" / auxiliary / relative
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.write_bytes(b"model")
        root = self.root / "run"
        root.mkdir()
        overlay = ws._private_overlay(root, base)
        self.assertEqual(json.loads((base / "pipeline.json").read_text()), source_json)
        config = json.loads((overlay / "pretrained/Pixal3D/pipeline.json").read_text())
        self.assertEqual(config["args"]["image_cond_model"]["args"]["model_name"], str(base / "auxiliary/dinov3"))
        self.assertEqual(config["args"]["rembg_model"]["args"]["model_name"], str(base / "auxiliary/rmbg"))
        self.assertNotIn(ws._RMBG, json.dumps(config))
        self.assertNotIn(ws._DINO, (overlay / "inference.py").read_text())
        self.assertIn('raise FileNotFoundError("WorldSculpt requires local model files")',
                      (overlay / "pixal3d/models/__init__.py").read_text())
        self.assertIn("local_files_only=True", (overlay / "pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py").read_text())
        self.assertNotIn('torch.hub.load(\n                "valeoai/NAF"', (overlay / "pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py").read_text())
        self.assertIn(ws._DINO, (ws.SOURCE / "inference.py").read_text())

    def test_stages_use_real_entrypoints_and_validate_boundaries(self):
        scene = self.root / "scene"
        scene.mkdir()
        adapters = self.root / "adapters"
        adapters.mkdir()
        base = self.root / "base"
        base.mkdir()
        naf = self.root / "naf.pth"
        naf.write_bytes(b"local")
        output = self.root / "output"
        calls = []
        progress = []
        def stage(args, cwd, env, cancel):
            calls.append(args)
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["WORLDSCULPT_NAF_CHECKPOINT"], str(naf))
            self.assertEqual(env["ATTN_BACKEND"], "sdpa")
            self.assertEqual(env["SPARSE_ATTN_BACKEND"], "sdpa")
            if args[0] == "reconstruct_batch.py":
                mesh = Path(args[args.index("--case_root") + 1]) / "_recon/obj01/mesh.pt"
                mesh.parent.mkdir(parents=True)
                mesh.write_bytes(b"stage artifact")
            return "stage complete"
        with patch.dict(os.environ, {"ATTN_BACKEND": "flash_attn", "SPARSE_ATTN_BACKEND": "flash_attn"}), \
             patch.object(ws, "missing_runtime", return_value=[]), patch.object(ws, "validate_scene", return_value=("obj01",)), \
             patch.object(ws, "validate_adapters"), patch.object(ws, "validate_base"), \
             patch.object(ws, "_private_overlay", return_value=self.root), \
             patch.object(ws, "_local_adapter_config", side_effect=lambda source, destination, base: destination), \
            patch.object(ws, "validate_crops"), patch.object(ws, "validate_output", return_value=self.root / "scene.glb"), \
            patch.object(ws, "_run", side_effect=stage):
            self.assertEqual(ws.run_worldsculpt(scene_dir=scene, adapter_root=adapters, base_root=base,
                             naf_path=naf, output_dir=output, workspace_dir=self.root,
                             face_budget=50000,
                             progress_cb=lambda pct, step: progress.append((pct, step))), self.root / "scene.glb")
        self.assertEqual([command[0] for command in calls],
                         ["prepare_crops_scene.py", "reconstruct_batch.py", "compose_scene.py"])
        self.assertIn("--no_tex", calls[1])
        self.assertIn("--no_glb", calls[1])
        self.assertIn("--normal", calls[2])
        self.assertEqual(calls[2][calls[2].index("--face_budget") + 1], "50000")
        self.assertEqual(calls[2][calls[2].index("--glb_decimation") + 1], "50000")
        self.assertEqual([pct for pct, _ in progress], [5, 30, 75, 100])
        for command in calls[1:]:
            self.assertEqual(command[command.index("--instances") + 1], "obj01")

    def test_output_parent_may_contain_normalized_scene_but_not_overlap_it(self):
        workflows = self.root / "workspace" / "Workflows"
        scene = workflows / "scene-normalized-123"
        scene.mkdir(parents=True)
        naf = self.root / "naf.pth"
        naf.write_bytes(b"local")

        common = dict(scene_dir=scene, adapter_root=self.root, workspace_dir=self.root / "workspace",
                      base_root=self.root, naf_path=naf)
        with patch.object(ws, "missing_runtime", return_value=[]), \
             patch.object(ws, "validate_scene", return_value=("obj01",)), \
             patch.object(ws, "validate_adapters"), patch.object(ws, "validate_base"), \
             patch.object(ws, "prepare_case_root", side_effect=RuntimeError("guard passed")):
            with self.assertRaisesRegex(RuntimeError, "guard passed"):
                ws.run_worldsculpt(output_dir=workflows, **common)
            for unsafe in (scene, scene / "generated"):
                with self.subTest(output=unsafe):
                    with self.assertRaisesRegex(ValueError, "separate from input scene"):
                        ws.run_worldsculpt(output_dir=unsafe, **common)
                    self.assertFalse((scene / "generated").exists())

    def test_output_parent_rejects_workspace_escapes_before_mutating_filesystem(self):
        workspace = self.root / "workspace"
        scene = workspace / "Workflows" / "scene-normalized-123"
        scene.mkdir(parents=True)
        outside = self.root / "outside"
        outside.mkdir()
        symlink = workspace / "escape"
        symlink.symlink_to(outside, target_is_directory=True)
        naf = self.root / "naf.pth"
        naf.write_bytes(b"local")
        common = dict(scene_dir=scene, adapter_root=self.root, base_root=self.root,
                      naf_path=naf, workspace_dir=workspace)
        candidates = (
            outside / "direct",
            workspace / ".." / "traversal",
            symlink / "symlinked",
            Path("/"),
        )

        with patch.object(ws, "missing_runtime", return_value=[]), \
             patch.object(ws, "validate_scene", return_value=("obj01",)), \
             patch.object(ws, "validate_adapters"), patch.object(ws, "validate_base"), \
             patch.object(ws, "prepare_case_root") as prepare_case_root:
            for candidate in candidates:
                with self.subTest(output=candidate):
                    with self.assertRaisesRegex(ValueError, "inside the workspace"):
                        ws.run_worldsculpt(output_dir=candidate, **common)

        prepare_case_root.assert_not_called()
        self.assertFalse((outside / "direct").exists())
        self.assertFalse((self.root / "traversal").exists())
        self.assertFalse((outside / "symlinked").exists())

    def test_filesystem_root_cannot_be_used_as_workspace_authority(self):
        scene = self.root / "scene"
        scene.mkdir()
        naf = self.root / "naf.pth"
        naf.write_bytes(b"local")
        with patch.object(ws, "missing_runtime", return_value=[]):
            with self.assertRaisesRegex(ValueError, "non-root directory"):
                ws.run_worldsculpt(scene_dir=scene, adapter_root=self.root,
                                   base_root=self.root, naf_path=naf,
                                   output_dir=self.root / "output", workspace_dir=Path("/"))
        self.assertFalse((self.root / "output").exists())

    def test_base_rejects_symlinked_parent(self):
        base = self.root / "base"
        base.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "pipeline.json").write_text('{"name":"Pixal3DImageTo3DPipeline","args":{"models":{"one":"ckpts/one"}}}')
        (outside / "ckpts").mkdir()
        base.joinpath("pipeline.json").write_bytes((outside / "pipeline.json").read_bytes())
        (base / "ckpts").symlink_to(outside / "ckpts", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "aliases|escapes"):
            ws._under(base, base / "ckpts/one.json", "model")

    def test_missing_runtime_blocks_before_writing(self):
        with patch.object(ws, "missing_runtime", return_value=["peft"]):
            with self.assertRaisesRegex(RuntimeError, "peft"):
                ws.run_worldsculpt(scene_dir=self.root, adapter_root=self.root,
                                   base_root=self.root, naf_path=self.root / "naf", output_dir=self.root / "out",
                                   workspace_dir=self.root)
        self.assertFalse((self.root / "out").exists())

    def test_subprocess_failure_retains_only_log_tail(self):
        script = self.root / "noisy.py"
        script.write_text('import sys\nsys.stdout.write("x" * 1000000 + "END")\nsys.exit(7)\n')
        with patch.object(ws, "python_path", return_value=Path(sys.executable)):
            with self.assertRaisesRegex(RuntimeError, "failed \\(7\\):") as raised:
                ws._run([str(script)], self.root, dict(os.environ), None)
        self.assertLess(len(str(raised.exception)), 3300)
        self.assertTrue(str(raised.exception).endswith("END"))

    def test_successful_subprocess_returns_bounded_log_tail(self):
        script = self.root / "noisy_success.py"
        script.write_text('import sys\nsys.stdout.write("x" * 1000000 + "ROOT_CAUSE")\n')
        with patch.object(ws, "python_path", return_value=Path(sys.executable)):
            tail = ws._run([str(script)], self.root, dict(os.environ), None)
        self.assertIsInstance(tail, str)
        self.assertLessEqual(len(tail), 3000)
        self.assertTrue(tail.endswith("ROOT_CAUSE"))

    def test_worldsculpt_cancellation_reaps_stage_descendants(self):
        marker = self.root / "worldsculpt-descendant-survived"
        child = (
            "import pathlib,time;time.sleep(1.0);"
            f"pathlib.Path({str(marker)!r}).write_text('survived')"
        )
        script = self.root / "parent.py"
        script.write_text(
            "import subprocess,sys,time\n"
            f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
            "time.sleep(10)\n",
            encoding="utf-8",
        )
        cancelled = threading.Event()
        timer = threading.Timer(0.15, cancelled.set)
        timer.start()
        try:
            with patch.object(ws, "python_path", return_value=Path(sys.executable)):
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    ws._run([str(script)], self.root, dict(os.environ), cancelled)
        finally:
            timer.cancel()
        time.sleep(1.2)
        self.assertFalse(marker.exists())

    def test_zero_exit_missing_artifacts_surface_stage_log_tail(self):
        scene = self.root / "scene"
        scene.mkdir()
        for failed_stage in ("crop", "reconstruct", "compose"):
            with self.subTest(stage=failed_stage):
                calls = []
                def stage(args, cwd, env, cancel):
                    calls.append(args[0])
                    if args[0] == "reconstruct_batch.py" and failed_stage != "reconstruct":
                        mesh = Path(args[args.index("--case_root") + 1]) / "_recon/obj01/mesh.pt"
                        mesh.parent.mkdir(parents=True)
                        mesh.write_bytes(b"stage artifact")
                    return "x" * 10000 + f"{failed_stage}: ROOT_CAUSE"
                with patch.object(ws, "missing_runtime", return_value=[]), \
                     patch.object(ws, "validate_scene", return_value=("obj01",)), \
                     patch.object(ws, "validate_adapters"), patch.object(ws, "validate_base"), \
                     patch.object(ws, "_private_overlay", return_value=self.root), \
                     patch.object(ws, "_local_adapter_config", side_effect=lambda source, destination, base: destination), \
                     patch.object(ws, "validate_crops", side_effect=ValueError("missing crops") if failed_stage == "crop" else None), \
                     patch.object(ws, "validate_output", side_effect=ValueError("missing GLB")) as validate_output, \
                     patch.object(ws, "_run", side_effect=stage):
                    naf = self.root / "naf.pth"
                    naf.write_bytes(b"local")
                    with self.assertRaisesRegex(RuntimeError, "ROOT_CAUSE") as raised:
                        ws.run_worldsculpt(scene_dir=scene, adapter_root=self.root,
                                           base_root=self.root, naf_path=naf,
                                           output_dir=self.root / f"output-{failed_stage}",
                                           workspace_dir=self.root)
                    self.assertLess(len(str(raised.exception)), 3600)
                    if failed_stage != "compose":
                        self.assertNotIn("compose_scene.py", calls)
                        validate_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()

class WorldSculptLocalizationTests(unittest.TestCase):
    def test_adapter_config_localized_in_private_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "source.json"
            original.write_text(json.dumps({"models": {"image_cond": {"args": {"model_name": ws._DINO}}}}))
            private = root / "private.json"
            base = root / "base"
            ws._local_adapter_config(original, private, base)
            self.assertEqual(json.loads(private.read_text())["models"]["image_cond"]["args"]["model_name"], str(base / "auxiliary/dinov3"))
            self.assertEqual(json.loads(original.read_text())["models"]["image_cond"]["args"]["model_name"], ws._DINO)

    def test_adapter_config_rejects_unknown_remote_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "source.json"
            original.write_text(json.dumps({"model_name": "owner/other-model"}))
            with self.assertRaisesRegex(ValueError, "unlocalized"):
                ws._local_adapter_config(original, root / "private.json", root / "base")
