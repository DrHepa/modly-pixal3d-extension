import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from pixal3d_extension import naf_checkpoint, worldsculpt as ws


class WorldSculptRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

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
        self._assert_staged_orchestration()

    def test_stages_canonicalize_owned_aliased_temp_before_artifact_validation(self):
        owned = self.root / "owned TEMP é spaces"
        owned.mkdir()
        temporary_directory = tempfile.TemporaryDirectory

        @contextmanager
        def aliased_temp(*args, **kwargs):
            with temporary_directory(*args, **kwargs, dir=owned) as directory:
                actual = Path(directory)
                aliased = actual / ".." / actual.name
                self.assertNotEqual(aliased, actual)
                self.assertEqual(aliased.resolve(strict=True), actual)
                yield str(aliased)

        with patch.object(ws.tempfile, "TemporaryDirectory", aliased_temp):
            self._assert_staged_orchestration()
        self.assertEqual(list(owned.iterdir()), [])

    def _assert_staged_orchestration(self):
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
            if args[0] == "compose_scene.py":
                scene_output = Path(args[args.index("--output_dir") + 1])
                scene_output.mkdir(parents=True)
                (scene_output / "scene_mesh.glb").write_bytes(b"mesh")
                (scene_output / "scene.glb").write_bytes(b"scene")
            return "stage complete"
        with patch.dict(os.environ, {"ATTN_BACKEND": "flash_attn", "SPARSE_ATTN_BACKEND": "flash_attn"}), \
             patch.object(ws, "missing_runtime", return_value=[]), patch.object(ws, "validate_scene", return_value=("obj01",)), \
             patch.object(ws, "validate_adapters"), patch.object(ws, "validate_base"), \
             patch.object(ws, "_private_overlay", return_value=self.root), \
             patch.object(ws, "_local_adapter_config", side_effect=lambda source, destination, base: destination), \
            patch.object(ws, "validate_crops"), patch.object(ws, "validate_output", return_value=self.root / "scene.glb"), \
            patch.object(ws, "_run", side_effect=stage):
            result = ws.run_worldsculpt(scene_dir=scene, adapter_root=adapters, base_root=base,
                                        naf_path=naf, output_dir=output, workspace_dir=self.root,
                                        face_budget=50000,
                                        progress_cb=lambda pct, step: progress.append((pct, step)))
        self.assertEqual(result.parent.parent, output)
        self.assertEqual(result.name, "scene.glb")
        self.assertEqual({entry.name for entry in result.parent.iterdir()}, {"scene.glb", "scene_mesh.glb"})
        self.assertEqual(result.read_bytes(), b"scene")
        self.assertEqual((result.parent / "scene_mesh.glb").read_bytes(), b"mesh")
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

    def test_cancellation_during_final_copy_cleans_partial_publication(self):
        case = self.root / "case"
        scene = case / "_scene"
        scene.mkdir(parents=True)
        (scene / "scene.glb").write_bytes(b"x" * (ws._COPY_CHUNK_SIZE * 2))
        (scene / "scene_mesh.glb").write_bytes(b"y" * (ws._COPY_CHUNK_SIZE * 2))
        output = self.root / "output"
        output.mkdir()

        class CancelDuringCopy:
            def __init__(self):
                self.checks = 0
            def is_set(self):
                self.checks += 1
                return self.checks >= 3

        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            ws._publish_final_glbs(case, output, CancelDuringCopy())
        self.assertEqual(list(output.iterdir()), [])

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
                        output = self.root / f"output-{failed_stage}"
                        ws.run_worldsculpt(scene_dir=scene, adapter_root=self.root,
                                           base_root=self.root, naf_path=naf,
                                           output_dir=output,
                                           workspace_dir=self.root)
                    self.assertLess(len(str(raised.exception)), 3600)
                    self.assertEqual(list(output.iterdir()), [])
                    if failed_stage != "compose":
                        self.assertNotIn("compose_scene.py", calls)
                        validate_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()

class WorldSculptLocalizationTests(unittest.TestCase):
    def test_adapter_config_localized_in_private_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original = root / "source.json"
            original.write_text(json.dumps({"models": {"image_cond": {"args": {"model_name": ws._DINO}}}}))
            private = root / "private.json"
            base = root / "base"
            ws._local_adapter_config(original, private, base)
            self.assertEqual(json.loads(private.read_text())["models"]["image_cond"]["args"]["model_name"], str(base / "auxiliary/dinov3"))
            self.assertEqual(json.loads(original.read_text())["models"]["image_cond"]["args"]["model_name"], ws._DINO)

    def test_adapter_config_rejects_unknown_remote_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original = root / "source.json"
            original.write_text(json.dumps({"model_name": "owner/other-model"}))
            with self.assertRaisesRegex(ValueError, "unlocalized"):
                ws._local_adapter_config(original, root / "private.json", root / "base")


class WorldSculptPthTests(unittest.TestCase):
    """Locale-reader source semantics and probe diagnostics, not Windows/Torch proof."""

    @staticmethod
    def _fixture(root):
        from pixal3d_extension import worldsculpt_lane as lane
        primary = lane.primary_python(root)
        primary.parent.mkdir(parents=True)
        primary.write_bytes(b'fixture interpreter, never executed')
        primary_site = root/'venv/Lib/site-packages'
        primary_site.mkdir(parents=True)
        overlay_site = root/'venv-worldsculpt/Lib/site-packages'
        overlay_site.mkdir(parents=True)
        info = {'site':str(primary_site),'prefix':str(root/'venv'),'executable':str(primary)}
        return primary_site, overlay_site, info

    def test_repair_ascii_pth_preserves_locale_unicode_non_bmp_and_overlay_precedence(self):
        import importlib
        import io
        import site
        from pixal3d_extension import worldsculpt_lane as lane
        with tempfile.TemporaryDirectory(prefix='worldsculpt pth \u00e9 spaces \U0001f984 ') as directory:
            root = Path(directory).resolve()
            primary_site, overlay_site, info = self._fixture(root)
            link = overlay_site/'primary-pixal3d.pth'
            link.write_bytes((str(primary_site)+'\n').encode('utf-8'))
            old_write = Path.write_bytes
            with patch.object(lane,'_probe',return_value=info), patch.object(lane,'_platform_gate'), \
                 patch.object(lane,'verify_wheels',return_value=[]), patch.object(lane,'_validate_target',return_value=overlay_site), \
                 patch.object(lane,'_validate_runtime'), patch.object(Path,'write_bytes',autospec=True,side_effect=old_write) as writes:
                lane.provision(root)
                lane.provision(root)
            wrapper = io.TextIOWrapper
            def cp1252_reader(*args,**kwargs):
                if kwargs.get('encoding')=='locale':
                    kwargs['encoding']='cp1252'
                return wrapper(*args,**kwargs)
            shared = '_modly_worldsculpt_pth_shared_probe'
            primary_only = '_modly_worldsculpt_pth_primary_probe'
            (overlay_site/(shared+'.py')).write_text("origin='overlay'\n",encoding='ascii')
            (primary_site/(shared+'.py')).write_text("origin='primary'\n",encoding='ascii')
            (primary_site/(primary_only+'.py')).write_text("origin='primary'\n",encoding='ascii')
            try:
                with patch.object(sys,'path',list(sys.path)), patch.object(site.io,'TextIOWrapper',side_effect=cp1252_reader) as reader:
                    site.addsitedir(str(overlay_site))
                    self.assertTrue(any(call.kwargs.get('encoding')=='locale' for call in reader.call_args_list))
                    self.assertIn(str(primary_site),sys.path)
                    self.assertLess(sys.path.index(str(overlay_site)),sys.path.index(str(primary_site)))
                    self.assertEqual(importlib.import_module(shared).origin,'overlay')
                    self.assertEqual(importlib.import_module(primary_only).origin,'primary')
            finally:
                sys.modules.pop(shared,None)
                sys.modules.pop(primary_only,None)
            expected=('import sys; sys.path.append('+ascii(str(primary_site))+')\n').encode('ascii')
            self.assertEqual(link.read_bytes(),expected)
            self.assertIn(b'\\U0001f984',expected)
            self.assertNotIn(b'\\ud83e\\udd84',expected)
            self.assertNotIn(b'\r',expected)
            self.assertEqual(writes.call_count,1,'Repair rewrites legacy bytes once, then reuses canonical ASCII/LF')

    def test_probe_failure_retains_returncode_stdout_and_stderr(self):
        import subprocess
        from pixal3d_extension import worldsculpt_lane as lane
        stdout='child initialization reached\n'
        stderr="ModuleNotFoundError: No module named 'torch'\n"
        def failed(command,**kwargs):
            if kwargs.get('check'):
                raise subprocess.CalledProcessError(7,command,output=stdout,stderr=stderr)
            return subprocess.CompletedProcess(command,7,stdout,stderr)
        with patch.object(lane.subprocess,'run',side_effect=failed):
            with self.assertRaisesRegex(RuntimeError,'returncode=7') as raised:
                lane._probe(Path('/fixture/python.exe'))
        message=str(raised.exception)
        self.assertIn(stdout,message)
        self.assertIn(stderr,message)
        self.assertNotIn('BLOCKED',message)

    def test_probe_success_keeps_existing_observation(self):
        import subprocess
        from pixal3d_extension import worldsculpt_lane as lane
        info={'machine':'AMD64','python':[3,11],'torch':'2.7.1+cu128','cuda':'12.8'}
        with patch.object(lane.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps(info),'warning')):
            self.assertEqual(lane._probe(Path('/fixture/python.exe')),info)

    def test_foreign_overlay_custody_still_blocks_pth_rewrite(self):
        from pixal3d_extension import worldsculpt_lane as lane
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            primary_site, overlay_site, info=self._fixture(root)
            link=overlay_site/'primary-pixal3d.pth'
            original=(str(primary_site)+'\n').encode('utf-8')
            link.write_bytes(original)
            with patch.object(lane,'_probe',return_value=info), patch.object(lane,'_platform_gate'), \
                 patch.object(lane,'verify_wheels',return_value=[]), \
                 patch.object(lane,'_validate_target',side_effect=RuntimeError('foreign environment')), \
                 patch.object(lane,'_validate_runtime') as validation:
                with self.assertRaisesRegex(RuntimeError,'foreign environment'):
                    lane.provision(root)
            self.assertEqual(link.read_bytes(),original)
            validation.assert_not_called()

    def test_primary_site_escape_still_blocks_pth_rewrite(self):
        from pixal3d_extension import worldsculpt_lane as lane
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            primary_site, overlay_site, info=self._fixture(root)
            outside=root/'not-primary'; outside.mkdir()
            link=overlay_site/'primary-pixal3d.pth'
            link.write_bytes(b'unchanged legacy content\n')
            with patch.object(lane,'_probe',return_value={**info,'site':str(outside)}), \
                 patch.object(lane,'_platform_gate'), patch.object(lane,'verify_wheels') as wheels:
                with self.assertRaisesRegex(RuntimeError,'escapes Pixal3D venv'):
                    lane.provision(root)
            self.assertEqual(link.read_bytes(),b'unchanged legacy content\n')
            wheels.assert_not_called()
