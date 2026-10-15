"""Portable contracts for the experimental advanced Windows lanes (not GPU proof)."""
import ast
import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from pixal3d_extension import worldsculpt as ws, worldsculpt_lane as lane
from pixal3d_extension import scene_prepare_lane as scene

ROOT = Path(__file__).resolve().parents[1]


class BlackwellAdvancedTests(unittest.TestCase):
    def test_windows_worldsculpt_exact_stack_and_paths(self):
        with patch.object(lane.sys, 'platform', 'win32'), patch.object(lane.platform, 'machine', return_value='AMD64'):
            self.assertEqual(lane.python_path(ROOT), ROOT / 'venv-worldsculpt/Scripts/python.exe')
            lane._platform_gate({'machine': 'AMD64', 'python': [3, 11], 'torch': '2.7.1+cu128', 'cuda': '12.8'})
            with self.assertRaises(RuntimeError):
                lane._platform_gate({'machine': 'AMD64', 'python': [3, 12], 'torch': '2.7.1+cu128', 'cuda': '12.8'})

    def test_scene_windows_is_explicit_experimental_312(self):
        with patch.object(scene.sys, 'platform', 'win32'), patch.object(scene.platform, 'machine', return_value='AMD64'):
            self.assertTrue(scene.capability()['supported'])
            self.assertEqual(scene.capability()['channel'], 'experimental')
            self.assertTrue(scene.da3_capability()['supported'])

    def test_worldsculpt_overlay_never_uses_symlinks_and_constants_compile(self):
        self._assert_worldsculpt_overlay()

    def test_worldsculpt_clean_distribution_preserves_every_pinned_blob_and_overlay(self):
        expected = {'worldsculpt_upstream/' + name: digest for digest, name in (
            line.split(None, 1) for line in (ws.SOURCE / 'EXPECTED_GIT_BLOBS').read_text().splitlines())}
        self.assertEqual(len(expected), 118)
        if (ROOT / '.git').exists():
            files = subprocess.check_output(['git', 'ls-files', '-z', '--', 'worldsculpt_upstream'], cwd=ROOT).decode().split('\0')
            files = [name for name in files if name]
        else:
            # A GitHub/CI export has no Git metadata; validate its shipped bytes.
            files = [path.relative_to(ROOT).as_posix() for path in ws.SOURCE.rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc']
        self.assertFalse(set(expected) - set(files), 'Pinned source missing from the published inventory')
        payload = io.BytesIO()
        with ZipFile(payload, 'w') as archive:
            for name in sorted(files):
                self.assertFalse((ROOT / name).is_symlink())
                archive.write(ROOT / name, name)
        with tempfile.TemporaryDirectory(prefix='tracked WorldSculpt é ') as directory, ZipFile(payload) as archive:
            projection = Path(directory).resolve()
            for name, digest in expected.items():
                data = archive.read(name)
                self.assertEqual(hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest(), digest, name)
            archive.extractall(projection)
            with patch.object(ws, 'SOURCE', projection / 'worldsculpt_upstream'), patch.object(ws.sys, 'platform', 'win32'):
                self._assert_worldsculpt_overlay()
        ignore = (ROOT / '.gitignore').read_text().splitlines()
        self.assertIn('/models/', ignore)
        self.assertNotIn('models/', ignore)

    def _assert_worldsculpt_overlay(self):
        with tempfile.TemporaryDirectory(prefix='advanced unicode é ') as directory:
            root = Path(directory).resolve()
            base = root / 'managed base'
            base.mkdir()
            (base / 'pipeline.json').write_text(json.dumps({'name':'Pixal3DImageTo3DPipeline', 'args': {'models': {'ss': 'ckpts/' + ws._BASE_MODELS[0]}}}))
            (base / 'ckpts').mkdir()
            for suffix in ('json', 'safetensors'):
                (base / 'ckpts' / (ws._BASE_MODELS[0] + '.' + suffix)).write_bytes(b'local')
            run = root / 'private'; run.mkdir()
            with patch.object(Path, 'symlink_to', side_effect=AssertionError('No Developer Mode dependency')):
                overlay = ws._private_overlay(run, base)
            data = json.loads((overlay / 'pretrained/Pixal3D/pipeline.json').read_text())
            self.assertEqual(data['args']['models']['ss'], str(base / 'ckpts' / ws._BASE_MODELS[0]))
            self.assertFalse(any(p.is_symlink() for p in overlay.rglob('*')))
            for relative in ('inference.py','reconstruct_object.py','mv_inference_common.py','pixal3d/pipelines/base.py'):
                source = (overlay / relative).read_text()
                compile(source, relative, 'exec')
            self.assertIn(str(base / 'ckpts').replace('\\', '\\\\'), (overlay / 'mv_inference_common.py').read_text())
            self.assertIn('os.path.isabs(v)', (overlay / 'pixal3d/pipelines/base.py').read_text())

    def test_mv_builder_blackwell_metadata_is_separate(self):
        spec = importlib.util.spec_from_file_location('builder', ROOT / 'tools/wheelhouse/build-pixal3d-mv-python-wheel.py')
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        from zipfile import ZipFile
        with ZipFile(ROOT / 'wheels/pixal3d_core-0.1.0+modly-py3-none-any.whl') as archive:
            original = archive.read(builder.METADATA)
        metadata = builder.platform_metadata(original, policy='blackwell').decode()
        self.assertIn('Requires-Dist: natten==0.21.6', metadata)
        self.assertIn('Version: 0.1.0+modly.blackwell.mv', metadata)
        self.assertIn('Requires-Dist: natten==0.21.0', builder.platform_metadata(original).decode())

    def test_mv_preflight_checks_da3_only_when_automatic_cameras_are_selected(self):
        from generator import Pixal3DGenerator
        generator = Pixal3DGenerator()
        generator.MODEL_NODE_ID = "generate-mv"
        generator.shared_model_dirs = {"pixal3d-base":"base", "pixal3d-mv":"mv"}
        with patch("pixal3d_extension.multiview.validate_mv_pipeline_config"), \
             patch("pixal3d_extension.scene_prepare.validate_da3_weights") as weights, \
             patch("pixal3d_extension.scene_prepare_lane.validate_da3_runtime") as calibration:
            generator._preflight_non_naf_shared_assets({"view_layout":"declared_roles"})
            weights.assert_not_called(); calibration.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "da3-base"):
                generator._preflight_non_naf_shared_assets({"view_layout":"auto"})

    def test_mv_readiness_does_not_claim_unselected_auto_calibration(self):
        from generator import Pixal3DGenerator
        generator = Pixal3DGenerator()
        generator.MODEL_NODE_ID = "generate-mv"
        generator.shared_model_dirs = {"pixal3d-base":"base", "pixal3d-mv":"mv"}
        with patch("pixal3d_extension.multiview.missing_mv_assets", return_value=[]), \
             patch("pixal3d_extension.multiview.mv_runtime_available", return_value=True), \
             patch("pixal3d_extension.scene_prepare_lane.da3_capability", return_value={"supported":True}):
            unselected = generator.readiness_status()
            self.assertTrue(unselected["ok"])
            self.assertEqual(unselected["camera_mode"], "not-selected")
            self.assertTrue(generator.readiness_status({"view_layout":"declared_roles"})["ok"])
            self.assertFalse(generator.readiness_status({"view_layout":"auto"})["ok"])

    def test_worldsculpt_bootstrap_preserves_vendored_source_priority(self):
        source = ws._bootstrap_code()
        self.assertLess(source.index('_install_windows_native_module_aliases()'), source.index('runpy.run_path'))
        self.assertIn('sys.path.insert(0, str(source))', source)
        self.assertIn('_install_natten_fallback()', source)
        self.assertIn('_silence_flex_gemm_autotuners()', source)


if __name__ == '__main__':
    unittest.main()
