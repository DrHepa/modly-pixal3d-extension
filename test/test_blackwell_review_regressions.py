"""Regressions for the three fresh-review blockers; no weights/GPU required."""
import ast
import hashlib
import inspect
import json
import os
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pixal3d_extension import worldsculpt as ws, windows_scene_runtime as runtime


class BlackwellReviewRegressions(unittest.TestCase):
    def test_actual_windows_crop_helper_materializes_frames_masks_and_directories_without_symlinks(self):
        with tempfile.TemporaryDirectory(prefix='crop unicode é ') as directory:
            root = Path(directory)
            base = root/'managed weights'; base.mkdir()
            weight = base/'do-not-copy.safetensors'; weight.write_bytes(b'original weights')
            (base/'pipeline.json').write_text(json.dumps({'name':'Pixal3DImageTo3DPipeline','args':{'models':{}}}))
            case = root/'case'; case.mkdir()
            with patch.object(ws.sys, 'platform', 'win32'):
                overlay = ws._private_overlay(case, base)
            target = overlay/'prepare_crops_scene.py'
            tree = ast.parse(target.read_text())
            helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name=='_symlink')
            namespace = {'Path':Path,'os':os,'__file__':str(target)}
            exec(compile(ast.Module(body=[helper],type_ignores=[]),str(target),'exec'),namespace)
            scene = root/'scene'; scene.mkdir()
            frame = scene/'frame.png'; frame.write_bytes(b'frame input')
            masks = scene/'masks'; masks.mkdir()
            (masks/'mask.png').write_bytes(b'mask input')
            with patch.object(os, 'symlink', side_effect=OSError(1314,'required privilege not held')):
                namespace['_symlink'](frame,case/'frames/0000.png')
                namespace['_symlink'](masks,case/'masks/object')
                frame.write_bytes(b'refreshed frame')
                namespace['_symlink'](frame,case/'frames/0000.png')
                with self.assertRaisesRegex(ValueError,'image'):
                    namespace['_symlink'](weight,case/'frames/weight.safetensors')
            self.assertEqual((case/'frames/0000.png').read_bytes(),b'refreshed frame')
            self.assertEqual((case/'masks/object/mask.png').read_bytes(),b'mask input')
            self.assertEqual(weight.read_bytes(),b'original weights')
            self.assertFalse((case/'frames/weight.safetensors').exists())
            self.assertFalse(any(path.is_symlink() for path in case.rglob('*')))
            with self.assertRaisesRegex(ValueError,'overlaps'):
                namespace['_symlink'](case/'frames',case/'frames/nested')
            self.assertIn('os.symlink(rel, dst)',(ws.SOURCE/'prepare_crops_scene.py').read_text())

    def test_failed_runtime_promotion_rolls_back_old_interpreter_without_touching_models(self):
        for lock_existing in (False,True):
            with self.subTest(lock_existing=lock_existing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root/'fixture.nupkg'
                files = {'python.exe':b'original interpreter','include/Python.h':b'header','libs/python312.lib':b'lib'}
                with zipfile.ZipFile(archive,'w') as package:
                    for name,data in files.items():package.writestr('tools/'+name,data)
                item = {'installed_files':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()},'archive_prefix':'tools','version':[3,12,10]}
                owned = root/runtime.RUNTIME_NAME
                info = {'version':[3,12,10],'machine':'AMD64','prefix':str(owned)}
                sentinel = root/'models/keep'; sentinel.parent.mkdir(); sentinel.write_bytes(b'untouched')
                with patch.object(runtime,'load_lock',return_value={'python_runtime':item}), patch.object(runtime,'artifact',return_value=archive), patch.object(runtime.subprocess,'run',return_value=types.SimpleNamespace(stdout=json.dumps(info))):
                    runtime.ensure_python(root)
                    (owned/'include/Python.h').write_bytes(b'needs repair')
                    before = {p.relative_to(owned).as_posix():p.read_bytes() for p in owned.rglob('*') if p.is_file()}
                    original_replace = Path.replace
                    def replace(path,destination):
                        if (lock_existing and path==owned) or (not lock_existing and Path(destination)==owned and path.name.startswith('.python-scene-prep-')):
                            raise PermissionError('injected Windows runtime promotion lock')
                        return original_replace(path,destination)
                    with patch.object(Path,'replace',replace), self.assertRaises((PermissionError,RuntimeError)):
                        runtime.ensure_python(root)
                self.assertTrue((owned/'python.exe').is_file())
                self.assertEqual({p.relative_to(owned).as_posix():p.read_bytes() for p in owned.rglob('*') if p.is_file()},before)
                self.assertEqual(sentinel.read_bytes(),b'untouched')

    def test_source_import_evidence_reaches_real_inference_boundaries_before_gpu_gate(self):
        code = runtime.source_import_probe_code()
        compile(code,'source-import-probe','exec')
        for anchor in ('from sam3.model_builder import build_sam3_video_predictor',
                       'from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor',
                       'from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity',
                       'from pixal3d_extension.da3_official_adapter import load_depth_anything3',
                       'load_depth_anything3()'):
            self.assertIn(anchor,code)
        self.assertNotIn('if not torch.cuda.is_available()',code)
        self.assertIn('validate_source_imports',inspect.getsource(runtime.validate))

    def test_source_import_receipts_fail_on_missing_dependency_and_never_promote_blocked_to_pass(self):
        names = ('sam3_builder','sam3_tracker','sam3_video_inference','da3_adapter')
        receipt = {'schema':'modly.scene-source-imports.v1','cuda_available':False,'checks':{name:{'status':'PASS'} for name in names}}
        def observed(value):return patch.object(runtime,'_run',return_value=types.SimpleNamespace(stdout=json.dumps(value)))
        with observed(receipt):
            result = runtime.validate_source_imports(Path('/fixture'),allow_no_driver=True)
            self.assertEqual(result['status'],'PASS')
            self.assertFalse(result['runtime_prepared'])
        receipt['checks']['sam3_builder']={'status':'FAIL','error_type':'ModuleNotFoundError','error':"No module named 'missing-real-dependency'"}
        with observed(receipt), self.assertRaisesRegex(RuntimeError,'missing-real-dependency'):
            runtime.validate_source_imports(Path('/fixture'),allow_no_driver=True)
        receipt['checks']['sam3_builder']={'status':'FAIL','error_type':'RuntimeError','error':'Found no NVIDIA driver on your system. Please check your NVIDIA GPU and driver.'}
        with observed(receipt):
            result = runtime.validate_source_imports(Path('/fixture'),allow_no_driver=True)
            self.assertEqual(result['status'],'BLOCKED')
            self.assertEqual(result['checks']['sam3_builder']['status'],'BLOCKED')
            self.assertFalse(result['runtime_prepared'])
        with observed(receipt), self.assertRaises(RuntimeError):
            runtime.validate_source_imports(Path('/fixture'),allow_no_driver=False)

    def test_source_import_receipts_reject_dll_errors_and_unreached_checks(self):
        names = ('sam3_builder','sam3_tracker','sam3_video_inference','da3_adapter')
        for failure in ({'status':'FAIL','error_type':'ImportError','error':'DLL load failed: missing module'},
                        {'status':'FAIL','error_type':'ImportError','error':'Found no NVIDIA driver on your system.'},
                        {'status':'NOT_RUN'}):
            receipt = {'schema':'modly.scene-source-imports.v1','cuda_available':False,'checks':{name:{'status':'PASS'} for name in names}}
            receipt['checks']['da3_adapter'] = failure
            with self.subTest(failure=failure), patch.object(runtime,'_run',return_value=types.SimpleNamespace(stdout=json.dumps(receipt))), self.assertRaises(RuntimeError):
                runtime.validate_source_imports(Path('/fixture'),allow_no_driver=True)

    def test_release_version_is_072(self):
        self.assertEqual(json.loads((Path(__file__).resolve().parents[1]/'manifest.json').read_text())['version'],'0.7.2')


if __name__=='__main__':
    unittest.main()
