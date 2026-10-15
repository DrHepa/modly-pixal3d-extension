import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pixal3d_extension import windows_scene_runtime as runtime


class WindowsSceneRuntimeCustodyTests(unittest.TestCase):
    def test_local_artifact_is_used_only_with_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / 'wheels/scene-prep/fixture.whl'
            wheel.parent.mkdir(parents=True)
            wheel.write_bytes(b'locked fixture')
            item = {'file': wheel.name, 'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}
            self.assertEqual(runtime.artifact(root, item), wheel)
            wheel.write_bytes(b'tampered')
            with self.assertRaisesRegex(RuntimeError, 'missing or tampered'):
                runtime.artifact(root, item)

    def test_artifact_rejects_parent_symlinks_without_following_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / 'outside'; outside.mkdir()
            (root / 'wheels').symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, 'symlinks or reparse'):
                runtime.artifact(root, {'file':'fixture.whl','sha256':'0'*64})
            self.assertEqual(list(outside.iterdir()), [])

    def test_portable_python_rejects_archive_traversal_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'fixture.nupkg'
            with zipfile.ZipFile(archive, 'w') as package:
                package.writestr('../escape', 'unsafe')
            item = {'installed_files': {'python.exe': hashlib.sha256(b'fixture').hexdigest()}, 'archive_prefix':'tools'}
            with patch.object(runtime, 'load_lock', return_value={'python_runtime':item}), patch.object(runtime, 'artifact', return_value=archive), patch.object(runtime.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'Unsafe member'):
                    runtime.ensure_python(root)
                run.assert_not_called()
            self.assertFalse((root / 'escape').exists())

    def test_runtime_repair_restores_only_owned_directory_and_members(self):
        with tempfile.TemporaryDirectory(prefix='runtime unicode é ') as directory:
            root = Path(directory)
            archive = root / 'fixture.nupkg'
            files = {'python.exe': b'fixture runtime', 'include/Python.h':b'header', 'libs/python312.lib':b'lib'}
            with zipfile.ZipFile(archive, 'w') as package:
                for name, data in files.items():
                    package.writestr('tools/' + name, data)
            item = {'installed_files':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}, 'archive_prefix':'tools', 'version':[3,12,10]}
            owned = root / runtime.RUNTIME_NAME
            (root / 'models').mkdir(); sentinel = root / 'models/keep'; sentinel.write_bytes(b'untouched')
            info = {'version':[3,12,10], 'machine':'AMD64', 'prefix':str(owned)}
            with patch.object(runtime, 'load_lock', return_value={'python_runtime':item}), patch.object(runtime, 'artifact', return_value=archive), patch.object(runtime.subprocess, 'run', return_value=type('Result', (), {'stdout':json.dumps(info)})()):
                self.assertEqual(runtime.ensure_python(root), owned / 'python.exe')
                (owned / 'include/Python.h').write_bytes(b'bad')
                runtime.ensure_python(root)
            self.assertEqual((owned / 'include/Python.h').read_bytes(), b'header')
            self.assertEqual(sentinel.read_bytes(), b'untouched')

    def test_production_probe_requires_exact_gpu_before_native_kernels(self):
        source = runtime.probe_code(require_gpu=True)
        self.assertIn('torch.cuda.get_device_capability(0)!=(12,0)', source)
        self.assertLess(source.index('if not torch.cuda.is_available()'), source.index('run_native_probe()'))
        self.assertNotIn('run_native_probe()', runtime.probe_code(require_gpu=False))


if __name__ == '__main__':
    unittest.main()
