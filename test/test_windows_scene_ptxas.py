"""Private compiler selection tests; mocked execution is not Windows GPU evidence."""
import contextlib
import hashlib
import io
import json
import os
import stat
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pixal3d_extension import scene_prepare, windows_scene_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]
MEMBER = 'triton/backends/nvidia/bin/ptxas.exe'
VERSION = 'ptxas: NVIDIA (R) Ptx optimizing assembler\nCuda compilation tools, release 12.8, V12.8.61\n'


class WindowsScenePtxasTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='private compiler é spaces ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.compiler = self.root / 'venv-scene-prep/Lib/site-packages' / MEMBER
        self.compiler.parent.mkdir(parents=True)
        self.compiler.write_bytes(b'fixture private compiler')
        self.lock = runtime.load_lock(ROOT)
        self.lock['ptxas'] = {'member': MEMBER, 'sha256': hashlib.sha256(self.compiler.read_bytes()).hexdigest(),
                              'size': self.compiler.stat().st_size, 'release': '12.8', 'version': '12.8.61'}
        (self.root / runtime.LOCK_NAME).write_text(json.dumps(self.lock), encoding='utf-8')
        self.result = subprocess.CompletedProcess([], 0, stdout=VERSION, stderr='')

    def managed(self, env):
        return runtime.managed_ptxas_environment(self.root, env)

    def test_owned_child_ignores_poison_and_logs_identity_without_mutating_parent(self):
        poison = str(self.root / 'external CUDA 13.4/ptxas.exe')
        with patch.dict(os.environ, {'TRITON_PTXAS_PATH': poison}), \
             patch.object(runtime.subprocess, 'run', return_value=self.result) as run, \
             contextlib.redirect_stderr(io.StringIO()) as log:
            parent = dict(os.environ)
            child = self.managed(os.environ)
            self.assertEqual(dict(os.environ), parent)
        self.assertEqual(child['TRITON_PTXAS_PATH'], str(self.compiler))
        self.assertEqual(run.call_args.args[0], [str(self.compiler), '--version'])
        self.assertEqual(run.call_args.kwargs['env']['TRITON_PTXAS_PATH'], str(self.compiler))
        identity = json.loads(log.getvalue().split('managed PTXAS: ', 1)[1])
        self.assertEqual(identity['path'], str(self.compiler))
        self.assertEqual(identity['sha256'], self.lock['ptxas']['sha256'])
        self.assertEqual(identity['version'], '12.8.61')

    def test_absent_corrupt_and_nonregular_compilers_fail_before_execution(self):
        for invalid in ('missing', 'corrupt', 'directory'):
            with self.subTest(invalid=invalid):
                self.compiler.unlink(missing_ok=True)
                if invalid == 'corrupt': self.compiler.write_bytes(b'tampered')
                if invalid == 'directory': self.compiler.mkdir()
                with patch.object(runtime.subprocess, 'run') as run:
                    with self.assertRaisesRegex(RuntimeError, 'PTXAS.*Repair'):
                        self.managed({'TRITON_PTXAS_PATH': 'external-13.4'})
                    run.assert_not_called()
                if invalid == 'directory': self.compiler.rmdir()

    def test_file_and_parent_symlinks_are_not_followed(self):
        outside = self.root / 'outside.exe'; outside.write_bytes(self.compiler.read_bytes())
        self.compiler.unlink(); self.compiler.symlink_to(outside)
        with patch.object(runtime.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'PTXAS.*Repair'):
                self.managed({})
            run.assert_not_called()
        self.compiler.unlink()
        parent = self.compiler.parent
        parent.rmdir(); parent.symlink_to(self.root, target_is_directory=True)
        with patch.object(runtime.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'PTXAS.*Repair'):
                self.managed({})
            run.assert_not_called()

    def test_windows_reparse_point_is_rejected(self):
        original = Path.lstat
        def metadata(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == self.compiler:
                return types.SimpleNamespace(st_mode=result.st_mode, st_file_attributes=getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024))
            return result
        with patch.object(Path, 'lstat', metadata), patch.object(runtime.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'PTXAS.*Repair'):
                self.managed({})
            run.assert_not_called()

    def test_lock_member_cannot_escape_or_select_another_binary(self):
        for member in ('../outside.exe', '/outside.exe', 'C:\\CUDA\\bin\\ptxas.exe', 'triton/other.exe'):
            self.lock['ptxas']['member'] = member
            (self.root / runtime.LOCK_NAME).write_text(json.dumps(self.lock), encoding='utf-8')
            with self.subTest(member=member), patch.object(runtime.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'PTXAS.*Repair'):
                    self.managed({})
                run.assert_not_called()

    def test_version_mismatch_timeout_or_nonzero_exit_fails_closed(self):
        for result in (subprocess.CompletedProcess([], 0, stdout=VERSION.replace('12.8', '13.4'), stderr=''),
                       subprocess.CompletedProcess([], 1, stdout='', stderr='broken compiler'),
                       subprocess.TimeoutExpired('ptxas', 30)):
            kwargs = {'side_effect': result} if isinstance(result, Exception) else {'return_value': result}
            with self.subTest(result=result), patch.object(runtime.subprocess, 'run', **kwargs):
                with self.assertRaisesRegex(RuntimeError, 'PTXAS.*Repair'):
                    self.managed({})

    def test_bootstrap_commands_do_not_require_installed_triton(self):
        self.compiler.unlink()
        with patch.object(runtime.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='', stderr='')) as run:
            runtime._run(['bootstrap', '-m', 'venv'], self.root)
        self.assertEqual(run.call_args.args[0], ['bootstrap', '-m', 'venv'])

    def test_probe_and_worker_share_policy_and_keep_offline_flags(self):
        def verify(command, **kwargs):
            self.assertEqual(kwargs['env']['TRITON_PTXAS_PATH'], str(self.compiler))
            return self.result if command[-1] == '--version' else subprocess.CompletedProcess(command, 0, stdout='{}', stderr='')
        with patch.dict(os.environ, {'TRITON_PTXAS_PATH': 'external-13.4'}), \
             patch.object(runtime.subprocess, 'run', side_effect=verify), \
             patch.object(scene_prepare.sys, 'platform', 'win32'), contextlib.redirect_stderr(io.StringIO()):
            runtime._run(['owned-python', '-c', 'probe'], self.root, managed_compiler=True)
            worker = scene_prepare.offline_environment(self.root / 'cache', self.root)
        self.assertEqual(worker['TRITON_PTXAS_PATH'], str(self.compiler))
        self.assertEqual(worker['HF_HUB_OFFLINE'], '1')
        self.assertEqual(worker['TRANSFORMERS_OFFLINE'], '1')

    def test_all_setup_validation_launches_opt_in_to_managed_policy(self):
        pins = {item['name']: item['version'] for item in self.lock['dependencies']}
        observed = {**runtime.EXPECTED, 'torch_file': str(self.compiler.parents[4] / 'torch/__init__.py')}
        def command(args, root, **kwargs):
            self.assertTrue(kwargs['managed_compiler'])
            stdout = json.dumps(pins if 'md.version(name)' in args[-1] else observed)
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr='')
        with patch('pixal3d_extension.scene_prepare_lane._interpreter_custody', return_value=(True, {})), \
             patch.object(runtime, '_runtime_valid', return_value=True), \
             patch.object(runtime, 'source_metadata', return_value={}), \
             patch.object(runtime, 'validate_source_imports', return_value={}), \
             patch.object(runtime, '_run', side_effect=command) as run:
            runtime.validate(self.root, require_gpu=True)
        self.assertEqual(run.call_count, 4)
        receipt = {'schema': 'modly.scene-source-imports.v1', 'cuda_available': False,
                   'checks': {name: {'status': 'PASS'} for name in runtime.SOURCE_IMPORT_CHECKS}}
        with patch.object(runtime, '_run', return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(receipt), stderr='')) as run:
            runtime.validate_source_imports(self.root, allow_no_driver=True)
        self.assertTrue(run.call_args.kwargs['managed_compiler'])
        with patch.object(runtime, 'source_wheels', return_value=[]), \
             patch.object(runtime, '_run', return_value=subprocess.CompletedProcess([], 0, stdout='{}', stderr='')) as run:
            with self.assertRaisesRegex(RuntimeError, 'distribution set mismatch'):
                runtime.source_metadata(self.root)
        self.assertTrue(run.call_args.kwargs['managed_compiler'])

    def test_linux_worker_environment_does_not_apply_windows_policy(self):
        with patch.object(scene_prepare.sys, 'platform', 'linux'), \
             patch.object(runtime, 'managed_ptxas_environment') as managed, \
             patch.dict(os.environ, {'TRITON_PTXAS_PATH': 'linux-managed-compiler'}):
            worker = scene_prepare.offline_environment(self.root / 'cache', self.root)
        managed.assert_not_called()
        self.assertEqual(worker['TRITON_PTXAS_PATH'], 'linux-managed-compiler')

    def test_corrupt_bundle_uses_existing_private_reinstall_repair(self):
        self.compiler.write_bytes(b'tampered')
        portable = self.root / runtime.RUNTIME_NAME / 'python.exe'
        primary = self.root / 'venv/Scripts/python.exe'
        primary.parent.mkdir(parents=True); primary.write_bytes(b'healthy primary runtime')
        weights = self.root / 'models/keep.safetensors'
        weights.parent.mkdir(); weights.write_bytes(b'untouched weights')
        def validation(root, **kwargs):
            self.managed({})
            return {'status': 'installed-not-gpu-qualified'}
        commands = []
        def command(args, root, **kwargs):
            commands.append(args)
            if args[0] == str(primary):
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps([[3,11], '2.7.1+cu128', '12.8']), stderr='')
            if 'install' in args:
                self.compiler.write_bytes(b'fixture private compiler')
            return subprocess.CompletedProcess(args, 0, stdout='', stderr='')
        with patch.object(runtime, 'ensure_python', return_value=portable), \
             patch('pixal3d_extension.scene_prepare_lane._interpreter_custody', return_value=(True, {'base_prefix': str(portable.parent)})), \
             patch.object(runtime, 'validate', side_effect=validation), \
             patch.object(runtime, 'artifact', side_effect=lambda root, item: root/item['file']), \
             patch.object(runtime, 'source_wheels', return_value=[]), patch.object(runtime, '_run', side_effect=command), \
             patch.object(runtime.subprocess, 'run', return_value=self.result), contextlib.redirect_stderr(io.StringIO()):
            receipt = runtime.provision(self.root)
        installs = [args for args in commands if 'install' in args]
        self.assertEqual(len(installs), 1)
        self.assertEqual(installs[0][0], str(self.root / 'venv-scene-prep/Scripts/python.exe'))
        self.assertIn('--force-reinstall', installs[0])
        self.assertTrue(any('triton_windows-3.3.1.post21' in arg for arg in installs[0]))
        self.assertFalse(receipt['runtime_prepared'])
        self.assertEqual(primary.read_bytes(), b'healthy primary runtime')
        self.assertEqual(weights.read_bytes(), b'untouched weights')

    def test_repair_never_installs_through_compiler_parent_alias(self):
        self.compiler.unlink(); self.compiler.parent.rmdir()
        outside = self.root / 'external-cuda'; outside.mkdir()
        self.compiler.parent.symlink_to(outside, target_is_directory=True)
        portable = self.root / runtime.RUNTIME_NAME / 'python.exe'
        with patch.object(runtime, 'ensure_python', return_value=portable), \
             patch('pixal3d_extension.scene_prepare_lane._interpreter_custody', return_value=(True, {'base_prefix': str(portable.parent)})), \
             patch.object(runtime, 'validate', side_effect=RuntimeError('unsafe compiler')), \
             patch.object(runtime, '_run', return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps([[3,11], '2.7.1+cu128', '12.8']), stderr='')) as run, \
             patch.object(runtime, 'artifact') as artifact:
            with self.assertRaisesRegex(RuntimeError, 'symlinks or reparse'):
                runtime.provision(self.root)
        artifact.assert_not_called()
        self.assertEqual(run.call_count, 1, 'Only the primary metadata probe may run')
        self.assertEqual(list(outside.iterdir()), [])

    def test_checked_in_identity_matches_the_audited_official_wheel(self):
        lock = runtime.load_lock(ROOT)
        wheel = next(item for item in lock['dependencies'] if item['name'] == 'triton-windows')
        self.assertEqual(wheel['sha256'], 'c75e473e2d7edb41037b50477bda5e9fd01939ce9ac25a143c9c149bc9d8adf5')
        self.assertEqual(wheel['size'], 41508919)
        self.assertEqual(lock['ptxas'], {'member': MEMBER, 'release': '12.8', 'version': '12.8.61', 'size': 24732160,
                                        'sha256': '882bdaf6121883e4363726dc8706902459300f336296214e4c3759ac73cb6fa0'})


if __name__ == '__main__':
    unittest.main()
