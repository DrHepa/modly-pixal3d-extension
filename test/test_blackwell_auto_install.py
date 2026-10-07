import contextlib
import io
import types
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import setup
from modly_wheelhouse import detect_runtime_lane, load_manifest, prepare_wheelhouse, select_asset

ROOT = Path(__file__).resolve().parents[1]
LANE = setup.BLACKWELL_RUNTIME_LANE
RUNTIME = detect_runtime_lane({'cuda_version': 128, 'gpu_sm': 120}, system='Windows', machine='AMD64', python_tag='cp311')


class BlackwellAutomaticInstallTests(unittest.TestCase):
    def test_normal_manifest_selects_exact_experimental_asset(self):
        asset = select_asset(load_manifest(ROOT / 'wheelhouse.manifest.json'), RUNTIME)
        self.assertEqual(asset['id'], LANE)
        self.assertEqual(asset['sha256'], '162f43973de2dcde2987c0322dbf9f143bed87d267e8f0374618bcbe1f760a97')
        self.assertEqual(asset['size_bytes'], 295071843)
        self.assertEqual(asset['channel'], 'experimental')
        self.assertIn('wheelhouse-blackwell-candidate-v0.1.0-r3/', asset['url'])

    def test_cache_and_result_preserve_asset_release_provenance(self):
        manifest = load_manifest(ROOT / 'wheelhouse.manifest.json')
        asset = select_asset(manifest, RUNTIME)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'tiny.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('wheelhouse/test.whl', b'fixture')
            asset.update(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), size_bytes=archive.stat().st_size)
            manifest['assets'] = [asset]
            result = prepare_wheelhouse(manifest, RUNTIME, root, downloader=lambda url, path: shutil.copy2(archive, path) and archive.stat().st_size)
            marker = json.loads(next(root.rglob('.modly-wheelhouse.json')).read_text())
            for observed in (result, marker):
                self.assertEqual(observed['release_tag'], 'wheelhouse-blackwell-candidate-v0.1.0-r3')
                self.assertEqual(observed['channel'], 'experimental')
                self.assertEqual(observed['provenance']['build_run_id'], 35537470167)

    def test_base_install_never_checks_or_installs_mv_overlay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / 'venv/Scripts/python.exe'
            python.parent.mkdir(parents=True)
            python.touch()
            (root / 'natten-0.21.6-cp311-cp311-win_amd64.whl').touch()
            (root / 'cumesh_vb-1.0-cp311-cp311-win_amd64.whl').touch()
            commands = []
            def run(args, **kwargs):
                commands.append(args)
                return {'args': args, 'ok': True, 'returncode': 0}
            with patch.object(setup, 'SCRIPT_DIR', root), patch.object(setup, '_run_setup_command', side_effect=run), patch.object(setup, '_runtime_cuda_check', return_value={'ok': True, 'args': []}):
                result = setup._install_prepare_dependencies(root, wheelhouse_path=root, runtime_evidence=RUNTIME)
            self.assertEqual(result['status'], 'installed')
            self.assertEqual(len(commands), 7)  # Six installs plus pip check; no overlay.
            local = [args for args in commands if '--find-links' in args]
            self.assertEqual(len(local), 2)
            for args in local:
                self.assertIn('--force-reinstall', args)
                self.assertIn('--no-index', args)
                self.assertIn('--no-deps', args)
            self.assertFalse(any(str(root / setup.MV_CORE_WHEEL) in args for args in commands))
            self.assertTrue(any('natten==0.21.6' in args for args in commands))

    def test_repair_replaces_same_version_core_naf_and_old_cuda_local_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'natten-0.21.6-cp311-cp311-win_amd64.whl').touch()
            (root / 'cumesh_vb-1.0+cu128torch2.7-cp311-cp311-win_amd64.whl').touch()
            # Installed stable/MV versions satisfy these public pins unless pip is forced.
            installed = root / 'venv/Lib/site-packages'
            for name, version in (('pixal3d_core', '0.1.0+modly'), ('naf', '0.1.0+modly'), ('cumesh_vb', '1.0+cu124torch2.6')):
                metadata = installed / f'{name}-{version}.dist-info/METADATA'
                metadata.parent.mkdir(parents=True)
                metadata.write_text(f'Name: {name}\nVersion: {version}\n')
            plan = setup._dependency_install_plan(root, root, RUNTIME)
            self.assertIn('--force-reinstall', plan['local_wheel_flags'])
            self.assertIn('pixal3d-core==0.1.0+modly', plan['local_wheel_packages'])
            self.assertIn('naf==0.1.0+modly', plan['local_wheel_packages'])
            self.assertIn('cumesh-vb==1.0', plan['local_wheel_packages'])
            self.assertIn('--force-reinstall', plan['natten_command'])
            for lane in ('windows-x64-cp311-cuda124', 'windows-x64-cp312-cuda124', 'linux-x64-cp312-cuda124', 'linux-aarch64-cp312-cuda124'):
                system, arch, python_tag, accelerator_lane = lane.split('-')
                stable = setup._dependency_install_plan(root, root, {'os': system, 'arch': arch, 'python_tag': python_tag, 'accelerator_lane': accelerator_lane})
                self.assertEqual(stable['local_wheel_flags'], [])
                self.assertNotIn('--force-reinstall', stable['natten_command'])

    def test_kernel_results_are_required_not_import_success_only(self):
        policy = setup._dependency_policy(RUNTIME)
        probe = dict(ok=True, torch_version=policy['torch'], torchvision_version=policy['torchvision'], torch_cuda_version='12.8', gpu_sm='120', torch_cuda_available=True, natten_version='0.21.6', natten_has_libnatten=True, imports=policy['required_imports'], upstream_imports=policy['required_upstream_imports'])
        self.assertFalse(setup._validate_runtime_probe(probe, policy)['ok'])
        probe['checks'] = {name: {'status': 'PASS'} for name in ('torch_stack', 'device', 'native_imports', 'natten_lib', 'torch_cuda', 'natten_sm120')}
        self.assertTrue(setup._validate_runtime_probe(probe, policy)['ok'])
        probe['checks']['natten_sm120']['status'] = 'NOT_RUN'
        self.assertFalse(setup._validate_runtime_probe(probe, policy)['ok'])

    def test_probe_timeout_fails_structurally_and_preserves_not_run(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(setup.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['probe'], 180, output='')):
            result = setup._runtime_cuda_check(Path('python.exe'), Path(temp), Path(temp), RUNTIME)
        self.assertFalse(result['ok'])
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['checks']['natten_sm120']['status'], 'NOT_RUN')
        self.assertFalse(result['inference_validated'])

    def test_weights_free_probe_exercises_verified_naf_backend_and_stops_on_wrong_device(self):
        from pixal3d_extension.setup_probe import run_blackwell_probe
        policy = setup._dependency_policy(RUNTIME)
        class Tensor:
            def __init__(self, shape, value):
                self.shape, self.value = shape, value
            def cpu(self):
                return self
            def __add__(self, other):
                return Tensor(self.shape, self.value + other.value)
        torch = types.ModuleType('torch')
        torch.__version__, torch.version = policy['torch'], types.SimpleNamespace(cuda='12.8')
        torch.compile = lambda function=None, **kwargs: function
        torch.float32, torch.float16 = 'float32', 'float16'
        torch.ones = lambda shape, **kwargs: Tensor(shape, 1)
        torch.zeros = lambda shape, **kwargs: Tensor(shape, 0)
        torch.ones_like = lambda tensor: Tensor(tensor.shape, 1)
        torch.full = lambda shape, value: Tensor(shape, value)
        torch.equal = torch.allclose = lambda a, b, **kwargs: a.shape == b.shape and a.value == b.value
        torch.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (12, 0), get_device_name=lambda: 'mock GPU', synchronize=lambda: None)
        calls = []
        natten = types.ModuleType('natten')
        natten.HAS_LIBNATTEN = True
        natten.na2d = lambda q, k, v, **kwargs: calls.append(kwargs) or Tensor(q.shape, v.value)
        modules = {name: types.ModuleType(name) for name in policy['required_imports']}
        modules.update(torch=torch, natten=natten)
        versions = {'torchvision': policy['torchvision'], 'transformers': setup.TRANSFORMERS_VERSION, 'natten': policy['natten']}
        for sm in ((12, 0), (8, 9)):
            torch.cuda.get_device_capability = lambda: sm
            with patch.dict('sys.modules', modules), patch('pixal3d_extension.setup_probe.version', side_effect=versions.__getitem__), contextlib.redirect_stdout(io.StringIO()):
                result = run_blackwell_probe(policy)
            self.assertEqual(result['ok'], sm == (12, 0))
            self.assertFalse(result['inference_validated'])
            self.assertEqual(result['checks']['natten_sm120']['status'], 'PASS' if result['ok'] else 'NOT_RUN')
        self.assertEqual(calls, [{'kernel_size': 3, 'stride': 1, 'backend': 'cutlass-fna'}])

    def test_timeout_preserves_completed_checks_even_after_non_json_native_output(self):
        from pixal3d_extension.setup_probe import CHECK_NAMES
        progress = {'ok': False, 'checks': {name: {'status': 'NOT_RUN'} for name in CHECK_NAMES}}
        progress['checks']['torch_stack']['status'] = 'PASS'
        progress['checks']['native_imports']['status'] = 'RUNNING'
        output = json.dumps(progress) + '\nNative output, not JSON\n'
        with tempfile.TemporaryDirectory() as temp, patch.object(setup.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['probe'], 180, output=output)):
            result = setup._runtime_cuda_check(Path('python.exe'), Path(temp), Path(temp), RUNTIME)
        self.assertEqual(result['checks']['torch_stack']['status'], 'PASS')
        self.assertEqual(result['checks']['native_imports']['status'], 'FAIL')
        self.assertEqual(result['checks']['natten_sm120']['status'], 'NOT_RUN')

    def test_hosted_blackwell_probe_is_blocked_not_pass_and_dll_errors_are_failures(self):
        from tools.validation.windows_dependency_install import classify_probe
        probe = dict(torch_version='2.7.1+cu128', torchvision_version='0.22.1+cu128', torch_cuda_version='12.8', torch_cuda_available=False, error='RuntimeError: Found no NVIDIA driver', checks={'natten_sm120': {'status': 'NOT_RUN'}})
        self.assertEqual(classify_probe(probe, True), 'BLOCKED')
        with self.assertRaises(AssertionError):
            classify_probe({**probe, 'error': 'ImportError: DLL load failed'}, True)


if __name__ == '__main__':
    unittest.main()
