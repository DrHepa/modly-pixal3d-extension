import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'
import test from 'node:test'

const repoRoot = dirname(dirname(fileURLToPath(import.meta.url)))
const python = process.env.PYTHON ?? 'python3'

function runPython(source) {
  const result = spawnSync(python, ['-c', source], {
    cwd: repoRoot,
    encoding: 'utf8',
  })
  if (result.status !== 0) {
    throw new Error(`python failed\nSTDOUT:\n${result.stdout}\nSTDERR:\n${result.stderr}`)
  }
  return JSON.parse(result.stdout)
}

test('wheelhouse manifest is pinned, checksum-verifiable, and selects supported cp312 cuda124 lanes', () => {
  const extensionManifest = JSON.parse(readFileSync(join(repoRoot, 'manifest.json'), 'utf8'))
  const manifest = JSON.parse(readFileSync(join(repoRoot, 'wheelhouse.manifest.json'), 'utf8'))

  assert.deepEqual(extensionManifest.setup.wheelhouse_manifest, {
    path: 'wheelhouse.manifest.json',
    schema_version: 'release-wheelhouse/v1',
  })
  assert.equal(manifest.schema_version, 'release-wheelhouse/v1')
  assert.equal(manifest.extension_id, 'pixal3d')
  assert.equal(manifest.release.tag, 'wheelhouse-v0.1.0')
  assert.equal(manifest.release.immutable_commit, '47e78252f5cb185623ba3aad564d879d30809bb6')
  assert.notEqual(manifest.release.tag, 'latest')
  assert.match(manifest.release.immutable_commit, /^[0-9a-f]{40}$/)
  assert.equal(manifest.assets.length, 4)
  for (const asset of manifest.assets) {
    assert.match(asset.sha256, /^[0-9a-f]{64}$/)
    assert.ok(asset.size_bytes > 0)
  }
  const x64Asset = manifest.assets.find((asset) => asset.id === 'linux-x64-cp312-cuda124')
  assert.ok(x64Asset)
  assert.ok(!x64Asset.packages.includes('natten'))
  assert.match(x64Asset.optional_packages[0].reason, /HAS_LIBNATTEN/)
  const windowsAsset = manifest.assets.find((asset) => asset.id === 'windows-x64-cp312-cuda124')
  assert.ok(windowsAsset)
  assert.equal(windowsAsset.filename, 'pixal3d-wheelhouse-v0.1.0-windows-x64-cp312-cuda124.zip')
  assert.equal(windowsAsset.size_bytes, 126804446)
  assert.equal(windowsAsset.sha256, 'cbb5a853576552bfe62807c793e60db9a64950f2c0f98bce7e0135d5cdf0b5c9')
  assert.ok(windowsAsset.packages.includes('flex-gemm-ap'))
  assert.ok(windowsAsset.packages.includes('nvdiffrec-render'))
  assert.ok(!windowsAsset.packages.includes('natten'))
  assert.match(windowsAsset.optional_packages[0].reason, /HAS_LIBNATTEN/)
  const windowsCp311Asset = manifest.assets.find((asset) => asset.id === 'windows-x64-cp311-cuda124')
  assert.ok(windowsCp311Asset)
  assert.equal(windowsCp311Asset.filename, 'pixal3d-wheelhouse-v0.1.0-windows-x64-cp311-cuda124.zip')
  assert.equal(windowsCp311Asset.size_bytes, 195767591)
  assert.equal(windowsCp311Asset.sha256, '3cc1ccad54f2ffb11190418e551144c0411eb4beceb3cab40be53f30848f685c')
  assert.ok(windowsCp311Asset.packages.includes('flex-gemm-ap'))
  assert.ok(windowsCp311Asset.packages.includes('natten'))
  assert.equal(windowsCp311Asset.optional_packages, undefined)
  assert.equal(manifest.fallback.vendored_wheels, 'wheels/')
  assert.equal(manifest.fallback.require_hashes, true)
  assert.equal(manifest.fallback.wheels.length, 11)

  const selected = runPython(`
import json
from pathlib import Path
from modly_wheelhouse import load_manifest, select_asset, validate_manifest
manifest = load_manifest(Path('wheelhouse.manifest.json'))
validate_manifest(manifest)
asset = select_asset(manifest, {'os':'linux','arch':'aarch64','python_tag':'cp312','accelerator_lane':'cuda124'})
print(json.dumps({'asset_id': asset['id'], 'release_tag': manifest['release']['tag'], 'cache_key': asset['cache_key']}))
`)

  assert.deepEqual(selected, {
    asset_id: 'linux-aarch64-cp312-cuda124',
    release_tag: 'wheelhouse-v0.1.0',
    cache_key: 'pixal3d/0.1.0/linux-aarch64-cp312-cuda124',
  })

  const selectedX64 = runPython(`
import json
from pathlib import Path
from modly_wheelhouse import load_manifest, select_asset, validate_manifest
manifest = load_manifest(Path('wheelhouse.manifest.json'))
validate_manifest(manifest)
asset = select_asset(manifest, {'os':'linux','arch':'x64','python_tag':'cp312','accelerator_lane':'cuda124'})
print(json.dumps({'asset_id': asset['id'], 'cache_key': asset['cache_key'], 'has_natten': 'natten' in asset.get('packages', [])}))
`)

  assert.deepEqual(selectedX64, {
    asset_id: 'linux-x64-cp312-cuda124',
    cache_key: 'pixal3d/0.1.0/linux-x64-cp312-cuda124',
    has_natten: false,
  })

  const selectedWindows = runPython(`
import json
from pathlib import Path
from modly_wheelhouse import load_manifest, select_asset, validate_manifest
manifest = load_manifest(Path('wheelhouse.manifest.json'))
validate_manifest(manifest)
asset = select_asset(manifest, {'os':'windows','arch':'x64','python_tag':'cp312','accelerator_lane':'cuda124'})
print(json.dumps({'asset_id': asset['id'], 'cache_key': asset['cache_key'], 'has_natten': 'natten' in asset.get('packages', [])}))
`)

  assert.deepEqual(selectedWindows, {
    asset_id: 'windows-x64-cp312-cuda124',
    cache_key: 'pixal3d/0.1.0/windows-x64-cp312-cuda124',
    has_natten: false,
  })

  const selectedWindowsCp311 = runPython(`
import json
from pathlib import Path
from modly_wheelhouse import load_manifest, select_asset, validate_manifest
manifest = load_manifest(Path('wheelhouse.manifest.json'))
validate_manifest(manifest)
asset = select_asset(manifest, {'os':'windows','arch':'x64','python_tag':'cp311','accelerator_lane':'cuda124'})
print(json.dumps({'asset_id': asset['id'], 'cache_key': asset['cache_key'], 'has_natten': 'natten' in asset.get('packages', [])}))
`)

  assert.deepEqual(selectedWindowsCp311, {
    asset_id: 'windows-x64-cp311-cuda124',
    cache_key: 'pixal3d/0.1.0/windows-x64-cp311-cuda124',
    has_natten: true,
  })
})

test('wheelhouse lane helpers support Windows venv Python paths, win_amd64 tags, and consistent cuda124 runtime lanes', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup
from modly_wheelhouse import _wheel_is_compatible, runtime_lane_id
from pixal3d_extension import readiness, runtime

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    windows_python = root / 'venv' / 'Scripts' / 'python.exe'
    windows_python.parent.mkdir(parents=True)
    windows_python.write_text('', encoding='utf-8')
    wheelhouse = root / 'wheels'
    wheelhouse.mkdir()
    calls = []
    setup._run_setup_command = lambda command, *, cwd: calls.append({'command': command, 'cwd': str(cwd)}) or {'args': command, 'returncode': 0, 'stdout_tail': '{"HAS_LIBNATTEN": false, "importable": true, "ok": true, "torch_cuda_available": true, "torch_cuda_version": "13.0"}\\n' if command[1] == '-c' else '', 'stderr_tail': '', 'ok': True}
    install = setup._install_prepare_dependencies(root)
    win_runtime = {'os':'windows','arch':'x64','python_tag':'cp312','accelerator_lane':'cuda124'}
    print(json.dumps({
        'install_status': install['status'],
        'venv_python': install['venv_python'],
        'first_command_python': calls[0]['command'][0],
        'win_native_compatible': _wheel_is_compatible('native-1.0.0-cp312-cp312-win_amd64.whl', win_runtime),
        'win_x86_incompatible': _wheel_is_compatible('native-1.0.0-cp312-cp312-win32.whl', win_runtime),
        'linux_lane': runtime_lane_id({'os':'linux','arch':'x64','python_tag':'cp312','accelerator_lane':'cuda124'}),
        'readiness_supported': sorted(readiness.SUPPORTED_RUNTIME_LANES),
        'runtime_supported': sorted(runtime.SUPPORTED_RUNTIME_LANES),
    }, sort_keys=True))
`)

  assert.equal(result.install_status, 'installed')
  assert.match(result.venv_python, /venv[\\/]Scripts[\\/]python\.exe$/)
  assert.equal(result.first_command_python, result.venv_python)
  assert.equal(result.win_native_compatible, true)
  assert.equal(result.win_x86_incompatible, false)
  assert.equal(result.linux_lane, 'linux-x64-cp312-cuda124')
  assert.deepEqual(result.readiness_supported, ['linux-aarch64-cp312-cuda124', 'linux-x64-cp312-cuda124', 'windows-x64-cp311-cuda124', 'windows-x64-cp312-cuda124'])
  assert.deepEqual(result.runtime_supported, ['linux-aarch64-cp312-cuda124', 'linux-x64-cp312-cuda124', 'windows-x64-cp311-cuda124', 'windows-x64-cp312-cuda124'])
})

test('logical storage paths accept Windows separators without allowing escapes', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
from pixal3d_extension.paths import require_safe_relative_path, resolve_modly_layout, resolve_storage_path

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / 'Modly' / 'extensions' / 'pixal3d'
    root.mkdir(parents=True)
    layout = resolve_modly_layout(root)
    normalized = require_safe_relative_path('models\\\\pixal3d\\\\generate\\\\pipeline.json')
    resolved = resolve_storage_path(layout, 'models\\\\pixal3d\\\\generate\\\\pipeline.json')
    errors = []
    for value in ['..\\\\escape', 'C:\\\\tmp\\\\x', '.hidden\\\\x']:
        try:
            require_safe_relative_path(value)
        except ValueError:
            errors.append(value)
    print(json.dumps({'normalized': normalized, 'resolved_suffix': str(resolved).replace('\\\\', '/').split('/Modly/')[-1], 'errors': errors}, sort_keys=True))
`)

  assert.equal(result.normalized, 'models/pixal3d/generate/pipeline.json')
  assert.equal(result.resolved_suffix, 'models/pixal3d/generate/pipeline.json')
  assert.deepEqual(result.errors, ['..\\escape', 'C:\\tmp\\x', '.hidden\\x'])
})

test('Modly home derivation recognizes model and workspace logical paths without hardcoded roots', () => {
  const result = runPython(`
import json
from pixal3d_extension.paths import derive_modly_home, derive_modly_home_from_model_dir, derive_modly_home_from_workspace_dir

def normalized(path):
    return None if path is None else str(path).replace('\\\\', '/')

print(json.dumps({
    'win_model': normalized(derive_modly_home_from_model_dir(r'C:\\Software\\llm\\models\\pixal3d\\generate')),
    'posix_model': normalized(derive_modly_home_from_model_dir('/opt/modly/models/pixal3d/generate')),
    'win_workspace': normalized(derive_modly_home_from_workspace_dir(r'C:\\Software\\llm\\workspace')),
    'win_workflows': normalized(derive_modly_home_from_workspace_dir(r'C:\\Software\\llm\\workspace\\Workflows')),
    'posix_workspace': normalized(derive_modly_home_from_workspace_dir('/opt/modly/workspace')),
    'posix_workflows': normalized(derive_modly_home_from_workspace_dir('/opt/modly/workspace/Workflows')),
    'prefer_model': normalized(derive_modly_home(
        model_dir=r'C:\\Software\\llm\\models\\pixal3d\\generate',
        workspace_dir=r'D:\\Wrong\\workspace\\Workflows',
    )),
    'unknown': normalized(derive_modly_home(model_dir='TencentARC/Pixal3D', workspace_dir='/tmp/not-a-modly-path')),
}, sort_keys=True))
`)

  assert.deepEqual(result, {
    posix_model: '/opt/modly',
    posix_workspace: '/opt/modly',
    posix_workflows: '/opt/modly',
    prefer_model: 'C:/Software/llm',
    unknown: null,
    win_model: 'C:/Software/llm',
    win_workspace: 'C:/Software/llm',
    win_workflows: 'C:/Software/llm',
  })
})

test('Pixal3DGenerator generated jobs include workspace_root derived from model_dir', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
from generator import Pixal3DGenerator
from pixal3d_extension import runtime

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / 'Modly'
    model_dir = root / 'models' / 'pixal3d' / 'generate'
    model_dir.mkdir(parents=True)
    output_dir = root / 'workspace' / 'Workflows'
    output_dir.mkdir(parents=True)
    captured = {}

    def fake_run_job(job, *, pipeline_factory=None):
        del pipeline_factory
        captured['job'] = dict(job)
        captured['input_exists_during_run'] = Path(job['input_image']).is_file()
        glb = Path(job['output_dir']) / 'generated-job.glb'
        glb.write_bytes(b'glb')
        return {'status': 'completed', 'output': {'glb_path': str(glb)}}

    original_run_job = runtime.run_job
    runtime.run_job = fake_run_job
    try:
        returned = Pixal3DGenerator(model_dir=model_dir, workspace_dir=output_dir).generate(b'png-bytes', params={'seed': 7})
    finally:
        runtime.run_job = original_run_job

    job = captured['job']
    print(json.dumps({
        'returned_name': returned.name,
        'workspace_root_matches': Path(job['workspace_root']) == root,
        'model_source_matches': Path(job['model_source']) == model_dir,
        'output_dir_matches': Path(job['output_dir']) == output_dir,
        'input_parent_matches_output': Path(job['input_image']).parent == output_dir,
        'input_exists_during_run': captured['input_exists_during_run'],
        'seed': job['params']['seed'],
    }, sort_keys=True))
`)

  assert.deepEqual(result, {
    input_exists_during_run: true,
    input_parent_matches_output: true,
    model_source_matches: true,
    output_dir_matches: true,
    returned_name: 'generated-job.glb',
    seed: 7,
    workspace_root_matches: true,
  })
})

test('Pixal3DGenerator load patches with derived Modly root instead of workspace dir', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
from generator import Pixal3DGenerator
from pixal3d_extension import pipeline_patch

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / 'Modly'
    model_dir = root / 'models' / 'pixal3d' / 'generate'
    workspace_dir = root / 'workspace' / 'Workflows'
    model_dir.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
    calls = []

    def fake_patch_pipeline(workspace_root, **kwargs):
        calls.append({'workspace_root': str(workspace_root), 'kwargs': kwargs})
        return {'status': 'patched', 'code': 'fake'}

    original_patch_pipeline = pipeline_patch.patch_pipeline
    pipeline_patch.patch_pipeline = fake_patch_pipeline
    try:
        Pixal3DGenerator(model_dir=model_dir, workspace_dir=workspace_dir).load()
    finally:
        pipeline_patch.patch_pipeline = original_patch_pipeline

    print(json.dumps({
        'call_count': len(calls),
        'called_with_root': Path(calls[0]['workspace_root']) == root,
        'called_with_workspace_dir': Path(calls[0]['workspace_root']) == workspace_dir,
        'auxiliary_mode': calls[0]['kwargs']['auxiliary_mode'],
        'network_available': calls[0]['kwargs']['network_available'],
    }, sort_keys=True))
`)

  assert.deepEqual(result, {
    auxiliary_mode: 'default',
    call_count: 1,
    called_with_root: true,
    called_with_workspace_dir: false,
    network_available: true,
  })
})

test('setup reports model path conflicts as JSON failures before installing dependencies', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup

class FakeEnvBuilder:
    def __init__(self, *args, **kwargs):
        pass
    def create(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'pyvenv.cfg').write_text('', encoding='utf-8')

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    conflict = Path(tmp) / 'Modly' / 'data' / 'models' / 'pixal3d' / 'auxiliary'
    conflict.parent.mkdir(parents=True)
    conflict.write_text('not a directory', encoding='utf-8')
    setup.stdlib_venv.EnvBuilder = FakeEnvBuilder
    result = setup.run_setup(['--prepare', '--skip-install', '--workspace-root', str(ext_dir)])
    print(json.dumps({
        'status': result['status'],
        'failure_code': result.get('failure_code'),
        'conflict_logical_path': result.get('path_conflict', {}).get('logical_path'),
        'conflict_path': result.get('path_conflict', {}).get('path'),
        'installs_started': result.get('installs_started'),
        'downloads_started': result.get('downloads_started'),
    }, sort_keys=True))
`)

  assert.equal(result.status, 'failed')
  assert.equal(result.failure_code, 'setup_path_conflict')
  assert.equal(result.conflict_logical_path, 'models/pixal3d/auxiliary/dinov3')
  assert.match(result.conflict_path.replaceAll('\\\\', '/'), /models\/pixal3d\/auxiliary$/)
  assert.equal(result.installs_started, false)
  assert.equal(result.downloads_started, false)
})

test('Pixal3D logical model paths avoid Windows reserved device names', () => {
  const result = runPython(`
import json
from pathlib import PurePosixPath
import setup
from pixal3d_extension.assets import AUXILIARY_ASSETS, REQUIRED_SENTINEL_PATHS
from pixal3d_extension.paths import require_safe_relative_path
from pixal3d_extension.readiness import SETUP_REQUIRED_PATHS

paths = [*setup.RUNTIME_DIRS, *SETUP_REQUIRED_PATHS, *REQUIRED_SENTINEL_PATHS, *(manifest.local_root for manifest in AUXILIARY_ASSETS.values())]
reserved_errors = []
for value in ['models/pixal3d/aux/dinov3', 'models/pixal3d/CON/file.txt', 'models/pixal3d/nul.txt']:
    try:
        require_safe_relative_path(value)
    except ValueError:
        reserved_errors.append(value)

print(json.dumps({
    'paths': paths,
    'reserved_errors': reserved_errors,
    'reserved_segments': sorted({part for path in paths for part in PurePosixPath(path).parts if part.upper() in {'AUX', 'CON', 'PRN', 'NUL'}}),
    'has_auxiliary': any('models/pixal3d/auxiliary/' in path for path in paths),
}, sort_keys=True))
`)

  assert.deepEqual(result.reserved_segments, [])
  assert.equal(result.has_auxiliary, true)
  assert.deepEqual(result.reserved_errors, ['models/pixal3d/aux/dinov3', 'models/pixal3d/CON/file.txt', 'models/pixal3d/nul.txt'])
  assert.ok(result.paths.every((value) => !value.includes('models/pixal3d/aux/')))
})

test('setup prepare bootstraps auxiliary model directories under auxiliary, not stale aux', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup

class FakeEnvBuilder:
    def __init__(self, *args, **kwargs):
        pass
    def create(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'pyvenv.cfg').write_text('', encoding='utf-8')

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    setup.stdlib_venv.EnvBuilder = FakeEnvBuilder
    result = setup.run_setup(['--prepare', '--skip-install', '--workspace-root', str(ext_dir), '--json'])
    models_root = Path(tmp) / 'Modly' / 'data' / 'models' / 'pixal3d'
    print(json.dumps({
        'status': result['status'],
        'downloads_started': result['downloads_started'],
        'installs_started': result['installs_started'],
        'dinov3_dir': (models_root / 'auxiliary' / 'dinov3').is_dir(),
        'rmbg_dir': (models_root / 'auxiliary' / 'rmbg').is_dir(),
        'moge_dir': (models_root / 'auxiliary' / 'moge').is_dir(),
        'naf_dir': (models_root / 'auxiliary' / 'naf').is_dir(),
        'legacy_aux_dir': (models_root / 'aux').exists(),
        'auxiliary_roots': result['auxiliary_assets']['logical_roots'],
        'created_paths': [entry['path'] for entry in result['created']],
    }, sort_keys=True))
`)

  assert.equal(result.status, 'prepared')
  assert.equal(result.downloads_started, false)
  assert.equal(result.installs_started, false)
  assert.equal(result.dinov3_dir, true)
  assert.equal(result.rmbg_dir, true)
  assert.equal(result.moge_dir, true)
  assert.equal(result.naf_dir, true)
  assert.equal(result.legacy_aux_dir, false)
  assert.deepEqual(result.auxiliary_roots, {
    dino: 'models/pixal3d/auxiliary/dinov3',
    moge: 'models/pixal3d/auxiliary/moge',
    naf: 'models/pixal3d/auxiliary/naf',
    rmbg: 'models/pixal3d/auxiliary/rmbg',
  })
  assert.ok(result.created_paths.includes('models/pixal3d/auxiliary/dinov3'))
  assert.ok(result.created_paths.includes('models/pixal3d/auxiliary/rmbg'))
  assert.ok(result.created_paths.includes('models/pixal3d/auxiliary/moge'))
  assert.ok(result.created_paths.includes('models/pixal3d/auxiliary/naf'))
})

test('normal setup prepare does not perform hidden auxiliary downloads', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup

class FakeEnvBuilder:
    def __init__(self, *args, **kwargs):
        pass
    def create(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'pyvenv.cfg').write_text('', encoding='utf-8')

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    calls = []
    def forbidden_bootstrap(*_args, **_kwargs):
        calls.append('bootstrap')
        raise AssertionError('normal setup must not bootstrap auxiliary weights')
    setup.stdlib_venv.EnvBuilder = FakeEnvBuilder
    setup.bootstrap_auxiliary_assets = forbidden_bootstrap
    prepared = setup.run_setup(['--prepare', '--skip-install', '--workspace-root', str(ext_dir), '--json'])
    print(json.dumps({
        'status': prepared['status'],
        'downloads_started': prepared['downloads_started'],
        'installs_started': prepared['installs_started'],
        'bootstrap_calls': calls,
    }, sort_keys=True))
`)

  assert.equal(result.status, 'prepared')
  assert.equal(result.downloads_started, false)
  assert.equal(result.installs_started, false)
  assert.deepEqual(result.bootstrap_calls, [])
})

test('setup explicit auxiliary bootstrap flag triggers mocked bootstrap and download plan documents intent', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    calls = []
    def fake_bootstrap(workspace_root, downloader=None, force=False, **_kwargs):
        calls.append({'workspace_root': str(workspace_root), 'downloader_is_none': downloader is None, 'force': force})
        return {
            'status': 'ready',
            'code': 'auxiliary_assets_bootstrapped',
            'downloads_started': True,
            'installs_started': False,
            'missing': [],
            'generation_allowed': True,
        }
    setup.bootstrap_auxiliary_assets = fake_bootstrap
    bootstrapped = setup.run_setup(['--bootstrap-auxiliary-assets', '--force-auxiliary-assets', '--workspace-root', str(ext_dir), '--json'])
    plan = setup.run_setup(['--download-plan', '--workspace-root', str(ext_dir), '--json'])
    print(json.dumps({
        'status': bootstrapped['status'],
        'code': bootstrapped['code'],
        'downloads_started': bootstrapped['downloads_started'],
        'installs_started': bootstrapped['installs_started'],
        'calls': calls,
        'plan_status': plan['status'],
        'bootstrap_command': plan['auxiliary']['bootstrap_command'],
        'bootstrap_intent': plan['auxiliary']['bootstrap_intent'],
        'plan_note': plan['note'],
    }, sort_keys=True))
`)

  assert.equal(result.status, 'bootstrap_auxiliary_assets')
  assert.equal(result.code, 'auxiliary_assets_bootstrapped')
  assert.equal(result.downloads_started, true)
  assert.equal(result.installs_started, false)
  assert.equal(result.calls.length, 1)
  assert.equal(result.calls[0].force, true)
  assert.match(result.bootstrap_command, /--bootstrap-auxiliary-assets/)
  assert.match(result.bootstrap_intent, /allowlisted DINO\/RMBG\/MoGe\/NAF/)
  assert.match(result.plan_note, /--bootstrap-auxiliary-assets/)
  assert.equal(result.plan_status, 'download_plan')
})

test('asset validator reports missing and complete auxiliary sentinels', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
from pixal3d_extension.assets import AUXILIARY_ASSETS, check_auxiliary_sentinels
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    layout = resolve_modly_layout(ext_dir)
    missing = check_auxiliary_sentinels(ext_dir)
    for manifest in AUXILIARY_ASSETS.values():
        for sentinel in manifest.sentinel_paths:
            path = resolve_storage_path(layout, sentinel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('sentinel', encoding='utf-8')
    complete = check_auxiliary_sentinels(ext_dir)
    print(json.dumps({
        'missing_code': missing['code'],
        'missing_count': len(missing['missing']),
        'missing_sample': missing['missing'][:2],
        'complete_code': complete['code'],
        'complete_missing': complete['missing'],
        'assets_complete': {key: value['complete'] for key, value in complete['assets'].items()},
    }, sort_keys=True))
`)

  assert.equal(result.missing_code, 'missing_auxiliary_assets')
  assert.equal(result.missing_count, 10)
  assert.ok(result.missing_sample.some((value) => value.includes('models/pixal3d/auxiliary/dinov3/')))
  assert.equal(result.complete_code, 'auxiliary_assets_ready')
  assert.deepEqual(result.complete_missing, [])
  assert.deepEqual(result.assets_complete, { dino: true, moge: true, naf: true, rmbg: true })
})

test('auxiliary bootstrap downloads only the exact DINO/RMBG/MoGe/NAF allowlist with a mocked downloader', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
from pixal3d_extension.assets import bootstrap_auxiliary_assets, check_auxiliary_sentinels

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    calls = []
    def downloader(*, repo_id, filename, destination, source_kind='hf_repo', url=None, **_kwargs):
        calls.append({'repo_id': repo_id, 'filename': filename, 'source_kind': source_kind, 'url': url})
        if source_kind == 'url' and (filename != 'naf_release.pth' or url != 'https://github.com/valeoai/NAF/releases/download/model/naf_release.pth'):
            raise AssertionError('URL assets must be restricted to the allowlisted NAF checkpoint')
        Path(destination).write_text(f'{repo_id}/{filename}', encoding='utf-8')
        return str(destination)
    bootstrap = bootstrap_auxiliary_assets(ext_dir, downloader=downloader)
    auxiliary_root = Path(tmp) / 'Modly' / 'data' / 'models' / 'pixal3d' / 'auxiliary'
    files = sorted(str(path.relative_to(auxiliary_root)).replace('\\\\', '/') for path in auxiliary_root.rglob('*') if path.is_file())
    sentinels = check_auxiliary_sentinels(ext_dir)
    print(json.dumps({
        'status': bootstrap['status'],
        'code': bootstrap['code'],
        'downloads_started': bootstrap['downloads_started'],
        'allowlist_files': sum(len(entry['files']) for entry in bootstrap['allowlist'].values()),
        'naf_allowlist': bootstrap['allowlist']['naf'],
        'calls': calls,
        'files': files,
        'sentinel_code': sentinels['code'],
        'missing': sentinels['missing'],
    }, sort_keys=True))
`)

  const expectedCalls = [
    { repo_id: 'camenduru/dinov3-vitl16-pretrain-lvd1689m', filename: 'config.json', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/dinov3-vitl16-pretrain-lvd1689m', filename: 'preprocessor_config.json', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/dinov3-vitl16-pretrain-lvd1689m', filename: 'model.safetensors', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/RMBG-2.0', filename: 'config.json', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/RMBG-2.0', filename: 'preprocessor_config.json', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/RMBG-2.0', filename: 'BiRefNet_config.py', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/RMBG-2.0', filename: 'birefnet.py', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/RMBG-2.0', filename: 'model.safetensors', source_kind: 'hf_repo', url: null },
    { repo_id: 'Ruicheng/moge-2-vitl', filename: 'model.pt', source_kind: 'hf_repo', url: null },
    { repo_id: 'https://github.com/valeoai/NAF/releases/download/model/naf_release.pth', filename: 'naf_release.pth', source_kind: 'url', url: 'https://github.com/valeoai/NAF/releases/download/model/naf_release.pth' },
  ]
  assert.equal(result.status, 'ready')
  assert.equal(result.code, 'auxiliary_assets_bootstrapped')
  assert.equal(result.downloads_started, true)
  assert.equal(result.allowlist_files, 10)
  assert.deepEqual(result.naf_allowlist, {
    files: ['naf_release.pth'],
    local_root: 'models/pixal3d/auxiliary/naf',
    source: 'https://github.com/valeoai/NAF/releases/download/model/naf_release.pth',
    source_kind: 'url',
    url: 'https://github.com/valeoai/NAF/releases/download/model/naf_release.pth',
  })
  assert.equal(result.calls.length, 10)
  assert.deepEqual(result.calls, expectedCalls)
  assert.deepEqual(result.files, [
    'dinov3/config.json',
    'dinov3/model.safetensors',
    'dinov3/preprocessor_config.json',
    'moge/model.pt',
    'naf/naf_release.pth',
    'rmbg/BiRefNet_config.py',
    'rmbg/birefnet.py',
    'rmbg/config.json',
    'rmbg/model.safetensors',
    'rmbg/preprocessor_config.json',
  ])
  assert.equal(result.sentinel_code, 'auxiliary_assets_ready')
  assert.deepEqual(result.missing, [])
})

test('auxiliary bootstrap validates sentinels and does not promote staged partial files on failure', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
from pixal3d_extension.assets import bootstrap_auxiliary_assets, check_auxiliary_sentinels

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    calls = []
    def downloader(*, repo_id, filename, destination, source_kind='hf_repo', **_kwargs):
        calls.append({'repo_id': repo_id, 'filename': filename, 'source_kind': source_kind})
        Path(destination).write_text('partial', encoding='utf-8')
        if source_kind == 'url' and filename == 'naf_release.pth':
            raise RuntimeError('simulated transfer failure')
        return str(destination)
    bootstrap = bootstrap_auxiliary_assets(ext_dir, downloader=downloader)
    auxiliary_root = Path(tmp) / 'Modly' / 'data' / 'models' / 'pixal3d' / 'auxiliary'
    final_files = sorted(str(path.relative_to(auxiliary_root)).replace('\\\\', '/') for path in auxiliary_root.rglob('*') if path.is_file()) if auxiliary_root.exists() else []
    staging_left = any(path.name.startswith('.bootstrap-') for path in auxiliary_root.rglob('*')) if auxiliary_root.exists() else False
    sentinels = check_auxiliary_sentinels(ext_dir)
    print(json.dumps({
        'bootstrap_status': bootstrap['status'],
        'bootstrap_code': bootstrap['code'],
        'downloads_started': bootstrap['downloads_started'],
        'call_count': len(calls),
        'final_files': final_files,
        'staging_left': staging_left,
        'sentinel_code': sentinels['code'],
        'missing_count': len(sentinels['missing']),
    }, sort_keys=True))
`)

  assert.equal(result.bootstrap_status, 'failed')
  assert.equal(result.bootstrap_code, 'auxiliary_bootstrap_failed')
  assert.equal(result.downloads_started, true)
  assert.equal(result.call_count, 10)
  assert.deepEqual(result.final_files, [])
  assert.equal(result.staging_left, false)
  assert.equal(result.sentinel_code, 'missing_auxiliary_assets')
  assert.equal(result.missing_count, 10)
})

test('auxiliary URL downloader rejects non-allowlisted URL filenames before network access', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import pixal3d_extension.assets as assets

with tempfile.TemporaryDirectory() as tmp:
    calls = []
    original_urlopen = assets.urllib.request.urlopen
    def forbidden_urlopen(*_args, **_kwargs):
        calls.append('network')
        raise AssertionError('mismatched URL basename must fail before network access')
    assets.urllib.request.urlopen = forbidden_urlopen
    try:
        try:
            assets._default_auxiliary_downloader(
                repo_id='https://github.com/valeoai/NAF/releases/download/model/not_allowlisted.pth',
                filename='naf_release.pth',
                destination=Path(tmp) / 'naf_release.pth',
                source_kind='url',
                url='https://github.com/valeoai/NAF/releases/download/model/not_allowlisted.pth',
            )
        except RuntimeError as exc:
            error = str(exc)
    finally:
        assets.urllib.request.urlopen = original_urlopen
    print(json.dumps({'error': error, 'network_calls': calls}, sort_keys=True))
`)

  assert.match(result.error, /URL basename is not allowlisted/)
  assert.deepEqual(result.network_calls, [])
})

test('auxiliary HF downloader falls back to direct allowlisted Hugging Face URL when hub is unavailable', () => {
  const result = runPython(`
import builtins, io, json, tempfile
from pathlib import Path
import pixal3d_extension.assets as assets

class FakeResponse:
    def __init__(self, payload):
        self.stream = io.BytesIO(payload)
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        self.stream.close()
    def read(self, size=-1):
        return self.stream.read(size)

with tempfile.TemporaryDirectory() as tmp:
    calls = []
    original_import = builtins.__import__
    original_urlopen = assets.urllib.request.urlopen
    def fake_import(name, *args, **kwargs):
        if name == 'huggingface_hub':
            raise ModuleNotFoundError("No module named 'huggingface_hub'")
        return original_import(name, *args, **kwargs)
    def fake_urlopen(request):
        calls.append({
            'url': request.full_url,
            'authorization_is_bearer': request.get_header('Authorization') == 'Bearer hf_redacted',
            'token_in_url': 'hf_redacted' in request.full_url,
        })
        return FakeResponse(b'dino-config')
    builtins.__import__ = fake_import
    assets.urllib.request.urlopen = fake_urlopen
    try:
        destination = Path(tmp) / 'stage' / 'config.json'
        downloaded = assets._default_auxiliary_downloader(
            repo_id='camenduru/dinov3-vitl16-pretrain-lvd1689m',
            filename='config.json',
            destination=destination,
            token='hf_redacted',
        )
    finally:
        builtins.__import__ = original_import
        assets.urllib.request.urlopen = original_urlopen
    print(json.dumps({
        'downloaded': str(downloaded),
        'destination_exists': destination.is_file(),
        'destination_text': destination.read_text(encoding='utf-8'),
        'calls': calls,
    }, sort_keys=True))
`)

  assert.equal(result.destination_exists, true)
  assert.equal(result.destination_text, 'dino-config')
  assert.match(result.downloaded.replaceAll('\\', '/'), /stage\/config\.json$/)
  assert.deepEqual(result.calls, [
    {
      authorization_is_bearer: true,
      token_in_url: false,
      url: 'https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m/resolve/main/config.json',
    },
  ])
})

test('auxiliary HF direct fallback encodes explicit revisions in resolve URLs', () => {
  const result = runPython(`
import builtins, io, json, tempfile
from pathlib import Path
import pixal3d_extension.assets as assets

class FakeResponse:
    def __init__(self, payload):
        self.stream = io.BytesIO(payload)
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        self.stream.close()
    def read(self, size=-1):
        return self.stream.read(size)

with tempfile.TemporaryDirectory() as tmp:
    calls = []
    original_import = builtins.__import__
    original_urlopen = assets.urllib.request.urlopen
    def fake_import(name, *args, **kwargs):
        if name == 'huggingface_hub':
            raise ModuleNotFoundError("No module named 'huggingface_hub'")
        return original_import(name, *args, **kwargs)
    def fake_urlopen(request):
        calls.append(request.full_url)
        return FakeResponse(b'moge')
    builtins.__import__ = fake_import
    assets.urllib.request.urlopen = fake_urlopen
    try:
        destination = Path(tmp) / 'model.pt'
        assets._default_auxiliary_downloader(
            repo_id='Ruicheng/moge-2-vitl',
            filename='model.pt',
            destination=destination,
            revision='refs/pr/42',
        )
    finally:
        builtins.__import__ = original_import
        assets.urllib.request.urlopen = original_urlopen
    print(json.dumps({'calls': calls, 'destination_text': destination.read_text(encoding='utf-8')}, sort_keys=True))
`)

  assert.deepEqual(result.calls, [
    'https://huggingface.co/Ruicheng/moge-2-vitl/resolve/refs%2Fpr%2F42/model.pt',
  ])
  assert.equal(result.destination_text, 'moge')
})

test('auxiliary HF direct fallback rejects non-allowlisted repo and filenames before network access', () => {
  const result = runPython(`
import builtins, json, tempfile
from pathlib import Path
import pixal3d_extension.assets as assets

with tempfile.TemporaryDirectory() as tmp:
    network_calls = []
    errors = []
    original_import = builtins.__import__
    original_urlopen = assets.urllib.request.urlopen
    def fake_import(name, *args, **kwargs):
        if name == 'huggingface_hub':
            raise ModuleNotFoundError("No module named 'huggingface_hub'")
        return original_import(name, *args, **kwargs)
    def forbidden_urlopen(*_args, **_kwargs):
        network_calls.append('network')
        raise AssertionError('non-allowlisted HF assets must fail before network access')
    builtins.__import__ = fake_import
    assets.urllib.request.urlopen = forbidden_urlopen
    try:
        cases = [
            {'repo_id': 'not-allowed/repo', 'filename': 'config.json'},
            {'repo_id': 'camenduru/dinov3-vitl16-pretrain-lvd1689m', 'filename': 'not_allowlisted.bin'},
            {'repo_id': 'https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m', 'filename': 'config.json'},
        ]
        for index, case in enumerate(cases):
            try:
                assets._default_auxiliary_downloader(
                    repo_id=case['repo_id'],
                    filename=case['filename'],
                    destination=Path(tmp) / f'case-{index}',
                )
            except RuntimeError as exc:
                errors.append(str(exc))
    finally:
        builtins.__import__ = original_import
        assets.urllib.request.urlopen = original_urlopen
    print(json.dumps({'errors': errors, 'network_calls': network_calls}, sort_keys=True))
`)

  assert.equal(result.errors.length, 3)
  assert.ok(result.errors.every((error) => /allowlisted HF auxiliary asset/.test(error)))
  assert.deepEqual(result.network_calls, [])
})

test('auxiliary URL downloader allows only the exact NAF release URL', () => {
  const result = runPython(`
import io, json, tempfile
from pathlib import Path
import pixal3d_extension.assets as assets

class FakeResponse:
    def __init__(self, payload):
        self.stream = io.BytesIO(payload)
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        self.stream.close()
    def read(self, size=-1):
        return self.stream.read(size)

with tempfile.TemporaryDirectory() as tmp:
    calls = []
    original_urlopen = assets.urllib.request.urlopen
    def fake_urlopen(request):
        calls.append(request.full_url)
        return FakeResponse(b'naf')
    assets.urllib.request.urlopen = fake_urlopen
    try:
        destination = Path(tmp) / 'naf_release.pth'
        assets._default_auxiliary_downloader(
            repo_id='https://github.com/valeoai/NAF/releases/download/model/naf_release.pth',
            filename='naf_release.pth',
            destination=destination,
            source_kind='url',
            url='https://github.com/valeoai/NAF/releases/download/model/naf_release.pth',
        )
    finally:
        assets.urllib.request.urlopen = original_urlopen
    print(json.dumps({'calls': calls, 'destination_text': destination.read_text(encoding='utf-8')}, sort_keys=True))
`)

  assert.deepEqual(result.calls, ['https://github.com/valeoai/NAF/releases/download/model/naf_release.pth'])
  assert.equal(result.destination_text, 'naf')
})

test('pipeline patch writes local DINO/RMBG paths and non-absolute metadata when auxiliary sentinels are complete', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
from pixal3d_extension.assets import AUXILIARY_ASSETS
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
from pixal3d_extension.pipeline_patch import patch_pipeline

PIPELINE = {
    'args': {
        'image_cond_model': {'args': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'}},
        'rembg_model': {'args': {'model_name': 'briaai/RMBG-2.0'}},
    }
}

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    layout = resolve_modly_layout(ext_dir)
    pipeline_path = resolve_storage_path(layout, 'models/pixal3d/generate/pipeline.json')
    pipeline_path.parent.mkdir(parents=True)
    pipeline_path.write_text(json.dumps(PIPELINE), encoding='utf-8')
    for manifest in AUXILIARY_ASSETS.values():
        for sentinel in manifest.sentinel_paths:
            path = resolve_storage_path(layout, sentinel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('sentinel', encoding='utf-8')

    patched = patch_pipeline(ext_dir, auxiliary_mode='local', network_available=False)
    data = json.loads(pipeline_path.read_text(encoding='utf-8'))
    metadata_path = resolve_storage_path(layout, 'models/pixal3d/readiness.json')
    metadata_text = metadata_path.read_text(encoding='utf-8')
    metadata = json.loads(metadata_text)['pipeline_patch']
    dino = data['args']['image_cond_model']['args']['model_name']
    rmbg = data['args']['rembg_model']['args']['model_name']
    print(json.dumps({
        'patch_code': patched['code'],
        'dino_is_absolute': Path(dino).is_absolute(),
        'rmbg_is_absolute': Path(rmbg).is_absolute(),
        'dino_suffix': dino.replace('\\\\', '/').split('/Modly/data/')[-1],
        'rmbg_suffix': rmbg.replace('\\\\', '/').split('/Modly/data/')[-1],
        'replacement_kinds': metadata['replacement_kinds'],
        'replacement_refs': metadata['replacement_refs'],
        'auxiliary_kinds': metadata['auxiliary_kinds'],
        'auxiliary_refs': metadata['auxiliary_refs'],
        'metadata_contains_tmp': str(Path(tmp)) in metadata_text,
        'unlocalized_keys': [entry['key'] for entry in metadata['unlocalized_runtime_dependencies']],
        'localizable_keys': [entry['key'] for entry in metadata['localizable_runtime_dependencies']],
    }, sort_keys=True))
`)

  assert.equal(result.patch_code, 'pipeline_patch_applied')
  assert.equal(result.dino_is_absolute, true)
  assert.equal(result.rmbg_is_absolute, true)
  assert.equal(result.dino_suffix, 'models/pixal3d/auxiliary/dinov3')
  assert.equal(result.rmbg_suffix, 'models/pixal3d/auxiliary/rmbg')
  assert.deepEqual(result.replacement_kinds, { dino: 'local', rmbg: 'local' })
  assert.deepEqual(result.replacement_refs, {
    dino: 'local:models/pixal3d/auxiliary/dinov3',
    rmbg: 'local:models/pixal3d/auxiliary/rmbg',
  })
  assert.deepEqual(result.auxiliary_kinds, { dino: 'local', moge: 'local', naf: 'local', rmbg: 'local' })
  assert.equal(result.auxiliary_refs.moge, 'local:models/pixal3d/auxiliary/moge/model.pt')
  assert.equal(result.auxiliary_refs.naf, 'local:models/pixal3d/auxiliary/naf/naf_release.pth')
  assert.equal(result.metadata_contains_tmp, false)
  assert.deepEqual(result.unlocalized_keys, [])
  assert.deepEqual(result.localizable_keys, ['moge', 'naf'])
})

test('default pipeline patch preserves remote/HF-cache fallback when local auxiliary assets are missing', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
from pixal3d_extension.assets import PRIMARY_ASSET
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
from pixal3d_extension.pipeline_patch import patch_pipeline
from pixal3d_extension.readiness import check_readiness

PIPELINE = {
    'args': {
        'image_cond_model': {'args': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'}},
        'rembg_model': {'args': {'model_name': 'briaai/RMBG-2.0'}},
    }
}

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    layout = resolve_modly_layout(ext_dir)
    for sentinel in PRIMARY_ASSET.sentinel_paths:
        path = resolve_storage_path(layout, sentinel)
        path.parent.mkdir(parents=True, exist_ok=True)
        if sentinel.endswith('pipeline.json'):
            path.write_text(json.dumps(PIPELINE), encoding='utf-8')
        else:
            path.write_text('primary', encoding='utf-8')

    patched = patch_pipeline(ext_dir, auxiliary_mode='default', network_available=True)
    pipeline = json.loads(resolve_storage_path(layout, 'models/pixal3d/generate/pipeline.json').read_text(encoding='utf-8'))
    readiness = check_readiness(ext_dir, auxiliary_mode='default', network_available=True, runtime_validated=True)
    print(json.dumps({
        'patch_code': patched['code'],
        'patch_aux_code': patched['auxiliary_source']['code'],
        'replacement_kinds': patched['auxiliary_source']['sources'],
        'dino': pipeline['args']['image_cond_model']['args']['model_name'],
        'rmbg': pipeline['args']['rembg_model']['args']['model_name'],
        'moge': patched['auxiliary_source']['sources']['moge']['value'],
        'naf': patched['auxiliary_source']['sources']['naf']['value'],
        'readiness_code': readiness['code'],
        'readiness_generation_allowed': readiness['generation_allowed'],
        'readiness_aux_code': readiness['auxiliary_source']['code'],
        'local_aux_exists': resolve_storage_path(layout, 'models/pixal3d/auxiliary/dinov3').exists(),
        'local_moge_exists': resolve_storage_path(layout, 'models/pixal3d/auxiliary/moge/model.pt').exists(),
        'local_naf_exists': resolve_storage_path(layout, 'models/pixal3d/auxiliary/naf/naf_release.pth').exists(),
    }, sort_keys=True))
`)

  assert.equal(result.patch_code, 'pipeline_patch_applied')
  assert.equal(result.patch_aux_code, 'remote_auxiliary_fallback')
  assert.equal(result.dino, 'camenduru/dinov3-vitl16-pretrain-lvd1689m')
  assert.equal(result.rmbg, 'camenduru/RMBG-2.0')
  assert.equal(result.moge, 'Ruicheng/moge-2-vitl')
  assert.equal(result.naf, 'https://github.com/valeoai/NAF/releases/download/model/naf_release.pth')
  assert.equal(result.readiness_code, 'ready')
  assert.equal(result.readiness_generation_allowed, true)
  assert.equal(result.readiness_aux_code, 'remote_auxiliary_fallback')
  assert.equal(result.local_aux_exists, false)
  assert.equal(result.local_moge_exists, false)
  assert.equal(result.local_naf_exists, false)
})

test('runtime derives Modly root from model_source and patches RMBG before pipeline execution', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
from pixal3d_extension.assets import PRIMARY_ASSET
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
from pixal3d_extension import runtime

PIPELINE = {
    'args': {
        'image_cond_model': {'args': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'}},
        'rembg_model': {'args': {'model_name': 'briaai/RMBG-2.0'}},
    }
}

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / 'Modly'
    workspace_dir = root / 'workspace' / 'Workflows'
    workspace_dir.mkdir(parents=True)
    layout = resolve_modly_layout(root)
    for sentinel in PRIMARY_ASSET.sentinel_paths:
        path = resolve_storage_path(layout, sentinel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(PIPELINE) if sentinel.endswith('pipeline.json') else 'primary', encoding='utf-8')
    pipeline_path = resolve_storage_path(layout, 'models/pixal3d/generate/pipeline.json')
    model_dir = resolve_storage_path(layout, 'models/pixal3d/generate')
    image = workspace_dir / 'input.png'
    image.write_bytes(b'image')
    calls = []
    captured = {}

    def downloader(*, repo_id, filename, destination, source_kind='hf_repo', url=None, **_kwargs):
        calls.append({'repo_id': repo_id, 'filename': filename, 'source_kind': source_kind, 'url': url})
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text(f'{repo_id}/{filename}', encoding='utf-8')
        return str(destination)

    def fake_pipeline(_source):
        def pipeline(**kwargs):
            active = json.loads(pipeline_path.read_text(encoding='utf-8'))
            captured['dino'] = active['args']['image_cond_model']['args']['model_name']
            captured['rmbg'] = active['args']['rembg_model']['args']['model_name']
            glb = Path(kwargs['output_dir']) / 'derived-root-patched.glb'
            glb.write_bytes(b'raw')
            return {'glb_path': str(glb)}
        return pipeline

    result = runtime.run_job({
        'input_image': str(image),
        'output_dir': str(workspace_dir),
        'model_source': str(model_dir),
        'readiness': {'generation_allowed': True, 'code': 'ready'},
        'auxiliary_mode': 'default',
        'network_available': True,
        'auxiliary_bootstrap_downloader': downloader,
        'params': {'seed': 1},
    }, pipeline_factory=fake_pipeline)

    print(json.dumps({
        'status': result['status'],
        'call_count': len(calls),
        'rmbg': captured['rmbg'],
        'rmbg_is_upstream_gated': captured['rmbg'] == 'briaai/RMBG-2.0',
        'rmbg_suffix': captured['rmbg'].replace('\\\\', '/').split('/Modly/')[-1],
        'dino_suffix': captured['dino'].replace('\\\\', '/').split('/Modly/')[-1],
        'source_code': result['auxiliary_source']['code'],
        'source_kinds': {key: value['kind'] for key, value in result['auxiliary_source']['sources'].items()},
    }, sort_keys=True))
`)

  assert.equal(result.status, 'completed')
  assert.equal(result.call_count, 10)
  assert.equal(result.rmbg_is_upstream_gated, false)
  assert.notEqual(result.rmbg, 'briaai/RMBG-2.0')
  assert.equal(result.rmbg_suffix, 'models/pixal3d/auxiliary/rmbg')
  assert.equal(result.dino_suffix, 'models/pixal3d/auxiliary/dinov3')
  assert.equal(result.source_code, 'local_auxiliary_assets_ready')
  assert.deepEqual(result.source_kinds, { dino: 'local', moge: 'local', naf: 'local', rmbg: 'local' })
})

test('runtime default mode attempts auxiliary bootstrap before remote fallback and uses local paths immediately', () => {
  const result = runPython(`
import json, sys, tempfile, types
from pathlib import Path
from pixal3d_extension.assets import PRIMARY_ASSET
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
from pixal3d_extension import runtime

PIPELINE = {
    'args': {
        'image_cond_model': {'args': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'}},
        'rembg_model': {'args': {'model_name': 'briaai/RMBG-2.0'}},
    }
}

class FakeScene:
    def apply_transform(self, matrix):
        pass
    def export(self, file_type):
        return b'rotated'

fake_trimesh = types.ModuleType('trimesh')
fake_trimesh.load = lambda path, file_type, force, process: FakeScene()
fake_trimesh.transformations = types.SimpleNamespace(rotation_matrix=lambda angle, axis: None)
sys.modules['trimesh'] = fake_trimesh

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    layout = resolve_modly_layout(ext_dir)
    for sentinel in PRIMARY_ASSET.sentinel_paths:
        path = resolve_storage_path(layout, sentinel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(PIPELINE) if sentinel.endswith('pipeline.json') else 'primary', encoding='utf-8')
    image = ext_dir / 'input.png'
    image.write_bytes(b'image')
    (ext_dir / 'outputs').mkdir()
    calls = []
    def downloader(*, repo_id, filename, destination, source_kind='hf_repo', url=None, **_kwargs):
        calls.append({'repo_id': repo_id, 'filename': filename, 'source_kind': source_kind, 'url': url})
        Path(destination).write_text(f'{repo_id}/{filename}', encoding='utf-8')
        return str(destination)
    def fake_pipeline(_source):
        def pipeline(**kwargs):
            glb = Path(kwargs['output_dir']) / 'bootstrap-local.glb'
            glb.write_bytes(b'raw')
            return {'glb_path': str(glb)}
        return pipeline
    result = runtime.run_job({
        'workspace_root': str(ext_dir),
        'input_image': 'input.png',
        'output_dir': 'outputs',
        'readiness': {'generation_allowed': True, 'code': 'ready'},
        'auxiliary_mode': 'default',
        'network_available': True,
        'auxiliary_bootstrap_downloader': downloader,
        'params': {},
    }, pipeline_factory=fake_pipeline)
    pipeline = json.loads(resolve_storage_path(layout, 'models/pixal3d/generate/pipeline.json').read_text(encoding='utf-8'))
    dino = pipeline['args']['image_cond_model']['args']['model_name']
    rmbg = pipeline['args']['rembg_model']['args']['model_name']
    moge = result['auxiliary_source']['sources']['moge']['value']
    naf = result['auxiliary_source']['sources']['naf']['value']
    print(json.dumps({
        'status': result['status'],
        'calls': calls,
        'dino_suffix': dino.replace('\\\\', '/').split('/Modly/data/')[-1],
        'rmbg_suffix': rmbg.replace('\\\\', '/').split('/Modly/data/')[-1],
        'moge_suffix': moge.replace('\\\\', '/').split('/Modly/data/')[-1],
        'naf_suffix': naf.replace('\\\\', '/').split('/Modly/data/')[-1],
        'source_code': result['auxiliary_source']['code'],
        'source_kinds': {key: value['kind'] for key, value in result['auxiliary_source']['sources'].items()},
        'bootstrap_code': result['auxiliary_source']['auxiliary_bootstrap']['code'],
        'glb_exists': Path(result['output']['glb_path']).is_file(),
    }, sort_keys=True))
`)

  assert.equal(result.status, 'completed')
  assert.deepEqual(result.calls, [
    { repo_id: 'camenduru/dinov3-vitl16-pretrain-lvd1689m', filename: 'config.json', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/dinov3-vitl16-pretrain-lvd1689m', filename: 'preprocessor_config.json', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/dinov3-vitl16-pretrain-lvd1689m', filename: 'model.safetensors', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/RMBG-2.0', filename: 'config.json', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/RMBG-2.0', filename: 'preprocessor_config.json', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/RMBG-2.0', filename: 'BiRefNet_config.py', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/RMBG-2.0', filename: 'birefnet.py', source_kind: 'hf_repo', url: null },
    { repo_id: 'camenduru/RMBG-2.0', filename: 'model.safetensors', source_kind: 'hf_repo', url: null },
    { repo_id: 'Ruicheng/moge-2-vitl', filename: 'model.pt', source_kind: 'hf_repo', url: null },
    { repo_id: 'https://github.com/valeoai/NAF/releases/download/model/naf_release.pth', filename: 'naf_release.pth', source_kind: 'url', url: 'https://github.com/valeoai/NAF/releases/download/model/naf_release.pth' },
  ])
  assert.equal(result.dino_suffix, 'models/pixal3d/auxiliary/dinov3')
  assert.equal(result.rmbg_suffix, 'models/pixal3d/auxiliary/rmbg')
  assert.equal(result.moge_suffix, 'models/pixal3d/auxiliary/moge/model.pt')
  assert.equal(result.naf_suffix, 'models/pixal3d/auxiliary/naf/naf_release.pth')
  assert.equal(result.source_code, 'local_auxiliary_assets_ready')
  assert.deepEqual(result.source_kinds, { dino: 'local', moge: 'local', naf: 'local', rmbg: 'local' })
  assert.equal(result.bootstrap_code, 'auxiliary_assets_bootstrapped')
  assert.equal(result.glb_exists, true)
})

test('runtime default mode preserves remote/HF/Torch-cache fallback when auxiliary bootstrap fails', () => {
  const result = runPython(`
import json, sys, tempfile, types
from pathlib import Path
from pixal3d_extension.assets import PRIMARY_ASSET
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
from pixal3d_extension import runtime

PIPELINE = {
    'args': {
        'image_cond_model': {'args': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'}},
        'rembg_model': {'args': {'model_name': 'briaai/RMBG-2.0'}},
    }
}

class FakeScene:
    def apply_transform(self, matrix):
        pass
    def export(self, file_type):
        return b'rotated'

fake_trimesh = types.ModuleType('trimesh')
fake_trimesh.load = lambda path, file_type, force, process: FakeScene()
fake_trimesh.transformations = types.SimpleNamespace(rotation_matrix=lambda angle, axis: None)
sys.modules['trimesh'] = fake_trimesh

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    layout = resolve_modly_layout(ext_dir)
    for sentinel in PRIMARY_ASSET.sentinel_paths:
        path = resolve_storage_path(layout, sentinel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(PIPELINE) if sentinel.endswith('pipeline.json') else 'primary', encoding='utf-8')
    image = ext_dir / 'input.png'
    image.write_bytes(b'image')
    (ext_dir / 'outputs').mkdir()
    calls = []
    def failing_downloader(*, repo_id, filename, destination, source_kind='hf_repo', **_kwargs):
        calls.append({'repo_id': repo_id, 'filename': filename, 'source_kind': source_kind})
        Path(destination).write_text('partial', encoding='utf-8')
        raise RuntimeError('simulated network failure')
    def fake_pipeline(_source):
        def pipeline(**kwargs):
            glb = Path(kwargs['output_dir']) / 'bootstrap-remote.glb'
            glb.write_bytes(b'raw')
            return {'glb_path': str(glb)}
        return pipeline
    result = runtime.run_job({
        'workspace_root': str(ext_dir),
        'input_image': 'input.png',
        'output_dir': 'outputs',
        'readiness': {'generation_allowed': True, 'code': 'ready'},
        'auxiliary_mode': 'default',
        'network_available': True,
        'auxiliary_bootstrap_downloader': failing_downloader,
        'params': {},
    }, pipeline_factory=fake_pipeline)
    pipeline = json.loads(resolve_storage_path(layout, 'models/pixal3d/generate/pipeline.json').read_text(encoding='utf-8'))
    print(json.dumps({
        'status': result['status'],
        'call_count': len(calls),
        'first_call': calls[0],
        'dino': pipeline['args']['image_cond_model']['args']['model_name'],
        'rmbg': pipeline['args']['rembg_model']['args']['model_name'],
        'moge': result['auxiliary_source']['sources']['moge']['value'],
        'naf': result['auxiliary_source']['sources']['naf']['value'],
        'source_code': result['auxiliary_source']['code'],
        'source_kinds': {key: value['kind'] for key, value in result['auxiliary_source']['sources'].items()},
        'bootstrap_status': result['auxiliary_source']['auxiliary_bootstrap']['status'],
        'warning_codes': [warning['code'] for warning in result['auxiliary_source'].get('warnings', [])],
        'glb_exists': Path(result['output']['glb_path']).is_file(),
    }, sort_keys=True))
`)

  assert.equal(result.status, 'completed')
  assert.equal(result.call_count, 1)
  assert.deepEqual(result.first_call, { repo_id: 'camenduru/dinov3-vitl16-pretrain-lvd1689m', filename: 'config.json', source_kind: 'hf_repo' })
  assert.equal(result.dino, 'camenduru/dinov3-vitl16-pretrain-lvd1689m')
  assert.equal(result.rmbg, 'camenduru/RMBG-2.0')
  assert.equal(result.moge, 'Ruicheng/moge-2-vitl')
  assert.equal(result.naf, 'https://github.com/valeoai/NAF/releases/download/model/naf_release.pth')
  assert.equal(result.source_code, 'remote_auxiliary_fallback')
  assert.deepEqual(result.source_kinds, { dino: 'remote', moge: 'remote', naf: 'remote', rmbg: 'remote' })
  assert.equal(result.bootstrap_status, 'failed')
  assert.deepEqual(result.warning_codes, ['auxiliary_bootstrap_failed_remote_fallback_preserved'])
  assert.equal(result.glb_exists, true)
})

test('strict local/offline readiness and runtime fail early on missing NAF without importing inference or hubconf', () => {
  const result = runPython(`
import importlib, json, tempfile
from pathlib import Path
from pixal3d_extension.assets import AUXILIARY_ASSETS, PRIMARY_ASSET
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
from pixal3d_extension.readiness import check_readiness
from pixal3d_extension import runtime

PIPELINE = {
    'args': {
        'image_cond_model': {'args': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'}},
        'rembg_model': {'args': {'model_name': 'briaai/RMBG-2.0'}},
    }
}

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    layout = resolve_modly_layout(ext_dir)
    for sentinel in PRIMARY_ASSET.sentinel_paths:
        path = resolve_storage_path(layout, sentinel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(PIPELINE) if sentinel.endswith('pipeline.json') else 'primary', encoding='utf-8')
    for key, manifest in AUXILIARY_ASSETS.items():
        if key == 'naf':
            continue
        for sentinel in manifest.sentinel_paths:
            path = resolve_storage_path(layout, sentinel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('aux', encoding='utf-8')

    image = ext_dir / 'input.png'
    image.write_bytes(b'image')
    (ext_dir / 'outputs').mkdir()
    imports = []
    downloader_calls = []
    def forbidden_downloader(*_args, **_kwargs):
        downloader_calls.append('network')
        raise AssertionError('local/offline/strict must not bootstrap auxiliary assets')
    original_import_module = runtime.importlib.import_module
    def forbidden_import(name):
        imports.append(name)
        if name in {'inference', 'hubconf'}:
            raise AssertionError('inference/hubconf must not be imported before missing_auxiliary_assets failure')
        return original_import_module(name)
    runtime.importlib.import_module = forbidden_import
    try:
        runtime_results = {}
        for mode in ['local', 'offline', 'strict']:
            runtime_results[mode] = runtime.run_job({
                'workspace_root': str(ext_dir),
                'input_image': 'input.png',
                'output_dir': 'outputs',
                'readiness': {'generation_allowed': True, 'code': 'ready'},
                'auxiliary_mode': mode,
                'network_available': False,
                'auxiliary_bootstrap_downloader': forbidden_downloader,
                'params': {},
            })
    finally:
        runtime.importlib.import_module = original_import_module

    readiness_codes = {
        mode: check_readiness(ext_dir, auxiliary_mode=mode, network_available=False, runtime_validated=True)['code']
        for mode in ['local', 'offline', 'strict']
    }
    print(json.dumps({
        'readiness_codes': readiness_codes,
        'runtime_statuses': {mode: result['status'] for mode, result in runtime_results.items()},
        'runtime_codes': {mode: result['code'] for mode, result in runtime_results.items()},
        'runtime_missing_counts': {mode: len(result.get('missing', [])) for mode, result in runtime_results.items()},
        'runtime_missing': {mode: result.get('missing', []) for mode, result in runtime_results.items()},
        'inference_imported': 'inference' in imports,
        'hubconf_imported': 'hubconf' in imports,
        'downloader_calls': downloader_calls,
    }, sort_keys=True))
`)

  assert.deepEqual(result.readiness_codes, {
    local: 'missing_auxiliary_assets',
    offline: 'missing_auxiliary_assets',
    strict: 'missing_auxiliary_assets',
  })
  assert.deepEqual(result.runtime_statuses, { local: 'failed', offline: 'failed', strict: 'failed' })
  assert.deepEqual(result.runtime_codes, { local: 'missing_auxiliary_assets', offline: 'missing_auxiliary_assets', strict: 'missing_auxiliary_assets' })
  assert.deepEqual(result.runtime_missing_counts, { local: 1, offline: 1, strict: 1 })
  assert.ok(result.runtime_missing.local.includes('models/pixal3d/auxiliary/naf/naf_release.pth'))
  assert.equal(result.inference_imported, false)
  assert.equal(result.hubconf_imported, false)
  assert.deepEqual(result.downloader_calls, [])
})

test('runtime local mode patches inference DINO configs and MoGe loader to local assets before generation', () => {
  const result = runPython(`
import json, sys, tempfile, types
from pathlib import Path
from pixal3d_extension.assets import AUXILIARY_ASSETS, PRIMARY_ASSET
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
from pixal3d_extension.pipeline_patch import patch_pipeline
from pixal3d_extension import runtime

PIPELINE = {
    'args': {
        'image_cond_model': {'args': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'}},
        'rembg_model': {'args': {'model_name': 'briaai/RMBG-2.0'}},
    }
}

class FakeScene:
    def apply_transform(self, matrix):
        pass
    def export(self, file_type):
        return b'rotated'

fake_trimesh = types.ModuleType('trimesh')
fake_trimesh.load = lambda path, file_type, force, process: FakeScene()
fake_trimesh.transformations = types.SimpleNamespace(rotation_matrix=lambda angle, axis: None)
sys.modules['trimesh'] = fake_trimesh

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    layout = resolve_modly_layout(ext_dir)
    for sentinel in PRIMARY_ASSET.sentinel_paths:
        path = resolve_storage_path(layout, sentinel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(PIPELINE) if sentinel.endswith('pipeline.json') else 'primary', encoding='utf-8')
    for manifest in AUXILIARY_ASSETS.values():
        for sentinel in manifest.sentinel_paths:
            path = resolve_storage_path(layout, sentinel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('aux', encoding='utf-8')

    patch_pipeline(ext_dir, auxiliary_mode='local', network_available=False)
    local_dino = str(resolve_storage_path(layout, 'models/pixal3d/auxiliary/dinov3'))
    local_moge = str(resolve_storage_path(layout, 'models/pixal3d/auxiliary/moge/model.pt'))
    image = ext_dir / 'input.png'
    image.write_bytes(b'image')
    (ext_dir / 'outputs').mkdir()

    fake_inference = types.ModuleType('inference')
    fake_config_object = types.SimpleNamespace(model_name='facebook/dinov3-vitl16-pretrain-lvd1689m')
    fake_inference.IMAGE_COND_CONFIGS = {
        'ss': {'model_name': 'camenduru/dinov3-vitl16-pretrain-lvd1689m'},
        'shape_512': fake_config_object,
    }
    captured = {}
    def load_moge_model(device='cuda', model_name='Ruicheng/moge-2-vitl'):
        captured['moge'] = {'device': device, 'model_name': model_name}
        return types.SimpleNamespace(to=lambda _device: None, eval=lambda: None)
    def run_inference(*, image_path, output_path, seed, model_path, manual_fov, low_vram, resolution):
        del image_path, seed, model_path, manual_fov, low_vram, resolution
        captured['configs'] = {
            'ss': fake_inference.IMAGE_COND_CONFIGS['ss']['model_name'],
            'shape_512': fake_inference.IMAGE_COND_CONFIGS['shape_512'].model_name,
        }
        fake_inference.load_moge_model(device='cuda')
        Path(output_path).write_bytes(b'raw')
    fake_inference.load_moge_model = load_moge_model
    fake_inference.run_inference = run_inference
    sys.modules['inference'] = fake_inference

    fake_hubconf = types.ModuleType('hubconf')
    class FakeNAF:
        def to(self, _device):
            return self
        def load_state_dict(self, _state_dict):
            return None
    fake_hubconf.NAF = FakeNAF
    fake_hubconf.naf = lambda pretrained=True, device='cpu': FakeNAF().to(device)
    sys.modules['hubconf'] = fake_hubconf

    runtime._prepare_runtime_compat = lambda: None
    runtime._install_windows_native_module_aliases = lambda: None
    runtime._install_natten_fallback = lambda: None
    runtime._silence_flex_gemm_autotuners = lambda: None

    result = runtime.run_job({
        'workspace_root': str(ext_dir),
        'input_image': 'input.png',
        'output_dir': 'outputs',
        'readiness': {'generation_allowed': True, 'code': 'ready'},
        'auxiliary_mode': 'local',
        'network_available': False,
        'model_source': str(resolve_storage_path(layout, 'models/pixal3d/generate')),
        'params': {'seed': 1},
    })
    print(json.dumps({
        'status': result['status'],
        'local_dino': local_dino,
        'local_moge': local_moge,
        'configs': captured['configs'],
        'moge': captured['moge'],
        'glb_exists': Path(result['output']['glb_path']).is_file(),
    }, sort_keys=True))
`)

  assert.equal(result.status, 'completed')
  assert.equal(result.configs.ss, result.local_dino)
  assert.equal(result.configs.shape_512, result.local_dino)
  assert.equal(result.moge.model_name, result.local_moge)
  assert.equal(result.glb_exists, true)
})

test('runtime local mode patches NAF to local checkpoint without Torch Hub downloader', () => {
  const result = runPython(`
import json, sys, tempfile, types
from pathlib import Path
from pixal3d_extension.assets import AUXILIARY_ASSETS, PRIMARY_ASSET
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
from pixal3d_extension.pipeline_patch import patch_pipeline
from pixal3d_extension import runtime

PIPELINE = {
    'args': {
        'image_cond_model': {'args': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'}},
        'rembg_model': {'args': {'model_name': 'briaai/RMBG-2.0'}},
    }
}

class FakeScene:
    def apply_transform(self, matrix):
        pass
    def export(self, file_type):
        return b'rotated'

fake_trimesh = types.ModuleType('trimesh')
fake_trimesh.load = lambda path, file_type, force, process: FakeScene()
fake_trimesh.transformations = types.SimpleNamespace(rotation_matrix=lambda angle, axis: None)
sys.modules['trimesh'] = fake_trimesh

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    layout = resolve_modly_layout(ext_dir)
    for sentinel in PRIMARY_ASSET.sentinel_paths:
        path = resolve_storage_path(layout, sentinel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(PIPELINE) if sentinel.endswith('pipeline.json') else 'primary', encoding='utf-8')
    for manifest in AUXILIARY_ASSETS.values():
        for sentinel in manifest.sentinel_paths:
            path = resolve_storage_path(layout, sentinel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('aux', encoding='utf-8')

    patch_pipeline(ext_dir, auxiliary_mode='local', network_available=False)
    local_naf = str(resolve_storage_path(layout, 'models/pixal3d/auxiliary/naf/naf_release.pth'))
    image = ext_dir / 'input.png'
    image.write_bytes(b'image')
    (ext_dir / 'outputs').mkdir()

    captured = {'torch_hub_calls': 0}
    fake_torch = types.ModuleType('torch')
    def fake_torch_load(path, map_location=None):
        captured['torch_load'] = {'path': str(path), 'map_location': str(map_location)}
        return {'weights': 'local'}
    fake_torch.load = fake_torch_load
    def forbidden_url_loader(*_args, **_kwargs):
        captured['torch_hub_calls'] += 1
        raise AssertionError('local NAF mode must not call torch.hub.load_state_dict_from_url')
    fake_torch.hub = types.SimpleNamespace(load_state_dict_from_url=forbidden_url_loader)
    sys.modules['torch'] = fake_torch

    fake_hubconf = types.ModuleType('hubconf')
    class FakeNAF:
        def to(self, device):
            captured['naf_device'] = str(device)
            return self
        def load_state_dict(self, state_dict):
            captured['naf_state_dict'] = state_dict
            return None
    fake_hubconf.NAF = FakeNAF
    def remote_naf(pretrained=True, device='cpu'):
        if pretrained:
            fake_torch.hub.load_state_dict_from_url('https://github.com/valeoai/NAF/releases/download/model/naf_release.pth', progress=True, map_location=device)
        return FakeNAF().to(device)
    fake_hubconf.naf = remote_naf
    sys.modules['hubconf'] = fake_hubconf

    fake_inference = types.ModuleType('inference')
    fake_inference.IMAGE_COND_CONFIGS = {}
    fake_inference.load_moge_model = lambda *args, **kwargs: None
    def run_inference(*, image_path, output_path, seed, model_path, manual_fov, low_vram, resolution):
        del image_path, seed, model_path, manual_fov, low_vram, resolution
        from hubconf import naf
        model = naf(pretrained=True, device='cuda')
        captured['patched_checkpoint'] = getattr(naf, '__modly_local_checkpoint__', None)
        captured['returned_model_type'] = type(model).__name__
        Path(output_path).write_bytes(b'raw')
    fake_inference.run_inference = run_inference
    sys.modules['inference'] = fake_inference

    runtime._prepare_runtime_compat = lambda: None
    runtime._install_windows_native_module_aliases = lambda: None
    runtime._install_natten_fallback = lambda: None
    runtime._silence_flex_gemm_autotuners = lambda: None

    result = runtime.run_job({
        'workspace_root': str(ext_dir),
        'input_image': 'input.png',
        'output_dir': 'outputs',
        'readiness': {'generation_allowed': True, 'code': 'ready'},
        'auxiliary_mode': 'local',
        'network_available': False,
        'model_source': str(resolve_storage_path(layout, 'models/pixal3d/generate')),
        'params': {'seed': 1},
    })
    print(json.dumps({
        'status': result['status'],
        'local_naf': local_naf,
        'patched_checkpoint': captured['patched_checkpoint'],
        'torch_load': captured.get('torch_load'),
        'torch_hub_calls': captured['torch_hub_calls'],
        'naf_device': captured['naf_device'],
        'naf_state_dict': captured['naf_state_dict'],
        'returned_model_type': captured['returned_model_type'],
        'glb_exists': Path(result['output']['glb_path']).is_file(),
    }, sort_keys=True))
`)

  assert.equal(result.status, 'completed')
  assert.equal(result.patched_checkpoint, result.local_naf)
  assert.deepEqual(result.torch_load, { path: result.local_naf, map_location: 'cuda' })
  assert.equal(result.torch_hub_calls, 0)
  assert.equal(result.naf_device, 'cuda')
  assert.deepEqual(result.naf_state_dict, { weights: 'local' })
  assert.equal(result.returned_model_type, 'FakeNAF')
  assert.equal(result.glb_exists, true)
})

test('DINO/RMBG/MoGe/NAF are local-first while full offline and NATTEN strict kernels remain separate', () => {
  const readme = readFileSync(join(repoRoot, 'README.md'), 'utf8')
  assert.match(readme, /not\*\* a full offline-generation guarantee|not\*\* a full offline/i)
  assert.match(readme, /Ruicheng\/moge-2-vitl/)
  assert.match(readme, /models\/pixal3d\/auxiliary\/moge\/model\.pt/)
  assert.match(readme, /models\/pixal3d\/auxiliary\/naf\/naf_release\.pth/)
  assert.match(readme, /MoGeModel\.from_pretrained\(\)/)
  assert.match(readme, /torch\.hub\.load_state_dict_from_url/)
  assert.match(readme, /hubconf\.naf/)
  assert.match(readme, /naf_release\.pth/)
  assert.match(readme, /no-DNS generation smoke test/i)
  assert.match(readme, /NATTEN\/libnatten.*separate|separate from strict NAF native kernels/i)

  const result = runPython(`
import json
from pixal3d_extension.assets import AUXILIARY_ASSETS, LOCALIZABLE_RUNTIME_DEPENDENCIES, UNLOCALIZED_RUNTIME_DEPENDENCIES
print(json.dumps({
    'moge_sentinels': list(AUXILIARY_ASSETS['moge'].sentinels),
    'naf_sentinels': list(AUXILIARY_ASSETS['naf'].sentinels),
    'localizable': {entry['key']: entry['offline_status'] for entry in LOCALIZABLE_RUNTIME_DEPENDENCIES},
    'strict_kernel_notes': {entry['key']: entry.get('strict_kernel_note') for entry in LOCALIZABLE_RUNTIME_DEPENDENCIES if entry.get('strict_kernel_note')},
    'unlocalized': {entry['key']: entry['offline_status'] for entry in UNLOCALIZED_RUNTIME_DEPENDENCIES},
}, sort_keys=True))
`)

  assert.deepEqual(result.moge_sentinels, ['model.pt'])
  assert.deepEqual(result.naf_sentinels, ['naf_release.pth'])
  assert.deepEqual(result.localizable, {
    moge: 'local_first_with_remote_or_hf_cache_fallback',
    naf: 'local_first_with_torch_cache_or_network_fallback_in_default',
  })
  assert.match(result.strict_kernel_notes.naf, /NATTEN\/libnatten native kernel availability/)
  assert.deepEqual(result.unlocalized, {})
})

test('setup.py can be loaded through runpy from a different current directory', () => {
  const result = spawnSync(python, ['-c', `
import os, runpy, tempfile
os.chdir(tempfile.gettempdir())
runpy.run_path(${JSON.stringify(join(repoRoot, 'setup.py'))}, run_name='pixal3d_setup_probe')
print('ok')
`], { encoding: 'utf8' })

  assert.equal(result.status, 0, `STDOUT:\n${result.stdout}\nSTDERR:\n${result.stderr}`)
  assert.match(result.stdout, /ok/)
})

test('setup installs torch before requirements and native wheelhouse packages', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    venv_python = root / 'venv' / 'bin' / 'python'
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text('#!/usr/bin/env python3\\n', encoding='utf-8')
    linux_x64 = root / '.modly' / 'cache' / 'wheelhouse' / 'pixal3d' / '0.1.0' / 'linux-x64-cp312-cuda124' / 'extracted'
    linux_x64.mkdir(parents=True)
    linux_aarch64 = root / '.modly' / 'cache' / 'wheelhouse' / 'pixal3d' / '0.1.0' / 'linux-aarch64-cp312-cuda124' / 'extracted'
    linux_aarch64.mkdir(parents=True)
    calls = []
    def fake_run(command, *, cwd):
        calls.append(command)
        return {'args': command, 'returncode': 0, 'stdout_tail': '{"HAS_LIBNATTEN": false, "importable": true, "ok": true, "torch_cuda_available": true, "torch_cuda_version": "13.0"}\\n' if command[1] == '-c' else '', 'stderr_tail': '', 'ok': True}
    setup._run_setup_command = fake_run
    setup._install_prepare_dependencies(root, wheelhouse_path=linux_x64)
    x64_commands = calls[:]
    calls.clear()
    setup._install_prepare_dependencies(root, wheelhouse_path=linux_aarch64)
    print(json.dumps({'x64': x64_commands[:4], 'aarch64': calls[:4]}, sort_keys=True))
`)

  assert.deepEqual(result.x64[0].slice(2, 5), ['pip', 'install', '--no-deps'])
  assert.ok(result.x64[0].some((arg) => /pip-25\.3-py3-none-any\.whl#sha256=/.test(arg)))
  assert.deepEqual(result.x64[1].slice(2, 7), ['pip', 'install', '--no-cache-dir', '--retries', '5'])
  assert.ok(result.x64[1].includes('--timeout'))
  assert.ok(result.x64[1].includes('https://download.pytorch.org/whl/cu124'))
  assert.ok(result.x64[1].includes('torch==2.6.0+cu124'))
  assert.ok(result.x64[1].includes('torchvision==0.21.0+cu124'))
  assert.deepEqual(result.x64[2].slice(2, 7), ['pip', 'install', '--no-cache-dir', '--retries', '5'])
  assert.ok(result.x64[2].includes('-r'))
  assert.ok(result.x64[2].includes('requirements.txt'))
  assert.ok(result.x64[3].includes('triton'))
  assert.ok(result.aarch64[3].includes('triton'))
  assert.deepEqual(result.aarch64[1].slice(2, 8), ['pip', 'install', '--no-cache-dir', '--retries', '5', '--timeout'])
  assert.ok(!result.aarch64[1].includes('--no-deps'))
  assert.ok(result.aarch64[1].includes('torch==2.12.0'))
  assert.ok(result.aarch64[1].includes('torchvision==0.27.0'))
  assert.ok(!result.aarch64[1].includes('torch==2.6.0'))
})

test('runtime failure helper returns structured diagnostics with restricted environment context', () => {
  const result = runPython(`
import json, os, tempfile
from pathlib import Path
from pixal3d_extension import runtime

allowed = {'TEMP', 'TMP', 'TMPDIR', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'TORCH_HOME', 'XDG_CACHE_HOME'}
for key in [*allowed, 'MODLY_TOKEN', 'HF_TOKEN', 'AWS_SECRET_ACCESS_KEY']:
    os.environ.pop(key, None)
os.environ['TEMP'] = 'C:\\\\Temp'
os.environ['HF_HOME'] = 'I:\\\\hf-cache'
os.environ['MODLY_TOKEN'] = 'secret-token'
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    image = root / 'input.png'
    output = root / 'outputs'
    image.write_bytes(b'image')
    output.mkdir()
    try:
        raise OSError("[WinError 3] Impossibile trovare il percorso specificato: 'I:\\\\'")
    except OSError as exc:
        failure = runtime._runtime_failure(
            exc,
            'import_inference',
            {'model_source': 'local-model-source'},
            image,
            output,
        )
    context = failure['runtime_context']
    print(json.dumps({
        'status': failure['status'],
        'code': failure['code'],
        'message': failure['message'],
        'generation_allowed': failure['generation_allowed'],
        'checkpoint': failure['checkpoint'],
        'exception_type': failure['exception_type'],
        'traceback_contains_oserror': 'OSError' in failure['traceback'],
        'traceback_contains_path': 'I:' in failure['traceback'],
        'runtime_context': context,
        'env_keys': sorted(context['env'].keys()),
        'has_secret_key': 'MODLY_TOKEN' in context['env'] or 'HF_TOKEN' in context['env'] or 'AWS_SECRET_ACCESS_KEY' in context['env'],
        'context_has_secret_top_level': 'MODLY_TOKEN' in context or 'HF_TOKEN' in context or 'AWS_SECRET_ACCESS_KEY' in context,
        'path_hint': failure.get('path_hint'),
    }, sort_keys=True))
`)

  assert.equal(result.status, 'failed')
  assert.equal(result.code, 'runtime_failed')
  assert.equal(result.generation_allowed, false)
  assert.match(result.message, /\[WinError 3\]/)
  assert.equal(result.checkpoint, 'import_inference')
  assert.equal(result.exception_type, 'OSError')
  assert.equal(result.traceback_contains_oserror, true)
  assert.equal(result.traceback_contains_path, true)
  assert.equal(result.runtime_context.model_source, 'local-model-source')
  assert.match(result.runtime_context.input_image.replaceAll('\\', '/'), /input\.png$/)
  assert.match(result.runtime_context.output_dir.replaceAll('\\', '/'), /outputs$/)
  assert.match(result.runtime_context.cwd.replaceAll('\\', '/'), /modly-pixal3d-extension$/)
  assert.deepEqual(result.env_keys, ['HF_HOME', 'TEMP'])
  assert.deepEqual(result.runtime_context.env, { HF_HOME: 'I:\\hf-cache', TEMP: 'C:\\Temp' })
  assert.equal(result.has_secret_key, false)
  assert.equal(result.context_has_secret_top_level, false)
  assert.deepEqual(result.path_hint, {
    code: 'missing_windows_path',
    message: 'A configured runtime/cache/model/input/output path appears to reference a missing Windows drive or root.',
    path: 'I:\\',
  })
})

test('run_job reports runtime checkpoint diagnostics when upstream inference import fails', () => {
  const result = runPython(`
import json, os, sys, tempfile, types
from pathlib import Path
from pixal3d_extension.assets import AUXILIARY_ASSETS, PRIMARY_ASSET
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
from pixal3d_extension.pipeline_patch import patch_pipeline
from pixal3d_extension import runtime

for key in ['TEMP', 'TMP', 'TMPDIR', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'TORCH_HOME', 'XDG_CACHE_HOME', 'MODLY_TOKEN']:
    os.environ.pop(key, None)
os.environ['HF_HUB_CACHE'] = 'I:\\\\hf-cache'
os.environ['MODLY_TOKEN'] = 'secret-token'

PIPELINE = {
    'args': {
        'image_cond_model': {'args': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'}},
        'rembg_model': {'args': {'model_name': 'briaai/RMBG-2.0'}},
    }
}

with tempfile.TemporaryDirectory() as tmp:
    ext_dir = Path(tmp) / 'Modly' / 'data' / 'extensions' / 'pixal3d'
    ext_dir.mkdir(parents=True)
    layout = resolve_modly_layout(ext_dir)
    for sentinel in PRIMARY_ASSET.sentinel_paths:
        path = resolve_storage_path(layout, sentinel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(PIPELINE) if sentinel.endswith('pipeline.json') else 'primary', encoding='utf-8')
    for manifest in AUXILIARY_ASSETS.values():
        for sentinel in manifest.sentinel_paths:
            path = resolve_storage_path(layout, sentinel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('aux', encoding='utf-8')
    patch_pipeline(ext_dir, auxiliary_mode='local', network_available=False)

    image = ext_dir / 'input.png'
    image.write_bytes(b'image')
    (ext_dir / 'outputs').mkdir()

    fake_hubconf = types.ModuleType('hubconf')
    class FakeNAF:
        def to(self, _device):
            return self
        def load_state_dict(self, _state_dict):
            return None
    fake_hubconf.NAF = FakeNAF
    fake_hubconf.naf = lambda pretrained=True, device='cpu': FakeNAF().to(device)
    sys.modules['hubconf'] = fake_hubconf

    runtime._prepare_runtime_compat = lambda: None
    runtime._install_windows_native_module_aliases = lambda: None
    runtime._install_natten_fallback = lambda: None
    runtime._silence_flex_gemm_autotuners = lambda: None

    original_import_module = runtime.importlib.import_module
    def failing_import(name):
        if name == 'hubconf':
            return fake_hubconf
        if name == 'inference':
            raise OSError("[WinError 3] Cannot find path: 'I:\\\\'")
        return original_import_module(name)
    runtime.importlib.import_module = failing_import
    try:
        result = runtime.run_job({
            'workspace_root': str(ext_dir),
            'input_image': 'input.png',
            'output_dir': 'outputs',
            'readiness': {'generation_allowed': True, 'code': 'ready'},
            'auxiliary_mode': 'local',
            'network_available': False,
            'model_source': str(resolve_storage_path(layout, 'models/pixal3d/generate')),
            'params': {'seed': 1},
        })
    finally:
        runtime.importlib.import_module = original_import_module

    print(json.dumps({
        'status': result['status'],
        'code': result['code'],
        'message': result['message'],
        'checkpoint': result.get('checkpoint'),
        'exception_type': result.get('exception_type'),
        'traceback_contains_import_module': 'failing_import' in result.get('traceback', ''),
        'traceback_contains_oserror': 'OSError' in result.get('traceback', ''),
        'context': result.get('runtime_context'),
        'path_hint': result.get('path_hint'),
    }, sort_keys=True))
`)

  assert.equal(result.status, 'failed')
  assert.equal(result.code, 'runtime_failed')
  assert.match(result.message, /\[WinError 3\]/)
  assert.equal(result.checkpoint, 'import_inference')
  assert.equal(result.exception_type, 'OSError')
  assert.equal(result.traceback_contains_import_module, true)
  assert.equal(result.traceback_contains_oserror, true)
  assert.match(result.context.input_image.replaceAll('\\', '/'), /input\.png$/)
  assert.match(result.context.output_dir.replaceAll('\\', '/'), /outputs$/)
  assert.match(result.context.model_source.replaceAll('\\', '/'), /models\/pixal3d\/generate$/)
  assert.deepEqual(Object.keys(result.context.env), ['HF_HUB_CACHE'])
  assert.equal(result.context.env.HF_HUB_CACHE, 'I:\\hf-cache')
  assert.deepEqual(result.path_hint, {
    code: 'missing_windows_path',
    message: 'A configured runtime/cache/model/input/output path appears to reference a missing Windows drive or root.',
    path: 'I:\\',
  })
})

test('runtime suppresses upstream FlexGEMM autotuner verbose without disabling cache', () => {
  const runtime = readFileSync(join(repoRoot, 'pixal3d_extension', 'runtime.py'), 'utf8')
  assert.match(runtime, /def _silence_flex_gemm_autotuners\(\)/)
  assert.match(runtime, /FLEX_GEMM_AUTOTUNER_VERBOSE"\] = "0"/)
  assert.match(runtime, /value\.verbose = False/)
  assert.match(runtime, /flex_gemm\.utils\.load_autotune_cache\(\)/)
  assert.match(runtime, /inference_module = importlib\.import_module\("inference"\)[\s\S]*?run_inference = inference_module\.run_inference[\s\S]*?_silence_flex_gemm_autotuners\(\)/)
})

test('low_vram schema is UI-compatible select and runtime parses values explicitly', () => {
  const manifest = JSON.parse(readFileSync(join(repoRoot, 'manifest.json'), 'utf8'))
  const manifestParam = manifest.nodes[0].params_schema.find((param) => param.id === 'low_vram')
  assert.ok(manifestParam)
  assert.equal(manifestParam.type, 'select')
  assert.equal(manifestParam.default, 'low_vram')
  assert.deepEqual(manifestParam.options, [
    { value: 'low_vram', label: 'Low VRAM' },
    { value: 'standard', label: 'Standard' },
  ])

  const generatorSchema = runPython(`
import json
from generator import Pixal3DGenerator
schema = Pixal3DGenerator.params_schema()
low_vram = next(param for param in schema if param['id'] == 'low_vram')
print(json.dumps(low_vram, sort_keys=True))
`)
  assert.equal(generatorSchema.type, 'select')
  assert.equal(generatorSchema.default, 'low_vram')
  assert.deepEqual(generatorSchema.options, [
    { value: 'low_vram', label: 'Low VRAM' },
    { value: 'standard', label: 'Standard' },
  ])

  const runtime = readFileSync(join(repoRoot, 'pixal3d_extension', 'runtime.py'), 'utf8')
  assert.match(runtime, /def _parse_low_vram\(value: Any, default: bool = True\) -> bool:/)
  assert.doesNotMatch(runtime, /bool\(params\.get\("low_vram", False\)\)/)

  const parsed = runPython(`
import json
from pixal3d_extension.runtime import _parse_low_vram
print(json.dumps({
    'default_none': _parse_low_vram(None),
    'bool_true': _parse_low_vram(True),
    'bool_false': _parse_low_vram(False),
    'string_true': _parse_low_vram('true'),
    'string_false': _parse_low_vram('false'),
    'string_low_vram': _parse_low_vram('low_vram'),
    'string_standard': _parse_low_vram('standard'),
    'int_one': _parse_low_vram(1),
    'int_zero': _parse_low_vram(0),
    'unknown_uses_default_false': _parse_low_vram('maybe', default=False),
}, sort_keys=True))
`)
  assert.deepEqual(parsed, {
    bool_false: false,
    bool_true: true,
    default_none: true,
    int_one: true,
    int_zero: false,
    string_false: false,
    string_low_vram: true,
    string_standard: false,
    string_true: true,
    unknown_uses_default_false: false,
  })
})

test('texture_size schema is UI-compatible select with safe default', () => {
  const manifest = JSON.parse(readFileSync(join(repoRoot, 'manifest.json'), 'utf8'))
  const manifestParam = manifest.nodes[0].params_schema.find((param) => param.id === 'texture_size')
  assert.ok(manifestParam)
  assert.equal(manifestParam.label, 'Texture Size')
  assert.equal(manifestParam.type, 'select')
  assert.equal(manifestParam.default, 1024)
  assert.deepEqual(manifestParam.options, [
    { value: 1024, label: '1024' },
    { value: 2048, label: '2048' },
  ])
  assert.match(manifestParam.tooltip, /final GLB texture atlas size/i)
  assert.match(manifestParam.tooltip, /1024.*reduces VRAM/i)
  assert.match(manifestParam.tooltip, /2048.*higher quality.*higher VRAM/i)

  const generatorSchema = runPython(`
import json
from generator import Pixal3DGenerator
schema = Pixal3DGenerator.params_schema()
texture_size = next(param for param in schema if param['id'] == 'texture_size')
print(json.dumps(texture_size, sort_keys=True))
`)
  assert.equal(generatorSchema.label, 'Texture Size')
  assert.equal(generatorSchema.type, 'select')
  assert.equal(generatorSchema.default, 1024)
  assert.deepEqual(generatorSchema.options, [
    { value: 1024, label: '1024' },
    { value: 2048, label: '2048' },
  ])
  assert.match(generatorSchema.tooltip, /final GLB texture atlas size/i)
  assert.match(generatorSchema.tooltip, /1024.*reduces VRAM/i)
  assert.match(generatorSchema.tooltip, /2048.*higher quality.*higher VRAM/i)
})

test('texture_size runtime parser scopes env override and passes safe value through pipeline factory', () => {
  const result = runPython(`
import json, os, tempfile
from pathlib import Path
from pixal3d_extension import runtime
from pixal3d_extension.runtime import PIXAL3D_TEXTURE_SIZE_ENV, _parse_texture_size, _scoped_texture_size_env

parsed = {
    'default_none': _parse_texture_size(None),
    'string_1024': _parse_texture_size('1024'),
    'string_2048_padded': _parse_texture_size(' 2048 '),
    'int_2048': _parse_texture_size(2048),
    'invalid_string': _parse_texture_size('4096'),
    'invalid_int': _parse_texture_size(4096),
    'bool_true': _parse_texture_size(True),
}

previous = os.environ.get(PIXAL3D_TEXTURE_SIZE_ENV)
try:
    os.environ.pop(PIXAL3D_TEXTURE_SIZE_ENV, None)
    with _scoped_texture_size_env(2048):
        absent_during = os.environ.get(PIXAL3D_TEXTURE_SIZE_ENV)
    absent_after = PIXAL3D_TEXTURE_SIZE_ENV in os.environ

    os.environ[PIXAL3D_TEXTURE_SIZE_ENV] = '2048'
    with _scoped_texture_size_env(1024):
        existing_during = os.environ.get(PIXAL3D_TEXTURE_SIZE_ENV)
    existing_after = os.environ.get(PIXAL3D_TEXTURE_SIZE_ENV)
finally:
    if previous is None:
        os.environ.pop(PIXAL3D_TEXTURE_SIZE_ENV, None)
    else:
        os.environ[PIXAL3D_TEXTURE_SIZE_ENV] = previous

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    image = root / 'input.png'
    image.write_bytes(b'image')
    output = root / 'outputs'
    output.mkdir()
    captured = {}
    def fake_pipeline(_source):
        def pipeline(**kwargs):
            captured.update(kwargs)
            glb_path = Path(kwargs['output_dir']) / 'texture-size.glb'
            glb_path.write_bytes(b'raw')
            return {'glb_path': str(glb_path)}
        return pipeline
    result = runtime.run_job({
        'input_image': str(image),
        'output_dir': str(output),
        'readiness': {'generation_allowed': True, 'code': 'ready'},
        'params': {'texture_size': '4096'},
    }, pipeline_factory=fake_pipeline)

print(json.dumps({
    'parsed': parsed,
    'absent_during': absent_during,
    'absent_after': absent_after,
    'existing_during': existing_during,
    'existing_after': existing_after,
    'status': result['status'],
    'pipeline_texture_size': captured.get('texture_size'),
    'params_texture_size': result['params']['texture_size'],
}, sort_keys=True))
`)

  assert.deepEqual(result.parsed, {
    bool_true: 1024,
    default_none: 1024,
    int_2048: 2048,
    invalid_int: 1024,
    invalid_string: 1024,
    string_1024: 1024,
    string_2048_padded: 2048,
  })
  assert.equal(result.absent_during, '2048')
  assert.equal(result.absent_after, false)
  assert.equal(result.existing_during, '1024')
  assert.equal(result.existing_after, '2048')
  assert.equal(result.status, 'completed')
  assert.equal(result.pipeline_texture_size, 1024)
  assert.equal(result.params_texture_size, 1024)
})

test('runtime scopes PIXAL3D_TEXTURE_SIZE around upstream run_inference without unsupported kwargs', () => {
  const result = runPython(`
import json, os, sys, tempfile, types
from pathlib import Path
from pixal3d_extension import runtime
from pixal3d_extension.runtime import PIXAL3D_TEXTURE_SIZE_ENV

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    image = root / 'input.png'
    image.write_bytes(b'image')
    output = root / 'outputs'
    output.mkdir()
    captured = {}

    fake_inference = types.ModuleType('inference')
    def run_inference(*, image_path, output_path, seed, model_path, manual_fov, low_vram, resolution):
        del image_path, seed, model_path, manual_fov, low_vram, resolution
        captured['env_during'] = os.environ.get(PIXAL3D_TEXTURE_SIZE_ENV)
        Path(output_path).write_bytes(b'raw')
    fake_inference.run_inference = run_inference
    fake_inference.IMAGE_COND_CONFIGS = {}
    fake_inference.load_moge_model = lambda *args, **kwargs: None
    sys.modules['inference'] = fake_inference

    runtime._prepare_runtime_compat = lambda: None
    runtime._install_windows_native_module_aliases = lambda: None
    runtime._install_natten_fallback = lambda: None
    runtime._silence_flex_gemm_autotuners = lambda: None

    previous = os.environ.get(PIXAL3D_TEXTURE_SIZE_ENV)
    try:
        os.environ[PIXAL3D_TEXTURE_SIZE_ENV] = '2048'
        result = runtime.run_job({
            'input_image': str(image),
            'output_dir': str(output),
            'readiness': {'generation_allowed': True, 'code': 'ready'},
            'params': {'texture_size': '1024'},
        })
        env_after = os.environ.get(PIXAL3D_TEXTURE_SIZE_ENV)
    finally:
        if previous is None:
            os.environ.pop(PIXAL3D_TEXTURE_SIZE_ENV, None)
        else:
            os.environ[PIXAL3D_TEXTURE_SIZE_ENV] = previous

print(json.dumps({
    'status': result['status'],
    'env_during': captured['env_during'],
    'env_after': env_after,
    'params_texture_size': result['params']['texture_size'],
}, sort_keys=True))
`)

  assert.equal(result.status, 'completed')
  assert.equal(result.env_during, '1024')
  assert.equal(result.env_after, '2048')
  assert.equal(result.params_texture_size, 1024)
})

test('o_voxel compat postprocess caps texture_size from PIXAL3D_TEXTURE_SIZE', () => {
  const result = runPython(`
import importlib, json, os, sys, types

for name in [
    'pixal3d_extension.o_voxel_compat.postprocess',
    'cv2',
    'cumesh',
    'nvdiffrast',
    'nvdiffrast.torch',
    'torch',
    'trimesh',
    'trimesh.visual',
    'trimesh.visual.material',
    'flex_gemm',
    'flex_gemm.ops',
    'flex_gemm.ops.grid_sample',
    'PIL',
    'PIL.Image',
    'tqdm',
]:
    sys.modules.pop(name, None)

def module(name):
    created = types.ModuleType(name)
    sys.modules[name] = created
    return created

module('cv2')
module('cumesh')

fake_numpy = types.ModuleType('numpy')
fake_numpy.ndarray = object
fake_numpy.radians = lambda value: value * 3.141592653589793 / 180.0
sys.modules['numpy'] = fake_numpy

fake_nvdiffrast = module('nvdiffrast')
fake_dr = module('nvdiffrast.torch')
fake_nvdiffrast.torch = fake_dr

fake_torch = module('torch')
fake_torch.Tensor = object

fake_trimesh = module('trimesh')
fake_trimesh.__path__ = []
fake_visual = module('trimesh.visual')
fake_material = module('trimesh.visual.material')
fake_visual.material = fake_material
fake_trimesh.visual = fake_visual

fake_flex = module('flex_gemm')
fake_flex.__path__ = []
fake_ops = module('flex_gemm.ops')
fake_ops.__path__ = []
fake_grid = module('flex_gemm.ops.grid_sample')
fake_grid.grid_sample_3d = lambda *args, **kwargs: None
fake_ops.grid_sample = fake_grid
fake_flex.ops = fake_ops

fake_pil = module('PIL')
fake_pil.__path__ = []
fake_image = module('PIL.Image')
fake_pil.Image = fake_image

fake_tqdm = module('tqdm')
fake_tqdm.tqdm = lambda *args, **kwargs: None

postprocess = importlib.import_module('pixal3d_extension.o_voxel_compat.postprocess')
env_key = postprocess.PIXAL3D_TEXTURE_SIZE_ENV
previous = os.environ.get(env_key)
try:
    os.environ.pop(env_key, None)
    no_override_2048 = postprocess._effective_texture_size(2048)
    no_override_4096 = postprocess._effective_texture_size(4096)
    os.environ[env_key] = '1024'
    env_1024_caps_4096 = postprocess._effective_texture_size(4096)
    os.environ[env_key] = '2048'
    env_2048_caps_4096 = postprocess._effective_texture_size(4096)
    env_2048_preserves_lower_caller = postprocess._effective_texture_size(1024)
    os.environ[env_key] = '4096'
    invalid_env = postprocess._effective_texture_size(2048)
finally:
    if previous is None:
        os.environ.pop(env_key, None)
    else:
        os.environ[env_key] = previous

print(json.dumps({
    'no_override_2048': no_override_2048,
    'no_override_4096': no_override_4096,
    'env_1024_caps_4096': env_1024_caps_4096,
    'env_2048_caps_4096': env_2048_caps_4096,
    'env_2048_preserves_lower_caller': env_2048_preserves_lower_caller,
    'invalid_env': invalid_env,
}, sort_keys=True))
`)

  assert.deepEqual(result, {
    env_1024_caps_4096: 1024,
    env_2048_caps_4096: 2048,
    env_2048_preserves_lower_caller: 1024,
    invalid_env: 1024,
    no_override_2048: 2048,
    no_override_4096: 2048,
  })
})

test('linux x64 cp312 cuda124 wheelhouse workflow is documented and manifest-backed', () => {
  const manifest = JSON.parse(readFileSync(join(repoRoot, 'wheelhouse.manifest.json'), 'utf8'))
  const workflowPath = join(repoRoot, '.github', 'workflows', 'wheelhouse-linux-x64-cp312-cuda124.yml')
  const scriptPath = join(repoRoot, 'tools', 'wheelhouse', 'build-linux-x64-cp312-cuda124.sh')
  const recipe = readFileSync(join(repoRoot, 'tools', 'wheelhouse', 'README.md'), 'utf8')

  assert.deepEqual([...manifest.assets.map((asset) => asset.id)].sort(), [
    'linux-aarch64-cp312-cuda124',
    'linux-x64-cp312-cuda124',
    'windows-x64-cp311-cuda124',
    'windows-x64-cp312-cuda124',
  ].sort())
  assert.equal(existsSync(workflowPath), true)

  const workflow = readFileSync(workflowPath, 'utf8')
  assert.match(workflow, /linux-x64-cp312-cuda124/)
  assert.match(workflow, /ubuntu-22\.04/)
  assert.match(workflow, /libcurand-dev-12-4/)
  assert.match(workflow, /libcusparse-dev-12-4/)
  assert.match(workflow, /libcublas-dev-12-4/)
  assert.match(workflow, /libcusolver-dev-12-4/)
  assert.match(workflow, /cuda-nvrtc-dev-12-4/)
  assert.match(workflow, /curand_kernel\.h/)
  assert.match(workflow, /cusparse\.h/)
  assert.match(workflow, /cublas_v2\.h/)
  assert.match(workflow, /cusolverDn\.h/)
  assert.match(workflow, /nvrtc\.h/)
  assert.match(workflow, /libcuda\.so/)
  assert.match(workflow, /libnvrtc\.so/)
  assert.match(workflow, /LIBRARY_PATH="\$\{CUDA_HOME\}\/lib64:\$\{CUDA_HOME\}\/lib64\/stubs:/)
  assert.match(workflow, /LD_LIBRARY_PATH="\$\{CUDA_HOME\}\/lib64:\$\{CUDA_HOME\}\/lib64\/stubs:/)
  assert.match(workflow, /native-sources\.linux-x64-cp312-cuda124\.env\.example/)
  assert.match(workflow, /build\/wheelhouse\/\$\{\{ env\.WHEELHOUSE_LANE \}\}\/dist\/wheelhouse\/\*/)
  assert.match(workflow, /fail clearly/i)
  assert.match(workflow, /Do not upload placeholders/i)
  assert.doesNotMatch(workflow, /upload-release-asset/i)
  assert.equal(existsSync(scriptPath), true)

  const script = readFileSync(scriptPath, 'utf8')
  assert.match(script, /linux-x64-cp312-cuda124/)
  assert.match(script, /nvcc/)
  assert.match(script, /CUDA 12\.4 toolchain/)
  assert.match(script, /placeholder/i)
  assert.match(script, /install_build_prerequisites\(\)/)
  assert.match(script, /copy_pure_wheels\(\)/)
  assert.match(script, /build_native_wheel\(\)/)
  assert.match(script, /WHEELHOUSE_NATIVE_SOURCES_ENV/)
  assert.match(script, /O_VOXEL_SOURCE_SUBDIR/)
  assert.match(script, /build_dir_for_package\(\)/)
  assert.match(script, /TORCH_CUDA_ARCH_LIST:-8\.0;8\.6;8\.9;9\.0/)
  assert.match(script, /WHEELHOUSE_NATTEN_CUDA_ARCH:-8\.9/)
  assert.match(script, /WHEELHOUSE_NATIVE_BUILD_TIMEOUT:-60m/)
  assert.match(script, /timeout "\$\{WHEELHOUSE_NATIVE_BUILD_TIMEOUT\}" python3 -m pip wheel/)
  assert.match(script, /native wheel build exceeded \$\{WHEELHOUSE_NATIVE_BUILD_TIMEOUT\}/)
  assert.match(script, /--no-build-isolation/)
  assert.match(script, /git clone --recurse-submodules/)
  assert.match(script, /git -C "\$\{source_dir\}" submodule update --init --recursive/)
  assert.match(script, /Missing native source directories/)
  assert.match(script, /Invalid native source directories/)
  assert.match(script, /Failed native wheel builds/)
  assert.match(script, /optional_native_packages=\(/)
  assert.match(script, /WHEELHOUSE_BUILD_STRICT_NATTEN/)
  assert.match(script, /Skipping optional NATTEN\/libnatten build/)
  assert.match(script, /strict NAF must be enabled only when natten\.HAS_LIBNATTEN is True/)
  assert.match(script, /finalize_wheelhouse_archive\(\)/)
  assert.match(recipe, /linux-x64-cp312-cuda124/)
  assert.match(recipe, /pixal3d-wheelhouse-v0\.1\.0-linux-x64-cp312-cuda124\.zip/)
  assert.match(recipe, /CUDA 12\.4 toolchain/)
  assert.match(recipe, /NATTEN_SOURCE_DIR/)
  assert.match(recipe, /copy the existing pure Python wheels/i)
  assert.match(recipe, /missing native source directories/i)
  assert.match(recipe, /native-sources\.linux-x64-cp312-cuda124\.env\.example/)
})

test('windows x64 cp312 cuda124 workflow is exact-stack and manifest-backed', () => {
  const manifest = JSON.parse(readFileSync(join(repoRoot, 'wheelhouse.manifest.json'), 'utf8'))
  const workflowPath = join(repoRoot, '.github', 'workflows', 'wheelhouse-windows-x64-cp312-cuda124.yml')
  const cp311WorkflowPath = join(repoRoot, '.github', 'workflows', 'wheelhouse-windows-x64-cp311-cuda124.yml')
  const scriptPath = join(repoRoot, 'tools', 'wheelhouse', 'build-windows-x64-cp312-cuda124.ps1')
  const recipe = readFileSync(join(repoRoot, 'tools', 'wheelhouse', 'README.md'), 'utf8')

  const windowsAsset = manifest.assets.find((asset) => asset.id === 'windows-x64-cp312-cuda124')
  assert.ok(windowsAsset)
  assert.equal(windowsAsset.filename, 'pixal3d-wheelhouse-v0.1.0-windows-x64-cp312-cuda124.zip')
  assert.equal(existsSync(workflowPath), true)
  assert.equal(existsSync(cp311WorkflowPath), true)
  assert.equal(existsSync(scriptPath), true)

  const workflow = readFileSync(workflowPath, 'utf8')
  assert.match(workflow, /windows-latest/)
  assert.match(workflow, /windows-x64-cp312-cuda124/)
  assert.doesNotMatch(workflow, /WHEELHOUSE_WINDOWS_ALLOW_INCOMPLETE: '1'/)
  assert.match(workflow, /Do not add to wheelhouse\.manifest\.json until the candidate archive is uploaded to the pinned release and checksum-pinned/)
  assert.match(workflow, /build-windows-x64-cp312-cuda124\.ps1/)
  assert.match(workflow, /pixal3d-wheelhouse-\$\{\{ env\.WHEELHOUSE_LANE \}\}-candidate/)
  const cp311Workflow = readFileSync(cp311WorkflowPath, 'utf8')
  assert.match(cp311Workflow, /windows-x64-cp311-cuda124/)
  assert.match(cp311Workflow, /python-version: '3\.11'/)
  assert.match(cp311Workflow, /Modly python-build-standalone 3\.11\.9 install contract/)

  const script = readFileSync(scriptPath, 'utf8')
  assert.match(script, /windows-x64-cp312-cuda124/)
  assert.match(script, /windows-x64-cp311-cuda124/)
  assert.match(script, /\$PythonTag = if \(\$Lane -match "cp311"\)/)
  assert.match(script, /cu124torch2\.6-\$PythonTag-\$PythonTag-win_amd64/)
  for (const wheel of ['flex_gemm_ap', 'cumesh_vb', 'o_voxel_vb_ap', 'drtk', 'flash_attn', 'nvdiffrast', 'nvdiffrec_render']) {
    assert.match(script, new RegExp(wheel))
  }
  assert.match(script, /Invoke-WebRequest/)
  assert.match(script, /nvdiffrast-0\.4\.0\+cu124torch2\.6-\$PythonTag-\$PythonTag-win_amd64\.whl/)
  assert.match(script, /nvdiffrec_render-0\.0\.1\+cu124torch2\.6-\$PythonTag-\$PythonTag-win_amd64\.whl/)
  assert.match(script, /Repairing Windows pure-wheel dependency metadata/)
  assert.match(script, /Repairing Windows external wheel dist-info directory names/)
  assert.match(script, /rename_dist_info_prefix\(wheel_by_prefix\(wheelhouse, "nvdiffrec_render"\), "nvdiffrec-render-", "nvdiffrec_render-"\)/)
  assert.match(script, /\$downloaded = @\(\)\nforeach \(\$wheel in \$externalWheels\) \{\n\s+\$destination = Join-Path \$wheelhouseDir \$wheel\.Filename/s)
  assert.match(script, /o-voxel-vb-ap==0\.0\.1; platform_system == "Windows"/)
  assert.match(script, /cumesh-vb==1\.0; platform_system == "Windows"/)
  assert.match(script, /flex-gemm-ap==1\.0\.0; platform_system == "Windows"/)
  assert.match(script, /nvdiffrec-render==0\.0\.1; platform_system == "Windows"/)
  assert.match(script, /natten; platform_system != "Windows"/)
  assert.match(script, /\$unresolvedRequired = @\(\)/)
  assert.match(script, /strict NAF requires natten\.HAS_LIBNATTEN == True/)
  assert.match(script, /WINDOWS-CANDIDATE\.json/)
  assert.match(script, /Compress-Archive/)
  assert.match(script, /Do not add to wheelhouse\.manifest\.json until this candidate archive is uploaded to the pinned release and checksum-pinned/)

  assert.match(recipe, /Windows x64 lane/)
  assert.doesNotMatch(recipe, /Windows lane is intentionally \*\*not declared as supported\*\*/)
  assert.match(recipe, /python-build-standalone `3\.11\.9`/)
  assert.match(recipe, /windows-x64-cp311-cuda124` is the primary GitHub-install lane/)
  assert.match(recipe, /WINDOWS-CANDIDATE\.json/)
  assert.match(recipe, /`nvdiffrast`/)
  assert.match(recipe, /`nvdiffrec_render`/)
})

test('windows cp311 cuda124 NATTEN candidate workflow is manual, exact-stack, and artifact-only', () => {
  const workflowPath = join(repoRoot, '.github', 'workflows', 'natten-windows-x64-cp311-cuda124-candidate.yml')
  const workflow = readFileSync(workflowPath, 'utf8')
  const recipe = readFileSync(join(repoRoot, 'tools', 'wheelhouse', 'README.md'), 'utf8')
  const manifest = readFileSync(join(repoRoot, 'wheelhouse.manifest.json'), 'utf8')

  assert.equal(existsSync(workflowPath), true)
  assert.match(workflow, /workflow_dispatch:/)
  assert.match(workflow, /cuda_arch_list:/)
  assert.match(workflow, /default: '7\.5;8\.6;8\.9'/)
  assert.match(workflow, /windows-latest/)
  assert.match(workflow, /NATTEN v0\.21\.0 Windows x64 cp311 th26 cu124 candidate/)
  assert.match(workflow, /Build natten v0\.21\.0 windows-x64-cp311-th26-cu124 candidate/)
  assert.match(workflow, /python-version: \$\{\{ env\.PYTHON_VERSION \}\}/)
  assert.match(workflow, /PYTHON_VERSION: '3\.11'/)
  assert.match(workflow, /TORCH_VERSION: 2\.6\.0\+cu124/)
  assert.match(workflow, /TORCHVISION_VERSION: 0\.21\.0\+cu124/)
  assert.match(workflow, /PYTORCH_CUDA_INDEX: https:\/\/download\.pytorch\.org\/whl\/cu124/)
  assert.match(workflow, /CUDA_VERSION: 12\.4\.1/)
  assert.match(workflow, /Jimver\/cuda-toolkit@v0\.2\.35/)
  assert.doesNotMatch(workflow, /sub-packages: '\["nvcc", "cudart-dev"\]'/)
  assert.match(workflow, /ilammy\/msvc-dev-cmd@v1/)
  assert.match(workflow, /arch: x64/)
  assert.match(workflow, /\n\s+cl\n\s+nvcc --version/)
  assert.match(workflow, /git clone --branch \$env:NATTEN_VERSION --depth 1 --recurse-submodules https:\/\/github\.com\/SHI-Labs\/NATTEN\.git/)
  assert.match(workflow, /Patch NATTEN MSVC-incompatible warning flags/)
  assert.match(workflow, /-Xcompiler=-Wconversion/)
  assert.match(workflow, /-Xcompiler=-fno-strict-aliasing/)
  assert.match(workflow, /Expected to patch NATTEN GCC-only compiler flags for MSVC/)
  assert.match(workflow, /Patch vendored CUTLASS C\+\+17 detection for MSVC/)
  assert.match(workflow, /third_party\\cutlass\\include\\cutlass\\platform\\platform\.h/)
  assert.match(workflow, /_MSVC_LANG/)
  assert.match(workflow, /Expected CUTLASS platform\.h C\+\+17 detection guard was not found/)
  assert.match(workflow, /Patch NATTEN MSVC CUDA alternative-token checks/)
  assert.match(workflow, /csrc\\include\\natten\\helpers\.h/)
  assert.match(workflow, /not x\.is_sparse\(\)/)
  assert.match(workflow, /!x\.is_sparse\(\)/)
  assert.match(workflow, /Expected NATTEN helpers\.h alternative-token check was not found/)
  assert.match(workflow, /NATTEN_VERSION: v0\.21\.0/)
  assert.doesNotMatch(workflow, /NATTEN_VERSION: v0\.17\.5/)
  assert.match(workflow, /\$env:TORCH_CUDA_ARCH_LIST = \$archList/)
  assert.match(workflow, /\$env:NATTEN_CUDA_ARCH = \$archList/)
  assert.match(workflow, /python -m pip wheel \. --no-build-isolation --no-deps --wheel-dir \$wheelDir/)
  assert.match(workflow, /python -m pip install --force-reinstall --no-deps \$wheelPath/)
  assert.match(workflow, /\$verifyScript = @'/)
  assert.match(workflow, /import natten/)
  assert.match(workflow, /import torch/)
  assert.match(workflow, /"__version__": getattr\(natten, "__version__", None\)/)
  assert.match(workflow, /HAS_LIBNATTEN/)
  assert.match(workflow, /Found no NVIDIA driver/)
  assert.match(workflow, /"verification_blocked": "no_nvidia_driver"/)
  assert.match(workflow, /\$verificationRaw = \$verifyScript \| python -/)
  assert.match(workflow, /\$verificationRaw \| Set-Content -Path \$verificationPath/)
  assert.match(workflow, /NATTEN-0\.21\.0-CANDIDATE\.json/)
  assert.match(workflow, /verification\.json/)
  assert.match(workflow, /actions\/upload-artifact@v4/)
  assert.match(workflow, /natten-0\.21\.0-windows-x64-cp311-cuda124-candidate/)
  assert.match(workflow, /do not update wheelhouse\.manifest\.json from this workflow/i)
  assert.doesNotMatch(workflow, /upload-release-asset/i)
  assert.doesNotMatch(workflow, /wheelhouse-v0\.1\.0/)
  assert.doesNotMatch(workflow, /release\.tag/)

  assert.match(recipe, /natten-windows-x64-cp311-cuda124-candidate\.yml/)
  assert.match(recipe, /SHI-Labs\/NATTEN` tag `v0\.21\.0`/)
  assert.match(recipe, /closest Windows port to upstream `requirements_th26_cu124\.txt`/)
  assert.match(recipe, /candidate-only/i)
  assert.match(recipe, /must not update `wheelhouse\.manifest\.json`/)
  assert.match(recipe, /cuda_arch_list` with default `7\.5;8\.6;8\.9`/)
  assert.match(recipe, /NATTEN-0\.21\.0-CANDIDATE\.json/)
  assert.match(recipe, /natten\.__version__/)
  assert.match(recipe, /verify `natten\.HAS_LIBNATTEN == True`/)
  assert.match(recipe, /MSVC-incompatible GCC warning flags/)
  assert.match(recipe, /CUTLASS C\+\+17 detection/)
  assert.match(recipe, /alternative token `not`/)

  assert.ok(!manifest.includes('natten-windows-x64-cp311-cuda124-candidate'))
  assert.ok(!manifest.includes('v0.17.5'))
  assert.ok(!manifest.includes('v0.21.0'))
})

test('Blackwell sm120 investigation remains probe-only and outside published wheelhouse lanes', () => {
  const workflowPath = join(repoRoot, '.github', 'workflows', 'blackwell-windows-x64-cp311-cuda128-probe.yml')
  const workflow = readFileSync(workflowPath, 'utf8')
  const docs = readFileSync(join(repoRoot, 'tools', 'wheelhouse', 'BLACKWELL-SM120.md'), 'utf8')
  const recipe = readFileSync(join(repoRoot, 'tools', 'wheelhouse', 'README.md'), 'utf8')
  const manifest = readFileSync(join(repoRoot, 'wheelhouse.manifest.json'), 'utf8')

  assert.equal(existsSync(workflowPath), true)
  assert.match(workflow, /workflow_dispatch:/)
  assert.match(workflow, /Blackwell Windows x64 cp311 cuda128 probe/)
  assert.match(workflow, /Probe Blackwell sm_120 prerequisites only/)
  assert.match(workflow, /default: '12\.8\.1'/)
  assert.match(workflow, /default: '2\.7\.1\+cu128'/)
  assert.match(workflow, /PYTORCH_CUDA_INDEX: https:\/\/download\.pytorch\.org\/whl\/cu128/)
  assert.match(workflow, /--list-gpu-arch/)
  assert.match(workflow, /--list-gpu-code/)
  assert.match(workflow, /compute_120/)
  assert.match(workflow, /sm_120/)
  assert.match(workflow, /BLACKWELL-SM120-PROBE\.json/)
  assert.match(workflow, /probe_only = \$true/)
  assert.match(workflow, /Do not add to wheelhouse\.manifest\.json/)
  assert.match(workflow, /actions\/upload-artifact@v4/)
  assert.doesNotMatch(workflow, /upload-release-asset/i)

  assert.match(docs, /RTX 5090 as compute capability `12\.0`/)
  assert.match(docs, /not\*\* a supported Pixal3D wheelhouse lane/i)
  assert.match(docs, /CUDA 12\.4\.1 `nvcc` documentation lists.*`compute_90` \/ `sm_90`/s)
  assert.match(docs, /does not list `compute_120` \/ `sm_120`/)
  assert.match(docs, /rebuilding only NATTEN for `sm_120` would not prove/)
  for (const wheel of ['flex_gemm_ap', 'cumesh_vb', 'o_voxel_vb_ap', 'drtk', 'flash_attn', 'nvdiffrast', 'nvdiffrec_render']) {
    assert.match(docs, new RegExp(wheel))
  }
  assert.match(docs, /Until those conditions are met, RTX 5090 \/ Blackwell remains experimental and unsupported/)
  assert.match(recipe, /BLACKWELL-SM120\.md/)
  assert.match(recipe, /do not mark RTX 5090 supported/i)
  assert.ok(!manifest.includes('cuda128'))
  assert.ok(!manifest.includes('blackwell'))
  assert.ok(!manifest.includes('sm120'))
})

test('Blackwell Windows wheelhouse candidate is exact-stack and artifact-only', () => {
  const workflowPath = join(repoRoot, '.github', 'workflows', 'wheelhouse-windows-x64-cp311-cuda128-blackwell-candidate.yml')
  const scriptPath = join(repoRoot, 'tools', 'wheelhouse', 'build-windows-x64-cp311-cuda128-blackwell.ps1')
  const workflow = readFileSync(workflowPath, 'utf8')
  const script = readFileSync(scriptPath, 'utf8')
  const docs = readFileSync(join(repoRoot, 'tools', 'wheelhouse', 'BLACKWELL-SM120.md'), 'utf8')
  const recipe = readFileSync(join(repoRoot, 'tools', 'wheelhouse', 'README.md'), 'utf8')
  const manifest = readFileSync(join(repoRoot, 'wheelhouse.manifest.json'), 'utf8')

  assert.equal(existsSync(workflowPath), true)
  assert.equal(existsSync(scriptPath), true)
  assert.match(workflow, /workflow_dispatch:/)
  assert.match(workflow, /windows-x64-cp311-cuda128-blackwell/)
  assert.match(workflow, /default: '12\.0'/)
  assert.match(workflow, /default: '12\.8\.1'/)
  assert.match(workflow, /default: 'v0\.21\.6'/)
  assert.match(workflow, /TORCH_VERSION: 2\.7\.1\+cu128/)
  assert.match(workflow, /TORCHVISION_VERSION: 0\.22\.1\+cu128/)
  assert.match(workflow, /PYTORCH_CUDA_INDEX: https:\/\/download\.pytorch\.org\/whl\/cu128/)
  assert.match(workflow, /NATTEN_PACKAGE_VERSION: '0\.21\.6'/)
  assert.match(workflow, /git config --global core\.longpaths true/)
  assert.match(workflow, /git clone --branch \$env:NATTEN_VERSION --depth 1 --recurse-submodules https:\/\/github\.com\/SHI-Labs\/NATTEN\.git/)
  assert.match(workflow, /Patch NATTEN MSVC-incompatible GCC warning flags recursively/)
  assert.match(workflow, /-Xcompiler=-Wconversion/)
  assert.match(workflow, /-Xcompiler=-fno-strict-aliasing/)
  assert.match(workflow, /-Xcompiler -Wall/)
  assert.match(workflow, /Expected to patch at least one NATTEN file containing MSVC-incompatible GCC compiler flags/)
  assert.match(workflow, /\$text = Get-Content -Raw -Path \$file\.FullName/)
  assert.match(workflow, /if \(\$null -eq \$text\)/)
  assert.match(workflow, /Skipping empty patchable file/)
  assert.match(workflow, /\$newText = \$text\.ToString\(\)/)
  assert.match(workflow, /#if \(201703L <=__cplusplus\)/)
  assert.match(workflow, /#if \(201703L <= __cplusplus\) \|\| \(defined\(_MSVC_LANG\) && _MSVC_LANG >= 201703L\)/)
  assert.match(workflow, /\$env:TORCH_CUDA_ARCH_LIST = \$env:CUDA_ARCH_LIST/)
  assert.match(workflow, /\$env:NATTEN_CUDA_ARCH = \$env:CUDA_ARCH_LIST/)
  assert.match(workflow, /natten-\$env:NATTEN_PACKAGE_VERSION-\*-win_amd64\.whl/)
  assert.match(workflow, /build-windows-x64-cp311-cuda128-blackwell\.ps1 -NattenWheelPath[\s\S]*-NattenVersion \$env:NATTEN_PACKAGE_VERSION/)
  assert.match(workflow, /actions\/upload-artifact@v4/)
  assert.doesNotMatch(workflow, /upload-release-asset/i)

  assert.match(script, /windows-x64-cp311-cuda128-blackwell/)
  assert.match(script, /cu128torch2\.7-cp311-cp311-win_amd64/)
  assert.match(script, /\$exactStackPattern = "\$\{CudaTag\}torch\$\{TorchMinor\}-\$\{PythonTag\}-\$\{PythonTag\}-win_amd64"/)
  assert.doesNotMatch(script, /\$CudaTag`torch\$TorchMinor/)
  assert.match(script, /\[string\]\$NattenVersion = "0\.21\.6"/)
  assert.match(script, /NattenWheelPath/)
  assert.match(script, /missing NATTEN wheel path; Blackwell runtime candidate requires natten==\$NattenVersion/)
  assert.match(script, /Unexpected Blackwell NATTEN wheel filename for natten \$\{NattenVersion\}:/)
  assert.match(script, /WINDOWS-BLACKWELL-CANDIDATE\.json/)
  assert.match(script, /candidate_complete_unvalidated/)
  assert.match(script, /Repairing Windows Blackwell external wheel dist-info directory names/)
  assert.match(script, /rename_dist_info_prefix\(wheel_by_prefix\(wheelhouse, "nvdiffrec_render"\), "nvdiffrec-render-", "nvdiffrec_render-"\)/)
  assert.match(script, /\$downloaded = @\(\)\nforeach \(\$wheel in \$externalWheels\) \{\n\s+\$destination = Join-Path \$wheelhouseDir \$wheel\.Filename/s)
  for (const wheel of ['flex_gemm_ap', 'cumesh_vb', 'o_voxel_vb_ap', 'drtk', 'flash_attn', 'nvdiffrast', 'nvdiffrec_render']) {
    assert.match(script, new RegExp(`${wheel}-[\\s\\S]*cu128torch2\\.7-cp311-cp311-win_amd64`))
  }
  assert.match(script, /natten==\{sys\.argv\[2\]\}/)
  assert.match(script, /Do not add to wheelhouse\.manifest\.json|Candidate-only/)

  assert.match(docs, /Candidate wheelhouse workflow/)
  assert.match(docs, /torch==2\.7\.1\+cu128/)
  assert.match(docs, /TORCH_CUDA_ARCH_LIST=12\.0/)
  assert.match(docs, /NATTEN v0\.21\.6/)
  assert.match(docs, /post-`v0\.21\.0` Blackwell fixes/)
  assert.match(docs, /run `27070460013` confirmed CUDA 12\.8 generated `compute_120` \/ `sm_120`/)
  assert.match(docs, /removes those GCC-only flags recursively/)
  assert.match(docs, /run `27070868300` progressed past long paths and GCC-only flag removal/)
  assert.match(docs, /patches that exact guard to also accept `_MSVC_LANG >= 201703L`/)
  assert.match(docs, /run `27071874204` progressed further into NATTEN `v0\.21\.0` Blackwell backward kernels/)
  assert.match(docs, /candidate now tests `v0\.21\.6` before adding local kernel patches/)
  assert.match(docs, /run `27075519968` failed before compiling NATTEN/)
  assert.match(docs, /inline `\[string\]\(\.\.\.\)` cast was not sufficient/)
  assert.match(docs, /explicitly checks `\$null`/)
  assert.match(docs, /run `27076133932` successfully built `natten-0\.21\.6-cp311-cp311-win_amd64\.whl`/)
  assert.match(docs, /uses `\$\{NattenVersion\}:`/)
  assert.match(docs, /run `27086897007` reached candidate assembly/)
  assert.match(docs, /interpreted PowerShell backtick-`t` as a tab/)
  assert.match(docs, /`\$\{CudaTag\}torch\$\{TorchMinor\}-\.\.\.`/)
  assert.match(docs, /must not update `wheelhouse\.manifest\.json`/)
  assert.match(recipe, /wheelhouse-windows-x64-cp311-cuda128-blackwell-candidate\.yml/)
  assert.match(recipe, /candidate-only/)
  assert.ok(!manifest.includes('windows-x64-cp311-cuda128-blackwell'))
})

test('linux x64 native source refs are immutable and confidence-documented', () => {
  const env = readFileSync(new URL('../tools/wheelhouse/native-sources.linux-x64-cp312-cuda124.env.example', import.meta.url), 'utf8')

  for (const name of ['NATTEN', 'O_VOXEL', 'CUMESH', 'FLEX_GEMM', 'NVDIFFRAST', 'NVDIFFREC_RENDER']) {
    assert.match(env, new RegExp(`${name}_SOURCE_URL=https://`))
    assert.match(env, new RegExp(`${name}_SOURCE_REF=[0-9a-f]{40}`))
  }
  assert.match(env, /O_VOXEL_SOURCE_SUBDIR=o-voxel/)
  assert.match(env, /Confidence legend:/)
  assert.match(env, /WHEELHOUSE_ALLOW_SOURCE_FETCH=1/)
})

test('linux x64 build recipe reuses pure wheels and fails clearly when native sources are missing', () => {
  const scriptPath = join(repoRoot, 'tools', 'wheelhouse', 'build-linux-x64-cp312-cuda124.sh')
  const tmp = mkdtempSync(join(tmpdir(), 'pixal3d-wheelhouse-'))
  const fakeBin = join(tmp, 'bin')

  mkdirSync(fakeBin, { recursive: true })
  writeFileSync(join(fakeBin, 'nvcc'), '#!/usr/bin/env bash\necho "Cuda compilation tools, release 12.4, V12.4.1"\n')
  writeFileSync(join(fakeBin, 'python3'), '#!/usr/bin/env bash\ncat >/dev/null\necho cp312\n')
  chmodSync(join(fakeBin, 'nvcc'), 0o755)
  chmodSync(join(fakeBin, 'python3'), 0o755)

  const result = spawnSync('bash', [scriptPath], {
    cwd: repoRoot,
    encoding: 'utf8',
    env: {
      ...process.env,
      PATH: `${fakeBin}:${process.env.PATH}`,
      WHEELHOUSE_BUILD_ROOT: tmp,
      WHEELHOUSE_SKIP_PREREQ_INSTALL: '1',
    },
  })

  assert.equal(result.status, 1)
  assert.match(result.stderr, /Missing native source directories for linux-x64-cp312-cuda124/)
  assert.match(result.stderr, /Using TORCH_CUDA_ARCH_LIST=8\.0;8\.6;8\.9;9\.0/)
  assert.match(result.stderr, /Using NATTEN_CUDA_ARCH=8\.9/)
  assert.match(result.stderr, /Using WHEELHOUSE_NATIVE_BUILD_TIMEOUT=60m/)
  for (const nativeName of ['o_voxel', 'cumesh', 'flex_gemm', 'nvdiffrast', 'nvdiffrec_render']) {
    assert.match(result.stderr, new RegExp(nativeName))
  }
  assert.match(result.stderr, /Skipping optional NATTEN\/libnatten build/)
  assert.doesNotMatch(result.stderr, /Missing native source directories[\s\S]*natten=/)
  assert.match(result.stderr, /No placeholder wheelhouse archive will be created/i)

  const copiedWheels = readdirSync(join(tmp, 'wheelhouse')).sort()
  assert.deepEqual(copiedWheels, [
    'moge-2.0.0+modly-py3-none-any.whl',
    'naf-0.1.0+modly-py3-none-any.whl',
    'pipeline-1.0.0+modly-py3-none-any.whl',
    'pixal3d_core-0.1.0+modly-py3-none-any.whl',
    'utils3d-1.3+modly.headless-py3-none-any.whl',
  ])
  assert.equal(existsSync(join(tmp, 'dist', 'wheelhouse', 'pixal3d-wheelhouse-v0.1.0-linux-x64-cp312-cuda124.zip')), false)
  rmSync(tmp, { recursive: true, force: true })
})

test('manifest validation rejects mutable releases, missing checksums, and unsafe cache roots before selection', () => {
  const result = runPython(`
import copy, json
from pathlib import Path
from modly_wheelhouse import WheelhouseError, load_manifest, validate_manifest
base = load_manifest(Path('wheelhouse.manifest.json'))
cases = {}
for name, mutate in {
    'mutable_latest': lambda m: m['release'].__setitem__('tag', 'latest'),
    'missing_hash': lambda m: m['assets'][0].__setitem__('sha256', ''),
    'unsafe_cache': lambda m: m['cache'].__setitem__('root', '../outside'),
}.items():
    manifest = copy.deepcopy(base)
    mutate(manifest)
    try:
        validate_manifest(manifest)
    except WheelhouseError as exc:
        cases[name] = exc.code
print(json.dumps(cases, sort_keys=True))
`)

  assert.deepEqual(result, {
    missing_hash: 'missing_hash',
    mutable_latest: 'mutable_release_tag',
    unsafe_cache: 'unsafe_cache_path',
  })
})

test('selector rejects unsupported and ambiguous runtime lanes without download intent', () => {
  const result = runPython(`
import copy, json
from pathlib import Path
from modly_wheelhouse import WheelhouseError, load_manifest, select_asset, validate_manifest
base = load_manifest(Path('wheelhouse.manifest.json'))
validate_manifest(base)
out = {}
for name, manifest, evidence in [
    ('unsupported', base, {'os':'linux','arch':'x64','python_tag':'cp312','accelerator_lane':'cuda125'}),
    ('ambiguous', copy.deepcopy(base), {'os':'linux','arch':'aarch64','python_tag':'cp312','accelerator_lane':'cuda124'}),
]:
    if name == 'ambiguous':
        manifest['assets'].append(copy.deepcopy(manifest['assets'][0]))
        manifest['assets'][1]['id'] = 'duplicate-linux-aarch64-cp312-cuda124'
    try:
        select_asset(manifest, evidence)
    except WheelhouseError as exc:
        out[name] = {'code': exc.code, 'downloads_started': exc.observation['downloads_started'], 'installs_started': exc.observation['installs_started']}
print(json.dumps(out, sort_keys=True))
`)

  assert.deepEqual(result, {
    ambiguous: { code: 'ambiguous_lane', downloads_started: false, installs_started: false },
    unsupported: { code: 'unsupported_lane', downloads_started: false, installs_started: false },
  })
})

test('release downloader maps GitHub 401 and 403 responses to auth_required before install intent', () => {
  const result = runPython(`
import json
import urllib.error
from pathlib import Path
from modly_wheelhouse import WheelhouseError, _download_url

out = {}
class RaisingUrlopen:
    def __init__(self, code):
        self.code = code
    def __call__(self, *_args, **_kwargs):
        raise urllib.error.HTTPError('https://example.invalid/asset.zip', self.code, 'blocked', {}, None)

import modly_wheelhouse
original = modly_wheelhouse.urllib.request.urlopen
try:
    for code in (401, 403):
        modly_wheelhouse.urllib.request.urlopen = RaisingUrlopen(code)
        try:
            _download_url('https://example.invalid/asset.zip', Path('/tmp/unused-wheelhouse.zip'))
        except WheelhouseError as exc:
            out[str(code)] = {'code': exc.code, 'downloads_started': exc.observation['downloads_started'], 'installs_started': exc.observation['installs_started']}
finally:
    modly_wheelhouse.urllib.request.urlopen = original
print(json.dumps(out, sort_keys=True))
`)

  assert.deepEqual(result, {
    401: { code: 'auth_required', downloads_started: false, installs_started: false },
    403: { code: 'auth_required', downloads_started: false, installs_started: false },
  })
})

test('setup dry run reports wheelhouse manifest metadata without downloading or installing', () => {
  const result = runPython(`
import json
import setup
print(json.dumps(setup.run_setup(['--json'])))
`)

  assert.equal(result.status, 'dry_run')
  assert.equal(result.downloads_started, false)
  assert.equal(result.installs_started, false)
  assert.deepEqual(result.wheelhouse_manifest, {
    status: 'available',
    release_tag: 'wheelhouse-v0.1.0',
    wheelhouse_version: '0.1.0',
    asset_count: 4,
  })
})

test('prepare_wheelhouse downloads one selected asset, verifies sha256, safely extracts it, and reports JSON observations', () => {
  const result = runPython(`
import copy, hashlib, json, tempfile, zipfile
from pathlib import Path
from modly_wheelhouse import load_manifest, prepare_wheelhouse, validate_manifest

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    asset_zip = tmp_path / 'asset.zip'
    with zipfile.ZipFile(asset_zip, 'w') as archive:
        archive.writestr('wheelhouse/pkg-1.0.0-py3-none-any.whl', b'fake-wheel')
    manifest = load_manifest(Path('wheelhouse.manifest.json'))
    manifest['assets'][0]['filename'] = 'asset.zip'
    manifest['assets'][0]['compression'] = 'zip'
    manifest['assets'][0]['size_bytes'] = asset_zip.stat().st_size
    manifest['assets'][0]['sha256'] = hashlib.sha256(asset_zip.read_bytes()).hexdigest()
    validate_manifest(manifest)
    calls = []
    def downloader(url, destination):
        calls.append({'url': url, 'destination': str(destination.relative_to(tmp_path))})
        destination.write_bytes(asset_zip.read_bytes())
        return destination.stat().st_size
    observation = prepare_wheelhouse(
        manifest,
        {'os':'linux','arch':'aarch64','python_tag':'cp312','accelerator_lane':'cuda124'},
        tmp_path,
        downloader=downloader,
    )
    extracted = Path(observation['wheelhouse_path'])
    print(json.dumps({
        'status': observation['status'],
        'selected_asset': observation['selected_asset'],
        'cache_hit': observation['cache_hit'],
        'downloaded': observation['downloaded'],
        'sha256_verified': observation['sha256_verified'],
        'downloads_started': observation['downloads_started'],
        'installs_started': observation['installs_started'],
        'bytes_downloaded': observation['bytes_downloaded'],
        'release_tag': observation['release_tag'],
        'calls': calls,
        'wheel_exists': (extracted / 'pkg-1.0.0-py3-none-any.whl').exists(),
        'wheelhouse_dir': extracted.name,
        'contained': tmp_path in extracted.parents,
    }, sort_keys=True))
`)

  assert.deepEqual(result, {
    bytes_downloaded: result.bytes_downloaded,
    cache_hit: false,
    calls: [{ destination: '.modly/cache/wheelhouse/pixal3d/0.1.0/linux-aarch64-cp312-cuda124/download-archive.zip', url: 'https://github.com/DrHepa/modly-pixal3d-extension/releases/download/wheelhouse-v0.1.0/asset.zip' }],
    contained: true,
    downloaded: true,
    downloads_started: true,
    installs_started: false,
    release_tag: 'wheelhouse-v0.1.0',
    selected_asset: 'linux-aarch64-cp312-cuda124',
    sha256_verified: true,
    status: 'ready',
    wheel_exists: true,
    wheelhouse_dir: 'wheelhouse',
  })
  assert.ok(result.bytes_downloaded > 0)
})

test('prepare_wheelhouse reuses a verified extracted cache without network access', () => {
  const result = runPython(`
import copy, hashlib, json, tempfile, zipfile
from pathlib import Path
from modly_wheelhouse import load_manifest, prepare_wheelhouse, validate_manifest

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    asset_zip = tmp_path / 'asset.zip'
    with zipfile.ZipFile(asset_zip, 'w') as archive:
        archive.writestr('wheelhouse/cached-1.0.0-py3-none-any.whl', b'cached-wheel')
    manifest = load_manifest(Path('wheelhouse.manifest.json'))
    manifest['assets'][0]['filename'] = 'asset.zip'
    manifest['assets'][0]['compression'] = 'zip'
    manifest['assets'][0]['size_bytes'] = asset_zip.stat().st_size
    manifest['assets'][0]['sha256'] = hashlib.sha256(asset_zip.read_bytes()).hexdigest()
    validate_manifest(manifest)
    first = prepare_wheelhouse(manifest, {'os':'linux','arch':'aarch64','python_tag':'cp312','accelerator_lane':'cuda124'}, tmp_path, downloader=lambda _url, destination: destination.write_bytes(asset_zip.read_bytes()))
    def forbidden_downloader(_url, _destination):
        raise AssertionError('network should not be used for verified extracted cache')
    second = prepare_wheelhouse(manifest, {'os':'linux','arch':'aarch64','python_tag':'cp312','accelerator_lane':'cuda124'}, tmp_path, downloader=forbidden_downloader)
    print(json.dumps({
        'first_status': first['status'],
        'second_status': second['status'],
        'cache_hit': second['cache_hit'],
        'downloaded': second['downloaded'],
        'downloads_started': second['downloads_started'],
        'sha256_verified': second['sha256_verified'],
        'same_path': first['wheelhouse_path'] == second['wheelhouse_path'],
    }, sort_keys=True))
`)

  assert.deepEqual(result, {
    cache_hit: true,
    downloaded: false,
    downloads_started: false,
    first_status: 'ready',
    same_path: true,
    second_status: 'ready',
    sha256_verified: true,
  })
})

test('prepare_wheelhouse blocks checksum mismatch and unsafe archive paths before install intent', () => {
  const result = runPython(`
import copy, hashlib, json, tempfile, zipfile
from pathlib import Path
from modly_wheelhouse import WheelhouseError, load_manifest, prepare_wheelhouse, validate_manifest

def configured_manifest(asset_path):
    manifest = load_manifest(Path('wheelhouse.manifest.json'))
    manifest['assets'][0]['filename'] = asset_path.name
    manifest['assets'][0]['compression'] = 'zip'
    manifest['assets'][0]['size_bytes'] = asset_path.stat().st_size
    manifest['assets'][0]['sha256'] = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    validate_manifest(manifest)
    return manifest

out = {}
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    good_zip = tmp_path / 'good.zip'
    with zipfile.ZipFile(good_zip, 'w') as archive:
        archive.writestr('wheelhouse/pkg.whl', b'ok')
    manifest = configured_manifest(good_zip)
    bad_bytes = b'not-the-asset'
    try:
        prepare_wheelhouse(manifest, {'os':'linux','arch':'aarch64','python_tag':'cp312','accelerator_lane':'cuda124'}, tmp_path, downloader=lambda _url, destination: destination.write_bytes(bad_bytes))
    except WheelhouseError as exc:
        out['checksum'] = {'code': exc.code, 'downloads_started': exc.observation['downloads_started'], 'installs_started': exc.observation['installs_started']}

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    unsafe_zip = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(unsafe_zip, 'w') as archive:
        archive.writestr('../evil.whl', b'evil')
    manifest = configured_manifest(unsafe_zip)
    try:
        prepare_wheelhouse(manifest, {'os':'linux','arch':'aarch64','python_tag':'cp312','accelerator_lane':'cuda124'}, tmp_path, downloader=lambda _url, destination: destination.write_bytes(unsafe_zip.read_bytes()))
    except WheelhouseError as exc:
        out['unsafe'] = {'code': exc.code, 'downloads_started': exc.observation['downloads_started'], 'installs_started': exc.observation['installs_started']}
print(json.dumps(out, sort_keys=True))
`)

  assert.deepEqual(result, {
    checksum: { code: 'checksum_mismatch', downloads_started: true, installs_started: false },
    unsafe: { code: 'unsafe_archive_path', downloads_started: true, installs_started: false },
  })
})

test('verified vendored fallback resolves only lane-compatible hashed wheels', () => {
  const result = runPython(`
import copy, hashlib, json, tempfile
from pathlib import Path
from modly_wheelhouse import WheelhouseError, load_manifest, resolve_verified_fallback, validate_manifest

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    wheel_defs = {
        'native-1.0.0-cp312-cp312-linux_aarch64.whl': b'native',
        'pure-1.0.0-py3-none-any.whl': b'pure',
    }
    manifest = load_manifest(Path('wheelhouse.manifest.json'))
    manifest['fallback']['vendored_wheels'] = 'wheels/'
    manifest['fallback']['wheels'] = []
    for filename, content in wheel_defs.items():
        path = wheels / filename
        path.write_bytes(content)
        manifest['fallback']['wheels'].append({'filename': filename, 'size_bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
    validate_manifest(manifest)
    ok = resolve_verified_fallback(manifest, tmp_path, {'os':'linux','arch':'aarch64','python_tag':'cp312','accelerator_lane':'cuda124'})
    missing = copy.deepcopy(manifest)
    missing['fallback']['wheels'][0]['sha256'] = '0' * 64
    incompatible = copy.deepcopy(manifest)
    incompatible['fallback']['wheels'][0]['filename'] = 'native-1.0.0-cp310-cp310-linux_x86_64.whl'
    out = {'ok_status': ok['status'], 'wheel_count': ok['wheel_count'], 'sha256_verified': ok['sha256_verified']}
    for name, candidate in [('bad_hash', missing), ('bad_lane', incompatible)]:
        try:
            resolve_verified_fallback(candidate, tmp_path, {'os':'linux','arch':'aarch64','python_tag':'cp312','accelerator_lane':'cuda124'})
        except WheelhouseError as exc:
            out[name] = {'code': exc.code, 'downloads_started': exc.observation['downloads_started'], 'installs_started': exc.observation['installs_started']}
    print(json.dumps(out, sort_keys=True))
`)

  assert.deepEqual(result, {
    bad_hash: { code: 'fallback_checksum_mismatch', downloads_started: false, installs_started: false },
    bad_lane: { code: 'fallback_incompatible_lane', downloads_started: false, installs_started: false },
    ok_status: 'ready',
    sha256_verified: true,
    wheel_count: 2,
  })
})

test('setup dependency install uses a verified wheelhouse with no-index instead of PyPI fallback for native packages', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    venv_python = root / 'venv' / 'bin' / 'python'
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text('#!/usr/bin/env python3\\n', encoding='utf-8')
    verified = root / '.modly' / 'cache' / 'wheelhouse' / 'pixal3d' / '0.1.0' / 'linux-aarch64-cp312-cuda124' / 'extracted'
    verified.mkdir(parents=True)
    calls = []
    def fake_run(command, *, cwd):
        calls.append({'command': command, 'cwd': str(cwd)})
        return {'args': command, 'returncode': 0, 'stdout_tail': '{"HAS_LIBNATTEN": false, "importable": true, "ok": true, "torch_cuda_available": true, "torch_cuda_version": "13.0"}\\n' if command[1] == '-c' else '', 'stderr_tail': '', 'ok': True}
    setup._run_setup_command = fake_run
    result = setup._install_prepare_dependencies(root, wheelhouse_path=verified)
    print(json.dumps({'status': result['status'], 'wheelhouse': result['wheelhouse'], 'commands': [call['command'] for call in calls]}, sort_keys=True))
`)

  assert.equal(result.status, 'installed')
  assert.match(result.wheelhouse, /linux-aarch64-cp312-cuda124\/extracted$/)
  assert.deepEqual(result.commands[4].slice(2, 8), ['pip', 'install', '--no-index', '--no-deps', '--find-links', result.wheelhouse])
  assert.ok(result.commands[4].includes('pixal3d-core==0.1.0+modly'))
  assert.ok(!result.commands[4].includes('natten==0.21.0'))
})

test('setup dependency install uses Windows exact-stack package names for win_amd64 wheelhouses', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    venv_python = root / 'venv' / 'Scripts' / 'python.exe'
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text('', encoding='utf-8')
    wheelhouse = root / 'windows-wheelhouse'
    wheelhouse.mkdir()
    (wheelhouse / 'flex_gemm_ap-1.0.0+cu124torch2.6-cp312-cp312-win_amd64.whl').write_text('', encoding='utf-8')
    calls = []
    def fake_run(command, *, cwd):
        calls.append(command)
        return {'args': command, 'returncode': 0, 'stdout_tail': '{"HAS_LIBNATTEN": false, "importable": true, "ok": true, "torch_cuda_available": true, "torch_cuda_version": "13.0"}\\n' if command[1] == '-c' else '', 'stderr_tail': '', 'ok': True}
    setup._run_setup_command = fake_run
    result = setup._install_prepare_dependencies(root, wheelhouse_path=wheelhouse)
    print(json.dumps({'status': result['status'], 'packages': result['local_wheel_packages'], 'metadata_command': calls[3], 'install_command': calls[4]}, sort_keys=True))
`)

  assert.equal(result.status, 'installed')
  assert.ok(result.metadata_command.includes('triton-windows'))
  for (const pkg of ['o-voxel-vb-ap==0.0.1', 'cumesh-vb==1.0', 'flex-gemm-ap==1.0.0', 'drtk==0.1.0', 'flash-attn==2.8.3', 'nvdiffrec-render==0.0.1']) {
    assert.ok(result.packages.includes(pkg), `${pkg} missing`)
    assert.ok(result.install_command.includes(pkg), `${pkg} not installed`)
  }
  assert.ok(!result.packages.includes('o-voxel==0.0.1'))
  assert.ok(!result.packages.includes('flex-gemm==1.0.0'))
})

test('setup runtime CUDA probe uses Windows exact-stack import modules', () => {
  const setupSource = readFileSync(join(repoRoot, 'setup.py'), 'utf8')
  assert.match(setupSource, /def import_with_torch_compile_disabled\(name\):/)
  assert.match(setupSource, /if name != 'natten':/)
  assert.match(setupSource, /torch\.compile = identity_compile/)
  assert.match(setupSource, /import_with_torch_compile_disabled\(name\)/)
  assert.match(setupSource, /def import_natten_with_torch_compile_disabled\(\):/)
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    linux = root / 'linux-wheelhouse'
    windows = root / 'windows-wheelhouse'
    linux.mkdir()
    windows.mkdir()
    (windows / 'cumesh_vb-1.0+cu124torch2.6-cp311-cp311-win_amd64.whl').write_text('', encoding='utf-8')
    print(json.dumps({
        'linux': setup._native_import_modules_for_wheelhouse(linux),
        'windows': setup._native_import_modules_for_wheelhouse(windows),
        'windows_aliases': setup._windows_native_aliases_for_wheelhouse(windows),
        'windows_upstream': setup._upstream_import_modules_for_wheelhouse(windows),
    }, sort_keys=True))
`)

  assert.deepEqual(result.linux, ['cumesh', 'flex_gemm', 'o_voxel', 'nvdiffrast', 'nvdiffrec_render'])
  assert.deepEqual(result.windows, ['cumesh_vb', 'flex_gemm_ap', 'o_voxel_vb_ap', 'nvdiffrast', 'nvdiffrec_render'])
  assert.deepEqual(result.windows_aliases, { cumesh: 'cumesh_vb', flex_gemm: 'flex_gemm_ap', o_voxel: 'o_voxel_vb_ap' })
  assert.deepEqual(result.windows_upstream, ['cumesh', 'flex_gemm', 'o_voxel'])
})

test('runtime installs Windows native aliases before importing upstream inference', () => {
  const runtime = readFileSync(join(repoRoot, 'pixal3d_extension', 'runtime.py'), 'utf8')
  assert.match(runtime, /def _install_windows_native_module_aliases\(\)/)
  assert.match(runtime, /"cumesh": "cumesh_vb"/)
  assert.match(runtime, /"flex_gemm": "flex_gemm_ap"/)
  assert.match(runtime, /"o_voxel": "o_voxel_vb_ap"/)
  assert.match(runtime, /sys\.modules\[upstream_name\] = module/)
  assert.match(runtime, /def _install_windows_o_voxel_python_compat\(o_voxel_module: Any\) -> None:/)
  assert.match(runtime, /"postprocess": "pixal3d_extension\.o_voxel_compat\.postprocess"/)
  assert.match(runtime, /"rasterize": "pixal3d_extension\.o_voxel_compat\.rasterize"/)
  assert.match(runtime, /_prepare_runtime_compat\(\)[\s\S]*?_install_windows_native_module_aliases\(\)[\s\S]*?_install_natten_fallback\(\)[\s\S]*?inference_module = importlib\.import_module\("inference"\)/)
})

test('runtime exposes Windows o_voxel postprocess and rasterize compatibility modules', () => {
  const result = runPython(`
import json, sys, types
from pixal3d_extension import runtime

for name in [
    'o_voxel',
    'o_voxel._C',
    'o_voxel.postprocess',
    'o_voxel.rasterize',
    'o_voxel_vb_ap',
    'o_voxel_vb_ap._C',
    'pixal3d_extension.o_voxel_compat.postprocess',
    'pixal3d_extension.o_voxel_compat.rasterize',
]:
    sys.modules.pop(name, None)

fake_o_voxel = types.ModuleType('o_voxel_vb_ap')
fake_native = types.ModuleType('o_voxel_vb_ap._C')
fake_postprocess = types.ModuleType('pixal3d_extension.o_voxel_compat.postprocess')
fake_rasterize = types.ModuleType('pixal3d_extension.o_voxel_compat.rasterize')
fake_postprocess.to_glb = lambda *args, **kwargs: 'glb'
fake_rasterize.VoxelRenderer = type('VoxelRenderer', (), {})

original_import_module = runtime.importlib.import_module
def fake_import_module(name):
    if name == 'o_voxel_vb_ap':
        return fake_o_voxel
    if name == 'o_voxel_vb_ap._C':
        return fake_native
    if name == 'pixal3d_extension.o_voxel_compat.postprocess':
        return fake_postprocess
    if name == 'pixal3d_extension.o_voxel_compat.rasterize':
        return fake_rasterize
    return original_import_module(name)

original_os_name = runtime.os.name
runtime.os.name = 'nt'
runtime.importlib.import_module = fake_import_module
try:
    runtime._install_windows_native_module_aliases()
    o_voxel = sys.modules['o_voxel']
    print(json.dumps({
        'postprocess_attr': o_voxel.postprocess is fake_postprocess,
        'rasterize_attr': o_voxel.rasterize is fake_rasterize,
        'native_attr': o_voxel._C is fake_native,
        'postprocess_submodule': sys.modules['o_voxel.postprocess'] is fake_postprocess,
        'rasterize_submodule': sys.modules['o_voxel.rasterize'] is fake_rasterize,
        'native_submodule': sys.modules['o_voxel._C'] is fake_native,
    }, sort_keys=True))
finally:
    runtime.importlib.import_module = original_import_module
    runtime.os.name = original_os_name
`)

  assert.equal(result.postprocess_attr, true)
  assert.equal(result.rasterize_attr, true)
  assert.equal(result.native_attr, true)
  assert.equal(result.postprocess_submodule, true)
  assert.equal(result.rasterize_submodule, true)
  assert.equal(result.native_submodule, true)
})

test('runtime installs NATTEN fallback without exposing legacy natten.functional API', () => {
  const result = runPython(`
import importlib.util, json, sys, types
from pixal3d_extension import runtime

original_import_module = runtime.importlib.import_module

def fake_import_module(name):
    if name == 'natten':
        raise ModuleNotFoundError("No module named 'natten'", name='natten')
    return original_import_module(name)

fake_torch = types.ModuleType('torch')
fake_torch.Tensor = object
fake_torch_nn = types.ModuleType('torch.nn')
fake_torch_nn.Module = type('Module', (), {})
fake_torch_functional = types.ModuleType('torch.nn.functional')
fake_torch_functional.scaled_dot_product_attention = lambda q, k, v, dropout_p=0.0: v
fake_torch.nn = fake_torch_nn
fake_torch_nn.functional = fake_torch_functional
sys.modules['torch'] = fake_torch
sys.modules['torch.nn'] = fake_torch_nn
sys.modules['torch.nn.functional'] = fake_torch_functional
sys.modules.pop('natten', None)
sys.modules.pop('natten.functional', None)

runtime.importlib.import_module = fake_import_module
runtime._install_natten_fallback()

natten = sys.modules['natten']
functional_import_failed = False
try:
    __import__('natten.functional', fromlist=['na2d_qk'])
except ModuleNotFoundError:
    functional_import_failed = True

print(json.dumps({
    'has_libnatten': natten.HAS_LIBNATTEN,
    'has_na2d': callable(natten.na2d),
    'has_spec': natten.__spec__ is not None,
    'find_spec_name': importlib.util.find_spec('natten').name,
    'legacy_import_failed': functional_import_failed,
    'functional_loaded': 'natten.functional' in sys.modules,
}, sort_keys=True))
`)

  assert.equal(result.has_libnatten, false)
  assert.equal(result.has_na2d, true)
  assert.equal(result.has_spec, true)
  assert.equal(result.find_spec_name, 'natten')
  assert.equal(result.legacy_import_failed, true)
  assert.equal(result.functional_loaded, false)
})

test('runtime still installs NATTEN fallback on Windows when native libnatten is absent', () => {
  const result = runPython(`
import json, sys, types
from pixal3d_extension import runtime

original_import_module = runtime.importlib.import_module

def fake_import_module(name):
    if name == 'natten':
        raise ModuleNotFoundError("No module named 'natten'", name='natten')
    return original_import_module(name)

fake_torch = types.ModuleType('torch')
fake_torch.Tensor = object
fake_torch_nn = types.ModuleType('torch.nn')
fake_torch_nn.Module = type('Module', (), {})
fake_torch_functional = types.ModuleType('torch.nn.functional')
fake_torch.nn = fake_torch_nn
fake_torch_nn.functional = fake_torch_functional
sys.modules['torch'] = fake_torch
sys.modules['torch.nn'] = fake_torch_nn
sys.modules['torch.nn.functional'] = fake_torch_functional
sys.modules.pop('natten', None)
sys.modules.pop('natten.functional', None)

runtime.importlib.import_module = fake_import_module
runtime.os.name = 'nt'
runtime._install_natten_fallback()

print(json.dumps({
    'has_natten': 'natten' in sys.modules,
    'has_libnatten': sys.modules['natten'].HAS_LIBNATTEN,
    'has_na2d': callable(sys.modules['natten'].na2d),
    'has_spec': sys.modules['natten'].__spec__ is not None,
}, sort_keys=True))
`)

  assert.equal(result.has_natten, true)
  assert.equal(result.has_libnatten, false)
  assert.equal(result.has_na2d, true)
  assert.equal(result.has_spec, true)
})

test('runtime imports native NATTEN with torch.compile disabled on Windows', () => {
  const result = runPython(`
import importlib.abc, importlib.machinery, json, sys, types
from pixal3d_extension import runtime

sys.modules.pop('natten', None)
sys.modules.pop('torch', None)

class FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__('torch')

def compile_should_not_run(function=None, *args, **kwargs):
    raise RuntimeError('torch.compile should be disabled during native natten import')

fake_torch = FakeTorch()
fake_torch.compile = compile_should_not_run
original_compile = compile_should_not_run
sys.modules['torch'] = fake_torch

class NattenLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return types.ModuleType(spec.name)
    def exec_module(self, module):
        import torch
        module.__version__ = '0.21.0'
        module.HAS_LIBNATTEN = True
        compiled = torch.compile(lambda: 'native-import-ok')
        module.decorated_result = compiled()

class NattenFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'natten':
            return importlib.machinery.ModuleSpec(fullname, NattenLoader())
        return None

finder = NattenFinder()
sys.meta_path.insert(0, finder)
original_os_name = runtime.os.name
runtime.os.name = 'nt'
try:
    runtime._install_natten_fallback()
    natten = sys.modules['natten']
    print(json.dumps({
        'has_libnatten': natten.HAS_LIBNATTEN,
        'decorated_result': natten.decorated_result,
        'compile_restored': fake_torch.compile is original_compile,
    }, sort_keys=True))
finally:
    runtime.os.name = original_os_name
    sys.meta_path.remove(finder)
    sys.modules.pop('natten', None)
    sys.modules.pop('torch', None)
`)

  assert.equal(result.has_libnatten, true)
  assert.equal(result.decorated_result, 'native-import-ok')
  assert.equal(result.compile_restored, true)
})

test('runtime NATTEN fallback is local neighborhood attention, not dense global SDPA', () => {
  const runtimeSource = readFileSync(join(repoRoot, 'pixal3d_extension', 'runtime.py'), 'utf8')
  const fallbackBlock = runtimeSource.match(/def _install_natten_fallback\(\) -> None:[\s\S]*?(?=^def _silence_flex_gemm_autotuners\(\) -> None:)/m)
  assert.ok(fallbackBlock)
  assert.match(fallbackBlock[0], /torch_functional\.unfold\(/)
  assert.match(fallbackBlock[0], /scores = scores\.masked_fill\(~valid_neighbors, torch\.finfo\(scores\.dtype\)\.min\)/)
  assert.doesNotMatch(fallbackBlock[0], /scaled_dot_product_attention/)

  const result = runPython(`
import json, math, sys, types
from pixal3d_extension import runtime

try:
    import torch
    torch_available = True
except ModuleNotFoundError:
    torch_available = False
    print(json.dumps({'torch_available': False}, sort_keys=True))
    raise SystemExit(0)

original_import_module = runtime.importlib.import_module

def fake_import_module(name):
    if name == 'natten':
        raise ModuleNotFoundError("No module named 'natten'", name='natten')
    return original_import_module(name)

runtime.importlib.import_module = fake_import_module
sys.modules.pop('natten', None)
sys.modules.pop('natten.functional', None)
runtime._install_natten_fallback()
na2d = sys.modules['natten'].na2d

q = torch.arange(1, 1 + 3 * 3 * 2, dtype=torch.float32).reshape(1, 3, 3, 1, 2)
k = torch.arange(101, 101 + 3 * 3 * 2, dtype=torch.float32).reshape(1, 3, 3, 1, 2)
v = torch.arange(1001, 1001 + 3 * 3 * 3, dtype=torch.float32).reshape(1, 3, 3, 1, 3)

actual = na2d(q, k, v, kernel_size=(3, 3), dilation=(1, 1))

def manual_na2d(q, k, v, kernel_size, dilation):
    b, h, w, n, d_qk = q.shape
    d_v = v.shape[-1]
    kh, kw = kernel_size
    dh, dw = dilation
    out = torch.empty((b, h, w, n, d_v), dtype=v.dtype, device=v.device)
    radius_h = kh // 2
    radius_w = kw // 2
    scale = 1.0 / math.sqrt(d_qk)
    for bi in range(b):
        for yi in range(h):
            for xi in range(w):
                for ni in range(n):
                    scores = []
                    values = []
                    for ky in range(kh):
                        for kx in range(kw):
                            src_y = yi + (ky - radius_h) * dh
                            src_x = xi + (kx - radius_w) * dw
                            if 0 <= src_y < h and 0 <= src_x < w:
                                scores.append((q[bi, yi, xi, ni] * k[bi, src_y, src_x, ni]).sum() * scale)
                                values.append(v[bi, src_y, src_x, ni])
                    weights = torch.softmax(torch.stack(scores), dim=0)
                    out[bi, yi, xi, ni] = sum(weight * value for weight, value in zip(weights, values))
    return out

expected = manual_na2d(q, k, v, kernel_size=(3, 3), dilation=(1, 1))
global_scores = torch.softmax(torch.matmul(q.reshape(1, 9, 2), k.reshape(1, 9, 2).transpose(1, 2)) * (1.0 / math.sqrt(2)), dim=-1)
global_out = torch.matmul(global_scores, v.reshape(1, 9, 3)).reshape(1, 3, 3, 1, 3)

print(json.dumps({
    'torch_available': True,
    'shape': list(actual.shape),
    'matches_manual': bool(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)),
    'differs_from_global': bool(not torch.allclose(actual, global_out, atol=1e-6, rtol=1e-6)),
}, sort_keys=True))
`)

  if (result.torch_available) {
    assert.deepEqual(result.shape, [1, 3, 3, 1, 3])
    assert.equal(result.matches_manual, true)
    assert.equal(result.differs_from_global, true)
  } else {
    assert.equal(result.torch_available, false)
  }
})

test('requirements install supplies scipy before no-deps local wheelhouse install', () => {
  const requirements = readFileSync(join(repoRoot, 'requirements.txt'), 'utf8')
  assert.match(requirements, /^scipy$/m)
  for (const dependency of ['einops', 'filelock', 'fsspec', 'jinja2', 'matplotlib', 'networkx', 'numpy', 'opencv-python', 'typing-extensions>=4.10.0']) {
    assert.match(requirements, new RegExp(`^${dependency.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`, 'm'))
  }
})

test('setup command retries pip JSON decode failures from truncated indexes', () => {
  const result = runPython(`
import json, subprocess, tempfile
from pathlib import Path
import setup

calls = []
class Completed:
    def __init__(self, returncode, stdout='', stderr=''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

def fake_run(args, *, cwd, text, capture_output):
    calls.append(args)
    if len(calls) < 3:
        return Completed(2, stderr='Traceback\\n  File "pip/_internal/index/package_finder.py"\\njson.decoder.JSONDecodeError: Unterminated string')
    return Completed(0, stdout='ok')

subprocess.run = fake_run
with tempfile.TemporaryDirectory() as tmp:
    result = setup._run_setup_command(['/venv/bin/python', '-m', 'pip', 'install', '--no-cache-dir', '-r', 'requirements.txt'], cwd=Path(tmp))
    print(json.dumps({'ok': result['ok'], 'returncode': result['returncode'], 'attempt_count': len(result['attempts'])}, sort_keys=True))
`)

  assert.deepEqual(result, { ok: true, returncode: 0, attempt_count: 3 })
})

test('setup reports NATTEN strict NAF availability separately from base wheelhouse install', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    venv_python = root / 'venv' / 'bin' / 'python'
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text('#!/usr/bin/env python3\\n', encoding='utf-8')
    wheelhouse = root / 'wheelhouse'
    wheelhouse.mkdir()
    calls = []
    def fake_run(command, *, cwd):
        calls.append(command)
        if command[1:] == ['-c', command[-1]]:
            return {'args': command, 'returncode': 0, 'stdout_tail': '{"HAS_LIBNATTEN": false, "importable": true, "ok": true, "torch_cuda_available": true, "torch_cuda_version": "13.0", "version": "0.21.0"}\\n', 'stderr_tail': '', 'ok': True}
        return {'args': command, 'returncode': 0, 'stdout_tail': '', 'stderr_tail': '', 'ok': True}
    setup._run_setup_command = fake_run
    result = setup._install_prepare_dependencies(root, wheelhouse_path=wheelhouse)
    print(json.dumps({'natten_runtime': result['natten_runtime'], 'commands': calls}, sort_keys=True))
`)

  assert.equal(result.natten_runtime.importable, true)
  assert.equal(result.natten_runtime.HAS_LIBNATTEN, false)
  assert.equal(result.natten_runtime.strict_naf_available, false)
  assert.equal(result.natten_runtime.fallback_required, true)
})

test('setup prepare resolves the wheelhouse before dependency install and passes the verified local path', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    verified = root / '.modly' / 'cache' / 'wheelhouse' / 'pixal3d' / '0.1.0' / 'linux-aarch64-cp312-cuda124' / 'extracted'
    verified.mkdir(parents=True)
    calls = []
    setup._create_prepare_paths = lambda _layout: ([], [])
    def fake_prepare(workspace_root):
        calls.append({'step': 'prepare_wheelhouse', 'workspace_root': str(workspace_root)})
        return {'status': 'ready', 'wheelhouse_path': str(verified), 'downloads_started': False, 'installs_started': False, 'selected_asset': 'linux-aarch64-cp312-cuda124'}
    def fake_install(workspace_root, *, wheelhouse_path=None):
        calls.append({'step': 'install', 'wheelhouse_path': str(wheelhouse_path)})
        return {'status': 'installed', 'code': 'dependencies_installed', 'wheelhouse': str(wheelhouse_path), 'commands': []}
    setup._prepare_wheelhouse_for_setup = fake_prepare
    setup._install_prepare_dependencies = fake_install
    result = setup.run_setup(['--workspace-root', str(root), '--prepare', '--json'])
    print(json.dumps({'status': result['status'], 'installs_started': result['installs_started'], 'wheelhouse': result['dependency_install']['wheelhouse'], 'wheelhouse_prepare': result['wheelhouse_prepare'], 'calls': calls}, sort_keys=True))
`)

  assert.equal(result.status, 'prepared')
  assert.equal(result.installs_started, true)
  assert.match(result.wheelhouse, /linux-aarch64-cp312-cuda124\/extracted$/)
  assert.deepEqual(result.calls.map((call) => call.step), ['prepare_wheelhouse', 'install'])
  assert.equal(result.calls[1].wheelhouse_path, result.wheelhouse)
  assert.equal(result.wheelhouse_prepare.selected_asset, 'linux-aarch64-cp312-cuda124')
})

test('setup prepare fails and exits nonzero when dependency installation fails', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    verified = root / '.modly' / 'cache' / 'wheelhouse' / 'pixal3d' / '0.1.0' / 'linux-aarch64-cp312-cuda124' / 'extracted'
    verified.mkdir(parents=True)
    setup._create_prepare_paths = lambda _layout: ([], [])
    setup._prepare_wheelhouse_for_setup = lambda _workspace_root: {'status': 'ready', 'wheelhouse_path': str(verified), 'selected_asset': 'linux-aarch64-cp312-cuda124'}
    setup._install_prepare_dependencies = lambda *_args, **_kwargs: {'status': 'failed', 'code': 'dependency_install_failed', 'commands': [{'ok': False}]}
    result = setup.run_setup(['--workspace-root', str(root), '--prepare', '--json'])
    print(json.dumps({'status': result['status'], 'dependency_code': result['dependency_install']['code'], 'next_steps': result['next_steps']}, sort_keys=True))
`)

  assert.deepEqual(result, {
    status: 'failed',
    dependency_code: 'dependency_install_failed',
    next_steps: ['fix dependency installation failure', 'rerun setup'],
  })
})

test('setup prepare stops before dependency install when wheelhouse preparation fails structurally', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup
from modly_wheelhouse import WheelhouseError

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    calls = []
    setup._create_prepare_paths = lambda _layout: ([], [])
    def fake_prepare(_workspace_root):
        raise WheelhouseError('checksum_mismatch', 'sha256 verification failed')
    def forbidden_install(*_args, **_kwargs):
        calls.append('install')
        raise AssertionError('install must not start after wheelhouse failure')
    setup._prepare_wheelhouse_for_setup = fake_prepare
    setup._install_prepare_dependencies = forbidden_install
    result = setup.run_setup(['--workspace-root', str(root), '--prepare', '--json'])
    print(json.dumps({'status': result['status'], 'installs_started': result['installs_started'], 'dependency_install': result['dependency_install'], 'failure_code': result['wheelhouse_prepare']['failure_code'], 'calls': calls}, sort_keys=True))
`)

  assert.deepEqual(result, {
    calls: [],
    dependency_install: null,
    failure_code: 'checksum_mismatch',
    installs_started: false,
    status: 'failed',
  })
})

test('setup wheelhouse policy falls back to hashed vendored wheels only for retryable release failures', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path
import setup
from modly_wheelhouse import WheelhouseError

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    calls = []
    setup.load_manifest = lambda _path: {'manifest': True}
    setup.detect_runtime_lane = lambda: {'os':'linux','arch':'aarch64','python_tag':'cp312','accelerator_lane':'cuda124'}
    def fake_prepare(_manifest, _runtime, _workspace_root):
        calls.append('release')
        raise WheelhouseError('network', 'temporary release failure')
    def fake_fallback(_manifest, _workspace_root, _runtime):
        calls.append('fallback')
        return {'status': 'ready', 'fallback_path': str(root / 'wheels'), 'downloads_started': False, 'installs_started': False, 'sha256_verified': True}
    setup.prepare_wheelhouse = fake_prepare
    setup.resolve_verified_fallback = fake_fallback
    retryable = setup._prepare_wheelhouse_for_setup(root)

    calls.clear()
    def bad_checksum(_manifest, _runtime, _workspace_root):
        calls.append('release')
        raise WheelhouseError('checksum_mismatch', 'bad asset')
    setup.prepare_wheelhouse = bad_checksum
    try:
        setup._prepare_wheelhouse_for_setup(root)
    except WheelhouseError as exc:
        structural = {'code': exc.code, 'calls': calls}
    print(json.dumps({'retryable': retryable, 'structural': structural}, sort_keys=True))
`)

  assert.equal(result.retryable.status, 'ready')
  assert.match(result.retryable.fallback_path, /wheels$/)
  assert.equal(result.retryable.fallback_reason, 'network')
  assert.equal(result.retryable.fallback_mode, 'migration_only')
  assert.deepEqual(result.structural, { code: 'checksum_mismatch', calls: ['release'] })
})

test('MVP wheelhouse release archive uses stdlib-extractable zip instead of tar.zst', () => {
  const manifest = JSON.parse(readFileSync(join(repoRoot, 'wheelhouse.manifest.json'), 'utf8'))

  for (const asset of manifest.assets) {
    assert.equal(asset.compression, 'zip')
    assert.match(asset.filename, /\.zip$/)
    assert.doesNotMatch(asset.filename, /\.tar\.zst$/)
  }
})

test('wheelhouse docs separate end-user release assets from maintainer build and publish recipes', () => {
  const readme = readFileSync(join(repoRoot, 'README.md'), 'utf8')
  const wheelhouseReadmePath = join(repoRoot, 'tools', 'wheelhouse', 'README.md')

  assert.match(readme, /release-backed wheelhouse/i)
  assert.match(readme, /vendored `wheels\/` fallback/i)
  assert.match(readme, /--no-index --find-links|--no-index --no-deps --find-links/)
  assert.match(readme, /Linux `x64` \/ Python `cp312` \/ `cuda124`/)
  assert.match(readme, /Windows `x64` \/ Python `cp312` \/ `cuda124`/)
  assert.match(readme, /Windows `x64` \/ Python `cp311` \/ `cuda124`/)
  assert.match(readme, /o-voxel-vb-ap/)
  assert.doesNotMatch(readme, /- `natten==0\.21\.0`/)
  assert.match(readme, /natten\.HAS_LIBNATTEN/)
  assert.equal(existsSync(wheelhouseReadmePath), true)

  const wheelhouseReadme = readFileSync(wheelhouseReadmePath, 'utf8')
  for (const required of [
    'Local build recipe',
    'Asset naming',
    'SHA256SUMS',
    'Release checklist',
    'Do not commit generated wheelhouse archives or new binary wheels',
  ]) {
    assert.match(wheelhouseReadme, new RegExp(required, 'i'))
  }
})

test('run_job returns the raw pipeline GLB without a fixed final yaw rewrite', () => {
  const result = runPython(`
import json, tempfile
from pathlib import Path

from pixal3d_extension.runtime import run_job

def fake_pipeline(_source):
    def pipeline(*, image_path, output_dir, seed, resolution, low_vram, texture_size, manual_fov):
        del image_path, seed, resolution, low_vram, texture_size, manual_fov
        glb_path = Path(output_dir) / 'pipeline-output.glb'
        glb_path.write_bytes(b'raw-final-glb')
        return {'glb_path': str(glb_path), 'pbr': {'baseColor': 'kept'}}
    return pipeline

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    image_path = root / 'input.png'
    image_path.write_bytes(b'not-an-image-but-existing-file')
    output_dir = root / 'outputs'
    output_dir.mkdir()
    result = run_job({
        'input_image': str(image_path),
        'output_dir': str(output_dir),
        'model_source': 'fake-source',
        'readiness': {'generation_allowed': True, 'code': 'ready'},
        'params': {},
    }, pipeline_factory=fake_pipeline)
    glb_path = Path(result['output']['glb_path'])
    print(json.dumps({
        'status': result['status'],
        'glb_name': glb_path.name,
        'pbr': result['output']['pbr'],
        'bytes': glb_path.read_bytes().decode('utf8'),
    }, sort_keys=True))
`)

  assert.equal(result.status, 'completed')
  assert.equal(result.glb_name, 'pipeline-output.glb')
  assert.deepEqual(result.pbr, { baseColor: 'kept' })
  assert.equal(result.bytes, 'raw-final-glb')
})

test('runtime emits crash diagnostic checkpoints to stderr during generation', () => {
  const result = spawnSync(python, ['-c', `
import json, tempfile, sys, types
from pathlib import Path

class FakeScene:
    def apply_transform(self, matrix):
        pass
    def export(self, file_type):
        return b'rotated'

fake_trimesh = types.ModuleType('trimesh')
fake_trimesh.load = lambda path, file_type, force, process: FakeScene()
fake_trimesh.transformations = types.SimpleNamespace(rotation_matrix=lambda angle, axis: None)
sys.modules['trimesh'] = fake_trimesh

from pixal3d_extension.runtime import run_job

def fake_pipeline(_source):
    def pipeline(**kwargs):
        glb_path = Path(kwargs['output_dir']) / 'diagnostic.glb'
        glb_path.write_bytes(b'raw')
        return {'glb_path': str(glb_path)}
    return pipeline

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    image = root / 'input.png'
    image.write_bytes(b'image')
    output = root / 'out'
    output.mkdir()
    result = run_job({'input_image': str(image), 'output_dir': str(output), 'readiness': {'generation_allowed': True}, 'params': {}}, pipeline_factory=fake_pipeline)
    print(json.dumps({'status': result['status']}))
`], { cwd: repoRoot, encoding: 'utf8' })

  assert.equal(result.status, 0, `STDOUT:\n${result.stdout}\nSTDERR:\n${result.stderr}`)
  assert.deepEqual(JSON.parse(result.stdout), { status: 'completed' })
  assert.match(result.stderr, /\[Pixal3D Diagnostic\] run_job:start/)
  assert.match(result.stderr, /\[Pixal3D Diagnostic\] pipeline_factory:create:start/)
  assert.match(result.stderr, /\[Pixal3D Diagnostic\] pipeline_factory:call:done/)
  assert.doesNotMatch(result.stderr, /final_glb_yaw_rotation/)
})
