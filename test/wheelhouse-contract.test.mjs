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
  assert.equal(windowsCp311Asset.size_bytes, 126801509)
  assert.equal(windowsCp311Asset.sha256, 'a6a3576fdb9a3a59ba002de4b2eb8a06c26e12a167eaa4a4dd15daaf156ad04c')
  assert.ok(windowsCp311Asset.packages.includes('flex-gemm-ap'))
  assert.ok(!windowsCp311Asset.packages.includes('natten'))
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
    has_natten: false,
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

test('runtime suppresses upstream FlexGEMM autotuner verbose without disabling cache', () => {
  const runtime = readFileSync(join(repoRoot, 'pixal3d_extension', 'runtime.py'), 'utf8')
  assert.match(runtime, /def _silence_flex_gemm_autotuners\(\)/)
  assert.match(runtime, /FLEX_GEMM_AUTOTUNER_VERBOSE"\] = "0"/)
  assert.match(runtime, /value\.verbose = False/)
  assert.match(runtime, /flex_gemm\.utils\.load_autotune_cache\(\)/)
  assert.match(runtime, /from inference import run_inference\n\s+_silence_flex_gemm_autotuners\(\)/)
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
  assert.match(runtime, /sys\.modules\[upstream_name\] = importlib\.import_module\(windows_name\)/)
  assert.match(runtime, /_prepare_runtime_compat\(\)\n\s+_install_windows_native_module_aliases\(\)\n\s+_install_natten_fallback\(\)\n\s+from inference import run_inference/)
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
