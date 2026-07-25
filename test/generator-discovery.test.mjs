import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import test from 'node:test'

const extensionRoot = dirname(dirname(fileURLToPath(import.meta.url)))
const python = process.env.PYTHON ?? 'python3'

function prepareHost(root, baseSource) {
  const hostRoot = join(root, 'host')
  const generatorsRoot = join(hostRoot, 'services', 'generators')
  mkdirSync(generatorsRoot, { recursive: true })
  writeFileSync(join(hostRoot, 'services', '__init__.py'), '', 'utf8')
  writeFileSync(join(generatorsRoot, '__init__.py'), '', 'utf8')
  writeFileSync(join(generatorsRoot, 'base.py'), baseSource, 'utf8')
  return hostRoot
}

function runPython(source, args, cwd) {
  return spawnSync(python, ['-c', source, ...args], {
    cwd,
    encoding: 'utf8',
    env: {
      ...process.env,
      PYTHONNOUSERSITE: '1',
      PYTHONPATH: '',
    },
  })
}

const hostBase = `
class GenerationCancelled(Exception):
    pass


class BaseGenerator:
    def __init__(self, model_dir, outputs_dir):
        self.model_dir = model_dir
        self.outputs_dir = outputs_dir
        self._model = None
        self.base_init_called = True
        self.base_unload_called = False

    def is_loaded(self):
        return self._model is not None

    def unload(self):
        self._model = None
        self.base_unload_called = True

    def _auto_download(self):
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id="unexpected/runtime-download")

    def _check_cancelled(self, cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled()
`

test('direct discovery imports local code, subclasses the host base, and preserves lifecycle and call shapes', () => {
  const root = mkdtempSync(join(tmpdir(), 'pixal3d-generator-discovery-'))
  try {
    const hostRoot = prepareHost(root, hostBase)
    const outsideRoot = join(root, 'outside')
    mkdirSync(outsideRoot)

    const result = runPython(
      `
import importlib.util
import json
import sys
import tempfile
import threading
import types
from pathlib import Path

extension_root = Path(sys.argv[1]).resolve()
host_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(host_root))
extension_was_absent = str(extension_root) not in sys.path

spec = importlib.util.spec_from_file_location(
    "pixal3d_direct_discovery_generator",
    extension_root / "generator.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from services.generators.base import BaseGenerator, GenerationCancelled
from pixal3d_extension import runtime

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    model_dir = root / "model"
    outputs_dir = root / "outputs"
    model_dir.mkdir()
    outputs_dir.mkdir()

    generator = module.Pixal3DGenerator(model_dir, outputs_dir)
    initially_loaded = generator.is_loaded()
    generator.load()
    loaded_after_load = generator.is_loaded()
    generator.unload()
    unloaded_after_unload = not generator.is_loaded()

    network_calls = []
    fake_hub = types.ModuleType("huggingface_hub")

    def guarded_snapshot_download(**kwargs):
        network_calls.append(kwargs)
        raise AssertionError("snapshot_download must not run from Pixal3D direct mode")

    fake_hub.snapshot_download = guarded_snapshot_download
    sys.modules["huggingface_hub"] = fake_hub
    auto_download_blocked = False
    try:
        generator._auto_download()
    except RuntimeError as exc:
        auto_download_blocked = "Modly Models" in str(exc) and "Repair" in str(exc)

    calls = []

    def fake_run_job(job, *, pipeline_factory=None):
        calls.append({
            "job": dict(job),
            "pipeline_factory": pipeline_factory,
            "input_exists": Path(job["input_image"]).is_file(),
        })
        output_path = outputs_dir / f"result-{len(calls)}.glb"
        output_path.write_bytes(b"glb")
        return {"status": "completed", "output": {"glb_path": str(output_path)}}

    original_run_job = runtime.run_job
    runtime.run_job = fake_run_job
    try:
        four_arg_event = threading.Event()
        byte_result = generator.generate(
            b"png-bytes",
            {"seed": 7},
            lambda _pct, _step: None,
            four_arg_event,
        )
        byte_call = calls[-1]
        byte_input_removed = not Path(byte_call["job"]["input_image"]).exists()

        source_image = root / "source.png"
        source_image.write_bytes(b"png")
        source_job = {
            "input_image": str(source_image),
            "output_dir": str(outputs_dir),
            "model_source": str(model_dir),
            "params": {"seed": 9},
        }
        job_result = generator.generate(source_job)
        job_call = calls[-1]

        before_event = threading.Event()
        before_event.set()
        calls_before_cancel = len(calls)
        cancelled_before = False
        try:
            generator.generate(source_job, cancel_evt=before_event)
        except GenerationCancelled:
            cancelled_before = True
        no_run_after_pre_cancel = len(calls) == calls_before_cancel
    finally:
        runtime.run_job = original_run_job

    print(json.dumps({
        "auto_download_blocked": auto_download_blocked,
        "base_initialized": generator.base_init_called,
        "base_unload_called": generator.base_unload_called,
        "byte_auxiliary_mode": byte_call["job"].get("auxiliary_mode"),
        "byte_input_existed": byte_call["input_exists"],
        "byte_input_removed": byte_input_removed,
        "byte_network_available": byte_call["job"].get("network_available"),
        "byte_result_exists": byte_result.is_file(),
        "cancelled_before": cancelled_before,
        "direct_import_injected_root": str(extension_root) in sys.path,
        "extension_was_absent": extension_was_absent,
        "four_arg_seed": byte_call["job"]["params"]["seed"],
        "host_subclass": issubclass(module.Pixal3DGenerator, BaseGenerator),
        "initially_loaded": initially_loaded,
        "job_auxiliary_mode": job_call["job"].get("auxiliary_mode"),
        "job_dict_seed": job_call["job"]["params"]["seed"],
        "job_model_source_matches": Path(job_call["job"]["model_source"]) == model_dir,
        "job_network_available": job_call["job"].get("network_available"),
        "job_result_exists": job_result.is_file(),
        "loaded_after_load": loaded_after_load,
        "model_dir_preserved": generator.model_dir == model_dir,
        "network_untouched": not network_calls,
        "no_run_after_pre_cancel": no_run_after_pre_cancel,
        "outputs_dir_preserved": generator.outputs_dir == outputs_dir,
        "unloaded_after_unload": unloaded_after_unload,
        "workspace_dir_preserved": generator.workspace_dir == outputs_dir,
    }, sort_keys=True))
`,
      [extensionRoot, hostRoot],
      outsideRoot,
    )

    assert.equal(result.status, 0, `STDOUT:\n${result.stdout}\nSTDERR:\n${result.stderr}`)
    assert.deepEqual(JSON.parse(result.stdout), {
      auto_download_blocked: true,
      base_initialized: true,
      base_unload_called: true,
      byte_auxiliary_mode: 'local',
      byte_input_existed: true,
      byte_input_removed: true,
      byte_network_available: false,
      byte_result_exists: true,
      cancelled_before: true,
      direct_import_injected_root: true,
      extension_was_absent: true,
      four_arg_seed: 7,
      host_subclass: true,
      initially_loaded: false,
      job_auxiliary_mode: 'local',
      job_dict_seed: 9,
      job_model_source_matches: true,
      job_network_available: false,
      job_result_exists: true,
      loaded_after_load: true,
      model_dir_preserved: true,
      network_untouched: true,
      no_run_after_pre_cancel: true,
      outputs_dir_preserved: true,
      unloaded_after_unload: true,
      workspace_dir_preserved: true,
    })
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('real runtime preflight blocks missing UI-managed assets without bootstrap or network access', () => {
  const root = mkdtempSync(join(tmpdir(), 'pixal3d-generator-local-assets-'))
  try {
    const outsideRoot = join(root, 'outside')
    mkdirSync(outsideRoot)

    const result = runPython(
      `
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

extension_root = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location(
    "pixal3d_local_assets_generator",
    extension_root / "generator.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from pixal3d_extension import assets, runtime

with tempfile.TemporaryDirectory() as tmp:
    modly_root = Path(tmp) / "Modly"
    model_dir = modly_root / "models" / "pixal3d" / "generate"
    outputs_dir = modly_root / "workspace" / "Workflows"
    model_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)
    (model_dir / "pipeline.json").write_text("{}", encoding="utf-8")
    image_path = outputs_dir / "input.png"
    image_path.write_bytes(b"png")

    bootstrap_calls = []
    downloader_calls = []
    network_calls = []
    pipeline_calls = []

    def forbidden_bootstrap(*args, **kwargs):
        bootstrap_calls.append({"args": args, "kwargs": kwargs})
        raise AssertionError("generator runtime must not bootstrap auxiliary assets")

    def forbidden_downloader(*args, **kwargs):
        downloader_calls.append({"args": args, "kwargs": kwargs})
        raise AssertionError("generator runtime must not invoke an auxiliary downloader")

    def forbidden_urlopen(*args, **kwargs):
        network_calls.append({"args": args, "kwargs": kwargs})
        raise AssertionError("generator runtime must not access the network")

    def forbidden_pipeline(_source):
        pipeline_calls.append(_source)
        raise AssertionError("missing local assets must fail before pipeline creation")

    original_bootstrap = runtime.bootstrap_auxiliary_assets
    original_downloader = assets._default_auxiliary_downloader
    original_urlopen = assets.urllib.request.urlopen
    runtime.bootstrap_auxiliary_assets = forbidden_bootstrap
    assets._default_auxiliary_downloader = forbidden_downloader
    assets.urllib.request.urlopen = forbidden_urlopen
    try:
        generator = module.Pixal3DGenerator(
            model_dir,
            outputs_dir,
            pipeline_factory=forbidden_pipeline,
        )
        generator.load()
        error = ""
        try:
            generator.generate({
                "input_image": str(image_path),
                "output_dir": str(outputs_dir),
                "model_source": "TencentARC/Pixal3D",
                "readiness": {"generation_allowed": True, "code": "ready"},
                "auxiliary_mode": "default",
                "network_available": True,
                "offline": True,
                "auxiliary_bootstrap_downloader": forbidden_downloader,
                "params": {
                    "auxiliary_mode": "remote",
                    "network_available": True,
                    "offline": True,
                    "seed": 5,
                },
            })
        except RuntimeError as exc:
            error = str(exc)
    finally:
        runtime.bootstrap_auxiliary_assets = original_bootstrap
        assets._default_auxiliary_downloader = original_downloader
        assets.urllib.request.urlopen = original_urlopen

    print(json.dumps({
        "actionable": "Modly Models" in error and "Repair" in error,
        "bootstrap_calls": len(bootstrap_calls),
        "downloader_calls": len(downloader_calls),
        "local_mode": '"auxiliary_mode": "local"' in error,
        "missing_assets": "missing_auxiliary_assets" in error,
        "network_calls": len(network_calls),
        "pipeline_calls": len(pipeline_calls),
        "strict_offline_not_used": "offline_runtime_dependencies_unresolved" not in error,
    }, sort_keys=True))
`,
      [extensionRoot],
      outsideRoot,
    )

    assert.equal(result.status, 0, `STDOUT:\n${result.stdout}\nSTDERR:\n${result.stderr}`)
    assert.deepEqual(JSON.parse(result.stdout), {
      actionable: true,
      bootstrap_calls: 0,
      downloader_calls: 0,
      local_mode: true,
      missing_assets: true,
      network_calls: 0,
      pipeline_calls: 0,
      strict_offline_not_used: true,
    })
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('standalone discovery falls back only when the host module is absent', () => {
  const root = mkdtempSync(join(tmpdir(), 'pixal3d-generator-fallback-'))
  try {
    const outsideRoot = join(root, 'outside')
    mkdirSync(outsideRoot)

    const result = runPython(
      `
import importlib.util
import json
import sys
from pathlib import Path

extension_root = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location(
    "pixal3d_standalone_generator",
    extension_root / "generator.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

generator = module.Pixal3DGenerator()
generator._model = object()
loaded = generator.is_loaded()
generator.unload()
print(json.dumps({
    "fallback_module": module.BaseGenerator.__module__,
    "loaded": loaded,
    "unloaded": not generator.is_loaded(),
}, sort_keys=True))
`,
      [extensionRoot],
      outsideRoot,
    )

    assert.equal(result.status, 0, `STDOUT:\n${result.stdout}\nSTDERR:\n${result.stderr}`)
    assert.deepEqual(JSON.parse(result.stdout), {
      fallback_module: 'pixal3d_standalone_generator',
      loaded: true,
      unloaded: true,
    })
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('host import failures are not hidden by the standalone fallback', () => {
  const root = mkdtempSync(join(tmpdir(), 'pixal3d-generator-host-error-'))
  try {
    const hostRoot = prepareHost(root, 'import pixal3d_missing_host_dependency\n')
    const outsideRoot = join(root, 'outside')
    mkdirSync(outsideRoot)

    const result = runPython(
      `
import importlib.util
import sys
from pathlib import Path

extension_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(Path(sys.argv[2]).resolve()))
spec = importlib.util.spec_from_file_location(
    "pixal3d_broken_host_generator",
    extension_root / "generator.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
`,
      [extensionRoot, hostRoot],
      outsideRoot,
    )

    assert.notEqual(result.status, 0)
    assert.match(result.stderr, /pixal3d_missing_host_dependency/)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
