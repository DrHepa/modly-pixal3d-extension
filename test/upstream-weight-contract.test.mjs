import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { join, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const manifest = JSON.parse(readFileSync(join(root, 'manifest.json'), 'utf8'))
const group = manifest.weight_groups?.find((item) => item.id === 'pixal3d-base')

function python(source) {
  const result = spawnSync('python3', ['-c', source], { cwd: root, encoding: 'utf8' })
  assert.equal(result.status, 0, result.stderr)
  return JSON.parse(result.stdout)
}

test('generate owns one shared multi-source group without a legacy or private download plan', () => {
  assert.ok(group)
  assert.deepEqual(manifest.nodes.map((node) => node.id), [
    'generate', 'generate-mv', 'worldsculpt', 'scene-from-estimates', 'normalize-annotated-scene',
  ])
  assert.deepEqual(manifest.nodes[0].weight_groups, ['pixal3d-base'])
  assert.equal(manifest.nodes[0].model_sources, undefined)
  assert.equal(manifest.nodes[0].hf_repo, undefined)
  assert.deepEqual(group.model_sources.map((source) => source.id), ['pixal3d', 'dinov3', 'rmbg', 'moge'])
  for (const source of group.model_sources) {
    assert.equal(source.provider, 'huggingface')
    assert.ok(source.checks.length)
    assert.ok(source.checks.every((check) => source.include_prefixes.some((prefix) => check.startsWith(prefix))))
  }
  assert.equal(group.model_sources[0].destination, '.')
  assert.ok(!group.model_sources[0].include_prefixes.some((path) => path.includes('_mv')))
  assert.deepEqual(group.model_sources.slice(1).map((source) => source.destination), [
    'auxiliary/dinov3', 'auxiliary/rmbg', 'auxiliary/moge',
  ])
})

test('runner uses host-provided shared root and rejects stale node-private weights', () => {
  const result = python(`
import json, tempfile
from pathlib import Path
from generator import Pixal3DGenerator
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path, shared_base_root
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    private = root / 'models/pixal3d/generate'
    private.mkdir(parents=True)
    (private / 'pipeline.json').write_text('{}')
    shared = root / 'host-owned-shared-base'
    gen = Pixal3DGenerator(private, root / 'workspace')
    gen.MODEL_NODE_ID = 'generate'
    no_host = None
    try: gen.is_downloaded()
    except RuntimeError as exc: no_host = str(exc)
    gen.shared_model_dirs = {'pixal3d-base': shared}
    before = gen.is_downloaded()
    shared.mkdir(parents=True)
    (shared / 'pipeline.json').write_text('{}')
    after = gen.is_downloaded()
    layout = resolve_modly_layout(root)
    logical = 'models/pixal3d/_shared/pixal3d-base/pipeline.json'
    with shared_base_root(shared):
        resolved = resolve_storage_path(layout, logical)
    outside = resolve_storage_path(layout, logical)
    print(json.dumps({'no_host': no_host, 'before': before, 'after': after, 'resolved': str(resolved), 'outside': str(outside), 'shared': str(shared / 'pipeline.json')}))
`)
  assert.match(result.no_host, /shared weight groups are required/)
  assert.equal(result.before, false)
  assert.equal(result.after, true)
  assert.equal(result.resolved, result.shared)
  assert.notEqual(result.outside, result.shared)
})

test('shared root scoping restores the previous root after nested calls and failures', () => {
  const result = python(`
import json, tempfile
from pathlib import Path
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path, shared_base_root
with tempfile.TemporaryDirectory() as tmp:
    home = Path(tmp)
    layout = resolve_modly_layout(home)
    logical = 'models/pixal3d/_shared/pixal3d-base/pipeline.json'
    first = home / 'first'
    second = home / 'second'
    with shared_base_root(first):
        outer = resolve_storage_path(layout, logical)
        try:
            with shared_base_root(second):
                inner = resolve_storage_path(layout, logical)
                raise RuntimeError('probe')
        except RuntimeError:
            pass
        restored = resolve_storage_path(layout, logical)
    outside = resolve_storage_path(layout, logical)
    print(json.dumps({'outer': str(outer), 'inner': str(inner), 'restored': str(restored), 'outside': str(outside), 'expected_outside': str(home / logical)}))
`)
  assert.match(result.outer, /first\/pipeline\.json$/)
  assert.match(result.inner, /second\/pipeline\.json$/)
  assert.equal(result.restored, result.outer)
  assert.equal(result.outside, result.expected_outside)
})

test('single-view runtime has no persistent global hubconf NAF mutation', () => {
  const runtime = readFileSync(join(root, 'pixal3d_extension', 'runtime.py'), 'utf8')
  assert.doesNotMatch(runtime, /_patch_hubconf_naf_loader|setattr\(hubconf_module, "naf"/)
  assert.match(runtime, /_scoped_single_view_naf_extractors\(inference_module, auxiliary_source\)/)
})

test('manual bootstrap only downloads NAF, not Modly-managed HF assets', () => {
  const result = python(`
import json, tempfile
from pathlib import Path
# Pin the synthetic downloader fixture without bypassing integrity validation.
import hashlib
from pixal3d_extension import naf_checkpoint
naf_checkpoint.NAF_SIZE = 10
naf_checkpoint.NAF_SHA256 = hashlib.sha256(b'checkpoint').hexdigest()
from pixal3d_extension.assets import bootstrap_auxiliary_assets
calls=[]
def fake(**kwargs):
    calls.append((kwargs['source_kind'], kwargs['filename']))
    Path(kwargs['destination']).write_bytes(b'checkpoint')
with tempfile.TemporaryDirectory() as tmp:
    result=bootstrap_auxiliary_assets(tmp, downloader=fake)
    print(json.dumps({'status': result['status'], 'calls': calls, 'allowlist': sorted(result['allowlist'])}))
`)
  assert.equal(result.status, 'ready')
  assert.deepEqual(result.calls, [['url', 'naf_release.pth']])
  assert.deepEqual(result.allowlist, ['naf'])
})

test('shared base assets and first-generation NAF bootstrap reach the existing generate node', () => {
  const result = python(`
import json, tempfile
from pathlib import Path
# Pin the synthetic downloader fixture without bypassing integrity validation.
import hashlib
from pixal3d_extension import naf_checkpoint
naf_checkpoint.NAF_SIZE = 3
naf_checkpoint.NAF_SHA256 = hashlib.sha256(b'naf').hexdigest()
from generator import Pixal3DGenerator
from pixal3d_extension.assets import PRIMARY_ASSET, AUXILIARY_ASSETS
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
with tempfile.TemporaryDirectory() as tmp:
    home = Path(tmp)
    private = home / 'models/pixal3d/generate'
    private.mkdir(parents=True)
    shared = home / 'models/pixal3d/_shared/pixal3d-base'
    layout = resolve_modly_layout(home)
    for asset in [PRIMARY_ASSET, *(value for key, value in AUXILIARY_ASSETS.items() if key != 'naf')]:
        for sentinel in asset.sentinel_paths:
            path = resolve_storage_path(layout, sentinel)
            path.parent.mkdir(parents=True, exist_ok=True)
            if sentinel.endswith('pipeline.json'):
                path.write_text(json.dumps({'args': {'image_cond_model': {'args': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'}}, 'rembg_model': {'args': {'model_name': 'briaai/RMBG-2.0'}}}}))
            else:
                path.write_bytes(b'model')
    output = home / 'workspace/Workflows'
    output.mkdir(parents=True)
    calls = []
    def downloader(**kwargs):
        calls.append(kwargs['filename'])
        Path(kwargs['destination']).write_bytes(b'naf')
    def factory(source):
        def run(**kwargs):
            target = Path(kwargs['output_dir']) / 'mesh.glb'
            target.write_bytes(b'glb')
            return {'glb_path': str(target)}
        return run
    gen = Pixal3DGenerator(private, output, pipeline_factory=factory)
    gen.MODEL_NODE_ID = 'generate'
    gen.shared_model_dirs = {'pixal3d-base': shared}
    gen.load()
    image = output / 'input.png'
    image.write_bytes(b'png')
    result = gen.generate({'input_image': str(image), 'output_dir': str(output), 'workspace_root': str(home), 'readiness': {'generation_allowed': True}, 'auxiliary_bootstrap_downloader': downloader})
    pipeline = json.loads((shared / 'pipeline.json').read_text())
    print(json.dumps({'output': result.name, 'calls': calls, 'dino': pipeline['args']['image_cond_model']['args']['model_name'], 'naf_exists': (home / 'models/pixal3d/auxiliary/naf/naf_release.pth').is_file()}))
`)
  assert.equal(result.output, 'mesh.glb')
  assert.deepEqual(result.calls, ['naf_release.pth'])
  assert.match(result.dino, /_shared\/pixal3d-base\/auxiliary\/dinov3$/)
  assert.equal(result.naf_exists, true)
})

test('generator operations scope a noncanonical host root without leaking into a later standalone call', () => {
  const result = python(`
import json, tempfile
from pathlib import Path
from generator import Pixal3DGenerator
from pixal3d_extension import pipeline_patch, runtime
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
with tempfile.TemporaryDirectory() as tmp:
    home = Path(tmp)
    private = home / 'models/pixal3d/generate'
    shared = home / 'host-selected-root'
    private.mkdir(parents=True)
    shared.mkdir(parents=True)
    (shared / 'pipeline.json').write_text('{}')
    output = home / 'workspace/Workflows'
    output.mkdir(parents=True)
    logical = 'models/pixal3d/_shared/pixal3d-base/pipeline.json'
    layout = resolve_modly_layout(home)
    seen = []
    def fake_patch(*args, **kwargs):
        seen.append(str(resolve_storage_path(layout, logical)))
        return {'status': 'patched'}
    def fake_job(*args, **kwargs):
        seen.append(str(resolve_storage_path(layout, logical)))
        glb = output / 'mesh.glb'
        glb.write_bytes(b'glb')
        return {'status': 'completed', 'output': {'glb_path': str(glb)}}
    pipeline_patch.patch_pipeline = fake_patch
    runtime.run_job = fake_job
    gen = Pixal3DGenerator(private, output)
    gen.MODEL_NODE_ID = 'generate'
    gen.shared_model_dirs = {'pixal3d-base': shared}
    gen.load()
    after_load = str(resolve_storage_path(layout, logical))
    gen.generate({'input_image': str(output / 'input.png'), 'output_dir': str(output)})
    after_generate = str(resolve_storage_path(layout, logical))
    standalone = str(resolve_storage_path(layout, logical))
    print(json.dumps({'seen': seen, 'after_load': after_load, 'after_generate': after_generate, 'standalone': standalone, 'shared': str(shared / 'pipeline.json'), 'canonical': str(home / logical)}))
`)
  assert.deepEqual(result.seen, [result.shared, result.shared])
  assert.equal(result.after_load, result.canonical)
  assert.equal(result.after_generate, result.canonical)
  assert.equal(result.standalone, result.canonical)
})
