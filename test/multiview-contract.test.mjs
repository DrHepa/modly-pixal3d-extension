import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const manifest = JSON.parse(readFileSync(join(root, 'manifest.json'), 'utf8'))

function python(source) {
  const run = spawnSync('python3', ['-c', source], { cwd: root, encoding: 'utf8' })
  assert.equal(run.status, 0, run.stderr)
  return JSON.parse(run.stdout)
}

test('multi-image node uses upstream ordered image ports and DA3 calibration weights', () => {
  const node = manifest.nodes.find((item) => item.id === 'generate-mv')
  assert.equal(node?.input, 'image')
  assert.deepEqual(node?.inputs, ['image', 'image', 'image', 'image'])
  assert.deepEqual(node?.input_labels, ['Primary view', 'View 2', 'View 3', 'View 4'])
  assert.deepEqual(node?.weight_groups, ['pixal3d-base', 'pixal3d-mv', 'da3-base'])
  const group = manifest.weight_groups.find((item) => item.id === 'pixal3d-mv')
  const source = group?.model_sources.find((item) => item.repo_id === 'TencentARC/Pixal3D')
  assert.ok(source?.checks.includes('pipeline_mv.json'))
  for (const prefix of ['ss_flow_img_dit_1_3B_64_bf16_mv', 'slat_flow_img2shape_dit_1_3B_512_bf16_mv', 'slat_flow_img2shape_dit_1_3B_1024_bf16_mv', 'slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv']) {
    assert.ok(source.checks.includes(`ckpts/${prefix}.json`))
    assert.ok(source.checks.includes(`ckpts/${prefix}.safetensors`))
  }
  assert.equal(source.checks.length, 9)
  assert.equal(group.model_sources.length, 1)
})

test('MV docs distinguish UI-managed model weights from intentional first-use NAF bootstrap', () => {
  const docs = readFileSync(join(root, 'README.md'), 'utf8')
  assert.match(docs, /DA3\s+and Pixal3D MV model weights are UI-managed and local-only/i)
  assert.match(docs, /NAF is the one\s+intentional auxiliary[\s\S]*?bootstrap atomically on first generation/i)
  assert.match(docs, /manual bootstrap[\s\S]*?fallback/i)
  assert.match(docs, /null gaps are ignored[\s\S]*?connected views retain port order/i)
  assert.match(docs, /input_labels[\s\S]*?not every private or older fork renders\s+those\s+labels/i)
  assert.doesNotMatch(docs, /No model weights are downloaded by this path/)
})

test('Windows MV custody workflow is SHA-pinned and exercises the real Windows filesystem contract', () => {
  const workflow = readFileSync(join(root, '.github/workflows/windows-mv-path-custody.yml'), 'utf8')
  assert.match(workflow, /runs-on:\s*windows-2022/)
  assert.match(workflow, /actions\/checkout@11d5960a326750d5838078e36cf38b85af677262\s*#\s*v4/)
  assert.match(workflow, /actions\/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065\s*#\s*v5/)
  assert.match(workflow, /test_windows_multiview_custody\.py/)
})

test('WorldSculpt scene node uses pinned official adapters and the Pixal3D base group', () => {
  const node = manifest.nodes.find((item) => item.id === 'worldsculpt')
  assert.equal(node?.input, 'scene')
  assert.equal(node?.output, 'mesh')
  assert.deepEqual(node?.weight_groups, ['pixal3d-base', 'worldsculpt-adapters'])
  assert.deepEqual(node?.params_schema, [{
    id: 'face_budget', label: 'Faces per Instance', type: 'int',
    default: 1000000, min: 1000, max: 3000000,
    tooltip: 'Maximum faces per composed instance; geometry-only GLB.',
  }])
  const group = manifest.weight_groups.find((item) => item.id === 'worldsculpt-adapters')
  assert.equal(group?.model_sources.length, 1)
  const source = group.model_sources[0]
  assert.equal(source.provider, 'huggingface')
  assert.equal(source.repo_id, 'AlayaLab/WorldSculpt')
  assert.equal(source.revision, '8cb81056d803c61371dd84ef18a14142a738610e')
  assert.equal(source.destination, '.')
  const stages = ['ss_ft64_mv_lora_ibr_texverse', 'shape_ft1024_mv_lora_ibr_texverse_fixedmem05']
  const files = stages.flatMap((stage) => [
    `${stage}/config.json`,
    `${stage}/ckpts/denoiser_step0015000.pt`,
    `${stage}/ckpts/mv_aggregator_step0015000.pt`,
  ])
  assert.deepEqual(source.include_prefixes, files)
  assert.deepEqual(source.checks, files)
})

test('WorldSculpt adapter lookup uses only the host-provided shared group', () => {
  const result = python(`
import json
from pathlib import Path
from generator import Pixal3DGenerator
generator = Pixal3DGenerator('/tmp/private-worldsculpt', '/tmp/Workspace')
generator.MODEL_NODE_ID = 'worldsculpt'
without_host = None
try:
    generator._model_source()
except RuntimeError as exc:
    without_host = str(exc)
generator.shared_model_dirs = {'worldsculpt-adapters': '/tmp/host-worldsculpt'}
print(json.dumps({'without_host': without_host, 'source': str(generator._model_source())}))
`)
  assert.match(result.without_host, /shared weight groups are required \(worldsculpt-adapters\)/)
  assert.equal(result.source, '/tmp/host-worldsculpt')
})

test('host MODEL_NODE_ID selects typed branches for schema, readiness, download, load, and generation', () => {
  const result = python(`
import json
from generator import Pixal3DGenerator
observed = {}
for node_id in ('generate-mv', 'worldsculpt'):
    gen = Pixal3DGenerator('/tmp/private-' + node_id, '/tmp/Workspace')
    gen.MODEL_NODE_ID = node_id
    observed[node_id] = {'schema': [p['id'] for p in gen.params_schema()]}
    observed[node_id]['readiness'] = gen.readiness_status()['machine_code']
    try:
        gen.is_downloaded()
    except RuntimeError as exc:
        observed[node_id]['download_error'] = str(exc)
    try:
        gen.load()
    except RuntimeError as exc:
        observed[node_id]['load_error'] = str(exc)
    try:
        gen.generate(b'image', {'scene_manifest_path': '/tmp/Workspace/scene.json'})
    except ValueError as exc:
        observed[node_id]['generate_error'] = str(exc)
print(json.dumps(observed))
`)
  assert.deepEqual(result['generate-mv'].schema, ['resolution', 'low_vram', 'seed'])
  assert.equal(result['generate-mv'].readiness, 'mv_shared_groups_unavailable')
  assert.match(result['generate-mv'].download_error, /pixal3d-mv/)
  assert.match(result['generate-mv'].load_error, /pixal3d-mv/)
  assert.match(result['generate-mv'].generate_error, /reserved transport parameter/)
  assert.deepEqual(result.worldsculpt.schema, ['face_budget'])
  assert.equal(result.worldsculpt.readiness, 'worldsculpt_assets_missing')
  assert.match(result.worldsculpt.download_error, /worldsculpt-adapters/)
  assert.match(result.worldsculpt.load_error, /worldsculpt_assets_missing/)
  assert.match(result.worldsculpt.generate_error, /scene manifest, not image bytes/)
})

test('scene validation rejects missing cameras, traversal, and image-byte substitution', () => {
  const result = python(`
import json, tempfile
from pathlib import Path
from pixal3d_extension.multiview import resolve_views_dir
with tempfile.TemporaryDirectory() as tmp:
    workspace = Path(tmp)
    scene = workspace / 'scene.json'
    views = workspace / 'views'
    views.mkdir()
    (views / 'front.png').write_bytes(b'png')
    scene.write_text(json.dumps({'schema':'modly.scene-manifest.v1','sceneRoot':'views'}))
    failures=[]
    for transforms in [None, {'frames':[{'file_path':'../outside.png','transform_matrix':[[1,0,0,0]]*4}]}]:
        if transforms is not None: (views/'transforms.json').write_text(json.dumps(transforms))
        try: resolve_views_dir(scene, workspace)
        except ValueError as exc: failures.append(str(exc))
    (views/'transforms.json').write_text(json.dumps({'camera_angle_x':0.35,'frames':[{'file_path':'front.png','transform_matrix':[[1,0,0,0],[0,1,0,-3],[0,0,1,0],[0,0,0,1]]}]}))
    valid=str(resolve_views_dir(scene, workspace))
    outside=workspace/'outside.json'; outside.write_bytes((views/'transforms.json').read_bytes())
    (views/'transforms.json').unlink(); (views/'transforms.json').symlink_to(outside)
    try: resolve_views_dir(scene, workspace)
    except ValueError as exc: failures.append(str(exc))
    print(json.dumps({'failures':failures,'valid':valid}))
`)
  assert.equal(result.failures.length, 3)
  assert.match(result.failures[0], /transforms\.json/)
  assert.match(result.failures[1], /unsafe|workspace|path/)
  assert.match(result.failures[2], /transforms\.json/)
  assert.match(result.valid, /\/views$/)
})

test('nested sceneRoot dot resolves beside its manifest, while other roots stay workspace-relative', () => {
  const result = python(`
import json, tempfile
from pathlib import Path
from pixal3d_extension.multiview import resolve_views_dir
with tempfile.TemporaryDirectory() as tmp:
    workspace = Path(tmp) / 'Workspace'
    nested = workspace / 'nested' / 'views'
    nested.mkdir(parents=True)
    (nested / 'front.png').write_bytes(b'image')
    (nested / 'transforms.json').write_text(json.dumps({'camera_angle_x': 0.35, 'frames': [
        {'file_path': 'front.png', 'transform_matrix': [[1,0,0,0],[0,1,0,-3],[0,0,1,0],[0,0,0,1]]}
    ]}))
    manifest = nested / 'scene.json'
    manifest.write_text(json.dumps({'schema': 'modly.scene-manifest.v1', 'sceneRoot': '.'}))
    dot = str(resolve_views_dir(manifest, workspace))
    manifest.write_text(json.dumps({'schema': 'modly.scene-manifest.v1', 'sceneRoot': 'nested/views'}))
    relative = str(resolve_views_dir(manifest, workspace))
    print(json.dumps({'dot': dot, 'relative': relative, 'expected': str(nested)}))
`)
  assert.equal(result.dot, result.expected)
  assert.equal(result.relative, result.expected)
})

test('MV config redirects shared decoders and RMBG; missing weights fail before inference', () => {
  const result = python(`
import json, shutil, tempfile
from pathlib import Path
from pixal3d_extension.multiview import BASE_DECODER_FILES, MV_MODEL_FILES, MV_WEIGHT_FILES, AUXILIARY_FILES, EXPECTED_MODEL_FILES, missing_mv_assets, prepare_mv_pipeline_config, run_multiview
with tempfile.TemporaryDirectory() as tmp:
    home=Path(tmp); base=home/'base'; mv=home/'mv'; workspace=home/'Workspace'; views=workspace/'views'
    base.mkdir(); mv.mkdir(); views.mkdir(parents=True)
    naf=home/'naf_release.pth'; naf.write_bytes(b'naf')
    models={key:f'ckpts/{name}' for key,name in EXPECTED_MODEL_FILES.items()}
    cfg={'name':'Pixal3DMVImageTo3DPipeline','args':{'models':models,'rembg_model':{'args':{'model_name':'briaai/RMBG-2.0'}}}}
    (mv/'pipeline_mv.json').write_text(json.dumps(cfg))
    for rel in MV_WEIGHT_FILES[1:]: p=mv/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'weight')
    for rel in (*[f'ckpts/{name}.{ext}' for name in BASE_DECODER_FILES for ext in ('json','safetensors')],*AUXILIARY_FILES):
        p=base/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'weight')
    scene=workspace/'scene.json';scene.write_text(json.dumps({'schema':'modly.scene-manifest.v1','sceneRoot':'views'}))
    (views/'front.png').write_bytes(b'image')
    (views/'transforms.json').write_text(json.dumps({'camera_angle_x':0.35,'frames':[{'file_path':'front.png','transform_matrix':[[1,0,0,0],[0,0,-1,-3],[0,1,0,0],[0,0,0,1]]}]}))
    missing_before=missing_mv_assets(mv,base,naf)
    calls=[]
    def runner(**kwargs): calls.append(kwargs);Path(kwargs['output_path']).write_bytes(b'glb')
    original=(mv/'pipeline_mv.json').read_bytes()
    output=run_multiview(scene_manifest_path=scene,workspace_dir=workspace,mv_root=mv,base_root=base,naf_path=naf,output_dir=workspace/'out',params={'num_views':1,'resolution':1024},inference_runner=runner)
    untouched=(mv/'pipeline_mv.json').read_bytes()==original
    patched=json.loads((mv/'pipeline_mv.json').read_text())
    (mv/'ckpts'/f'{MV_MODEL_FILES[0]}.safetensors').unlink()
    blocked=''
    try: run_multiview(scene_manifest_path=scene,workspace_dir=workspace,mv_root=mv,base_root=base,naf_path=naf,output_dir=workspace/'out',params={'num_views':1},inference_runner=runner)
    except RuntimeError as exc: blocked=str(exc)
    (mv/'ckpts'/f'{MV_MODEL_FILES[0]}.safetensors').write_bytes(b'weight')
    # Simulate an older install that patched absolute paths in shared config.
    for key,value in patched['args']['models'].items():
        if value.rsplit('/',1)[-1] in BASE_DECODER_FILES: patched['args']['models'][key]='C:\\\\old\\\\ckpts\\\\'+value.rsplit('/',1)[-1]
    patched['args']['rembg_model']['args']['model_name']='C:\\\\old\\\\auxiliary\\\\rmbg'
    (mv/'pipeline_mv.json').write_text(json.dumps(patched))
    moved=home/'moved-base';shutil.copytree(base,moved)
    private=prepare_mv_pipeline_config(mv,moved,home/'private'/'pipeline_mv.local.json')
    repatched=json.loads(private.read_text())
    print(json.dumps({'missing':missing_before,'output':output.is_file(),'calls':calls,'source_untouched':untouched,'private_removed':not Path(calls[0]['config_file']).exists(),'models':list(repatched['args']['models'].values()),'rmbg':repatched['args']['rembg_model']['args']['model_name'],'blocked':blocked}))
`)
  assert.deepEqual(result.missing, [])
  assert.equal(result.output, true)
  assert.equal(result.calls.length, 1)
  assert.match(result.calls[0].config_file, /pipeline_mv\.local\.json$/)
  assert.match(result.calls[0].views_dir, /\/Workspace\/views$/)
  assert.equal(result.models.filter((path) => path.includes('/moved-base/ckpts/')).length, 3)
  assert.equal(result.models.filter((path) => path.endsWith('_mv')).length, 4)
  assert.match(result.rmbg, /\/moved-base\/auxiliary\/rmbg$/)
  assert.equal(result.source_untouched, true)
  assert.equal(result.private_removed, true)
  assert.match(result.blocked, /Pixal3D MV weights missing/)
})

test('MV generator requires upstream extra_image_paths and rejects legacy transport parameters', () => {
  const result = python(`
import json
from generator import Pixal3DGenerator
gen=Pixal3DGenerator('/tmp/models/pixal3d/generate-mv','/tmp/Workspace')
gen.MODEL_NODE_ID='generate-mv'
gen.shared_model_dirs={'pixal3d-base':'/tmp/base','pixal3d-mv':'/tmp/mv'}
errors=[]
for image,params in [(b'image',{'scene_manifest_path':'/tmp/Workspace/scene.json'}),(b'image',{})]:
    try: gen.generate(image,params)
    except ValueError as exc: errors.append(str(exc))
print(json.dumps(errors))
`)
  assert.match(result[0], /reserved transport parameter/)
  assert.match(result[1], /extra_image_paths/)
})

test('normal setup verifies and installs the bundled MV core after base wheelhouse', () => {
  const result = python(`
import json, tempfile
from pathlib import Path
from zipfile import ZipFile
import setup
wheel=setup.SCRIPT_DIR/setup.MV_CORE_WHEEL
with ZipFile(wheel) as archive:
    has_class='pixal3d/pipelines/pixal3d_mv_image_to_3d.py' in archive.namelist()
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp); py=root/'venv/bin/python';py.parent.mkdir(parents=True);py.write_text('')
    wh=root/'wheelhouse';wh.mkdir()
    calls=[]
    def fake_run(command,*,cwd):
        calls.append(command)
        return {'args':command,'returncode':0,'ok':True,'stdout_tail':'{"ok":true,"importable":true,"HAS_LIBNATTEN":false,"transformers_version":"4.57.3"}', 'stderr_tail':''}
    setup._run_setup_command=fake_run
    good=setup._install_prepare_dependencies(root,wheelhouse_path=wh)
    setup.MV_CORE_WHEEL_SHA256='0'*64
    bad=setup._install_prepare_dependencies(root,wheelhouse_path=wh)
    print(json.dumps({'has_class':has_class,'good':good['status'],'bad':bad['code'],'commands':calls}))
`)
  assert.equal(result.has_class, true)
  assert.equal(result.good, 'installed')
  assert.equal(result.bad, 'mv_core_wheel_missing_or_invalid')
  const base = result.commands.findIndex((command) => command.includes('pixal3d-core==0.1.0+modly'))
  const overlay = result.commands.findIndex((command) => command.some((arg) => arg.endsWith('/wheels/mv/pixal3d_core-0.1.0+modly-py3-none-any.whl')))
  assert.ok(base >= 0 && overlay > base)
  assert.ok(result.commands[overlay].includes('--force-reinstall'))
  assert.ok(result.commands[overlay].includes('--no-index'))
})

test('MV NAF loader verifies digest, wheel provenance, and weights-only load without patching Torch Hub', () => {
  const result = python(`
import json, sys, tempfile, types
from pathlib import Path
from unittest.mock import patch
from pixal3d_extension import multiview as mv
from pixal3d_extension import naf_checkpoint
calls=[]
torch=types.ModuleType('torch'); torch.hub=types.SimpleNamespace(load=lambda *a,**k: 'remote')
def load(path, map_location, weights_only=False):
    calls.append({'path':path,'device':str(map_location),'weights_only':weights_only})
    return {'local':True}
torch.load=load;sys.modules['torch']=torch
hubconf=types.ModuleType('hubconf')
class NAF:
    def to(self, device): self.device=device;return self
    def load_state_dict(self, value): self.weights=value
    def eval(self): return self
    def requires_grad_(self, value): return self
hubconf.NAF=NAF;sys.modules['hubconf']=hubconf
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'naf_release.pth';path.write_bytes(b'local')
    hubconf.__file__=str(Path(tmp)/'hubconf.py')
    class Dist:
        files=[Path('hubconf.py')]
        def locate_file(self, value): return Path(tmp)/value
    with patch.object(mv.metadata,'distribution',return_value=Dist()), patch.object(naf_checkpoint,'NAF_SIZE',5), patch.object(naf_checkpoint,'NAF_SHA256',__import__('hashlib').sha256(b'local').hexdigest()):
        mv._verify_naf_checkpoint(path)
        module=types.SimpleNamespace(build_image_cond_model=lambda config: types.SimpleNamespace(naf_model=None,model=types.SimpleNamespace(parameters=lambda:iter([types.SimpleNamespace(device='cpu')]))))
        with mv._local_naf_extractors(module,path):
            model=module.build_image_cond_model({});model._load_naf()
            loaded=model.naf_model.weights
            untouched=torch.hub.load('other','model')
        restored=not hasattr(module.build_image_cond_model({}),'_load_naf')
        path.write_bytes(b'bad!!')
        try: mv._verify_naf_checkpoint(path)
        except RuntimeError as exc: corrupt=str(exc)
        hubconf.__file__=str(Path(tmp)/'shadow.py')
        try: mv._verified_naf_hubconf()
        except RuntimeError as exc: shadow=str(exc)
    print(json.dumps({'loaded':loaded,'calls':calls,'untouched':untouched,'restored':restored,'corrupt':corrupt,'shadow':shadow}))
`)
  assert.deepEqual(result.loaded, { local: true })
  assert.equal(result.calls[0].weights_only, true)
  assert.equal(result.untouched, 'remote')
  assert.equal(result.restored, true)
  assert.match(result.corrupt, /SHA256 mismatch/)
  assert.match(result.shadow, /not imported from the installed NAF/)
})

test('MV extractor patch serializes overlapping runs and restores after reentrant use', () => {
  const result = python(`
import json, threading, time, types
from pathlib import Path
from pixal3d_extension.multiview import _MV_RUN_LOCK, _local_naf_extractors
original=lambda config: types.SimpleNamespace(value=config['id'])
module=types.SimpleNamespace(build_image_cond_model=original)
entered=threading.Event(); order=[]
def first():
    with _MV_RUN_LOCK, _local_naf_extractors(module,Path('/one')):
        entered.set();order.append('first-enter')
        with _MV_RUN_LOCK: order.append('reentered')
        time.sleep(.05)
        order.append('first-exit')
def second():
    entered.wait()
    with _MV_RUN_LOCK, _local_naf_extractors(module,Path('/two')):
        order.append('second-enter')
a=threading.Thread(target=first);b=threading.Thread(target=second);a.start();b.start();a.join();b.join()
print(json.dumps({'order':order,'restored':module.build_image_cond_model is original}))
`)
  assert.deepEqual(result.order, ['first-enter', 'reentered', 'first-exit', 'second-enter'])
  assert.equal(result.restored, true)
})

test('MV progress and cancellation cross generator and inference boundaries', () => {
  const result = python(`
import json, tempfile, threading
from pathlib import Path
from unittest.mock import patch
from generator import Pixal3DGenerator
from pixal3d_extension.multiview import run_multiview
events=[];cancel=threading.Event()
callback=events.append
gen=Pixal3DGenerator('/tmp/models/pixal3d/generate-mv','/tmp/Workspace');gen.MODEL_NODE_ID='generate-mv'
gen.shared_model_dirs={'pixal3d-base':'/tmp/base','pixal3d-mv':'/tmp/mv','da3-base':'/tmp/da3'}
with patch.object(gen,'_prepare_generation_assets',return_value=Path('/tmp/models/pixal3d/auxiliary/naf/naf_release.pth')), patch('pixal3d_extension.multiview_images.validate_ordered_images'), patch('pixal3d_extension.multiview_images.run_multiview_from_images',return_value=Path('/tmp/result.glb')) as mock:
    gen.generate(b'image',{'extra_image_paths':['/tmp/Workspace/view2.png']},callback,cancel)
    forwarded=mock.call_args.kwargs['cancel_event'] is cancel and mock.call_args.kwargs['progress_cb'] is callback
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);workspace=root/'Workspace';views=workspace/'views';views.mkdir(parents=True)
    (views/'front.png').write_bytes(b'image')
    (views/'transforms.json').write_text(json.dumps({'camera_angle_x':0.35,'frames':[{'file_path':'front.png','transform_matrix':[[1,0,0,0],[0,1,0,-3],[0,0,1,0],[0,0,0,1]]}]}))
    scene=workspace/'scene.json';scene.write_text(json.dumps({'schema':'modly.scene-manifest.v1','sceneRoot':'views'}))
    from pixal3d_extension.multiview import MV_WEIGHT_FILES,BASE_DECODER_FILES,AUXILIARY_FILES,EXPECTED_MODEL_FILES
    mv=root/'mv';base=root/'base';mv.mkdir();base.mkdir();naf=root/'naf_release.pth';naf.write_bytes(b'naf')
    (mv/'pipeline_mv.json').write_text(json.dumps({'name':'Pixal3DMVImageTo3DPipeline','args':{'models':{k:'ckpts/'+v for k,v in EXPECTED_MODEL_FILES.items()},'rembg_model':{'args':{'model_name':'briaai/RMBG-2.0'}}}}))
    for rel in MV_WEIGHT_FILES[1:]: p=mv/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'x')
    for rel in (*[f'ckpts/{n}.{ext}' for n in BASE_DECODER_FILES for ext in ('json','safetensors')],*AUXILIARY_FILES):
        p=base/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'x')
    kwargs=dict(scene_manifest_path=scene,workspace_dir=workspace,mv_root=mv,base_root=base,naf_path=naf,output_dir=root/'out',params={'num_views':1},cancel_event=cancel)
    progress=[]
    def runner(**values):
        progress.append('runner');cancel.set();Path(values['output_path']).write_bytes(b'glb')
    try:run_multiview(**kwargs,inference_runner=runner,progress_cb=lambda n,s:progress.append(n))
    except RuntimeError as exc: after=str(exc)
    cancel.clear();cancel.set()
    try:run_multiview(**kwargs,inference_runner=runner,progress_cb=lambda n,s:progress.append(n))
    except RuntimeError as exc: before=str(exc)
    print(json.dumps({'forwarded':forwarded,'progress':progress,'before':before,'after':after}))
`)
  assert.equal(result.forwarded, true)
  assert.deepEqual(result.progress, [2, 8, 'runner'])
  assert.match(result.before, /cancelled/)
  assert.match(result.after, /cancelled/)
})
