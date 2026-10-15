"""Real Modly-PBS entry + advanced production provisioning on a no-GPU runner.

This evidence helper never converts the production GPU gate into a setup pass.
It intentionally calls the weights-free provisioning layer separately after the
ordinary setup failure, so all optional environments can be tested on hosted CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pixal3d_extension.windows_scene_runtime import artifact
from tools.validation.windows_dependency_install import classify_probe


HOSTED_GPU_BLOCK = "Windows Scene Prep native GPU probe BLOCKED: CUDA device unavailable"


def native_snapshot(extension: Path) -> dict:
    site = extension / 'venv/Lib/site-packages'
    prefixes = ('cumesh', 'flex_gemm', 'o_voxel', 'nvdiffrast', 'nvdiffrec', 'drtk', 'flash_attn', 'natten')
    files = {path.relative_to(site).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in site.rglob('*') if path.is_file() and path.suffix.lower() in {'.pyd', '.dll'} and
             path.relative_to(site).parts[0].startswith(prefixes)}
    assert files, 'No primary native payload found'
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve(); evidence.mkdir(parents=True, exist_ok=True)
    summary = {'full_setup':'NOT_RUN', 'primary_native':'NOT_RUN', 'scene_prep_native':'NOT_RUN',
               'package_installation':'NOT_RUN', 'repair':'NOT_RUN', 'weights':'NOT_DOWNLOADED', 'inference':'NOT_RUN'}
    def run(command, cwd, label, *, check=True):
        completed = subprocess.run(list(map(str, command)), cwd=cwd, text=True, capture_output=True,
                                   timeout=3600, env={**os.environ, 'PYTHONUTF8':'1', 'PIP_NO_CACHE_DIR':'1'})
        (evidence / (label+'.json')).write_text(json.dumps({'args':list(map(str, command)), 'returncode':completed.returncode,
                                                          'stdout':completed.stdout, 'stderr':completed.stderr}, indent=2), encoding='utf-8')
        if check and completed.returncode:
            raise RuntimeError(f'{label} failed: '+completed.stderr[-3000:]+completed.stdout[-3000:])
        return completed
    try:
        assert os.name == 'nt', 'This helper requires real Windows'
        # Copy only the published checkout, never local environments or model assets.
        extension = Path(tempfile.mkdtemp(prefix='Pixal3D advanced é spaces ', dir=os.environ['RUNNER_TEMP']))
        tracked = subprocess.check_output(['git','ls-files','-z'], cwd=ROOT).decode().split('\0')
        for name in filter(None, tracked):
            assert Path(name).parts[0] not in {'models','venv','venv-scene-prep','venv-worldsculpt'}
            destination = extension/name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, destination)
        lock = json.loads((extension/'scene-prep-windows.lock.json').read_text(encoding='utf-8'))
        item = lock['modly_host_runtime']
        archive = artifact(extension, item)
        host = extension/'.ci-modly-python311'; host.mkdir()
        with tarfile.open(archive, 'r:gz') as package:
            for member in package.getmembers():
                path = PurePosixPath(member.name)
                assert not path.is_absolute() and '..' not in path.parts and '\\' not in member.name
                assert member.isdir() or member.isfile(), 'Unsupported PBS tar member'
                prefix = item['archive_prefix'].rstrip('/')+'/'
                if not member.name.startswith(prefix) or member.isdir():
                    continue
                destination = host.joinpath(*PurePosixPath(member.name[len(prefix):]).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with package.extractfile(member) as source, destination.open('wb') as output:
                    shutil.copyfileobj(source, output)
        for member, digest in item['critical_members'].items():
            assert hashlib.sha256((host/member).read_bytes()).hexdigest() == digest, member
        host_python = host/'python.exe'
        host_probe = run([host_python,'-I','-c',"import json,sys,venv,ensurepip; print(json.dumps({'version':list(sys.version_info[:3]),'prefix':sys.prefix}))"], extension, 'modly-host-python')
        assert json.loads(host_probe.stdout)['version'] == [3,11,9]
        # Preserve the precise host download identity and headers/libs evidence.
        assert (host/'include/Python.h').is_file() and (host/'libs/python311.lib').is_file()
        summary['modly_host_runtime'] = {**item, 'headers_present':(host/'include/Python.h').is_file(), 'import_library_present':(host/'libs/python311.lib').is_file()}
        payload = {'python_exe':str(host_python), 'ext_dir':str(extension), 'cuda_version':128, 'gpu_sm':120,
                   'accelerator':'cuda','torch_flavor':'cuda','platform':'win32','arch':'x64'}
        for label in ('install','repair'):
            completed = run([host_python,extension/'setup.py',json.dumps(payload)], extension, label, check=False)
            setup = json.loads(completed.stdout)
            assert completed.returncode == 1 and setup['status'] == 'failed', 'No-GPU production setup must fail'
            install = setup['dependency_install']
            assert install['pip_check']['ok'] and all(command['ok'] for command in install['commands']), install
            assert classify_probe(install['runtime_check'], True) == 'BLOCKED'
            assert setup['wheelhouse_prepare']['selected_asset'] == 'windows-x64-cp311-cuda128-blackwell'
            assert setup['wheelhouse_prepare']['sha256_verified'] is True
            manifest = json.loads((extension/'wheelhouse.manifest.json').read_text())
            selected = next(item for item in manifest['assets'] if item['id'] == setup['wheelhouse_prepare']['selected_asset'])
            assert selected['sha256'] == '162f43973de2dcde2987c0322dbf9f143bed87d267e8f0374618bcbe1f760a97'
            archives = list((extension/manifest['cache']['root']).rglob('archive.zip'))
            assert len(archives) == 1 and hashlib.sha256(archives[0].read_bytes()).hexdigest() == selected['sha256']
            if label == 'install':
                baseline = native_snapshot(extension)
                # Tamper MV and remove a second overlaid module before actual Repair.
                site = extension/'venv/Lib/site-packages'
                (site/'pixal3d/pipelines/pixal3d_mv_image_to_3d.py').write_bytes(b'tampered overlay')
                (site/'pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py').unlink()
            else:
                assert native_snapshot(extension) == baseline, 'Repair changed genuine primary native bytes'
                run([extension/'venv/Scripts/python.exe','-c',"from importlib.metadata import version; assert version('pixal3d-core')=='0.1.0+modly.blackwell.mv'"], extension, 'mv-overlay-version')
        summary.update(full_setup='FAIL', primary_native='BLOCKED', primary_binary_hashes=baseline)
        primary = extension/'venv/Scripts/python.exe'
        # Exact production WorldSculpt repair, without any host/GPU gate patching.
        world_code = "import json; from pixal3d_extension.worldsculpt_lane import provision; print(json.dumps(provision()))"
        run([primary,'-c',world_code], extension, 'worldsculpt-provision')
        world_site = extension/'venv-worldsculpt/Lib/site-packages'
        original = (world_site/'peft/__init__.py').read_bytes()
        (world_site/'peft/__init__.py').write_bytes(b'tampered')
        (world_site/'ftfy/__init__.py').unlink()
        run([primary,'-c',world_code], extension, 'worldsculpt-tamper-repair')
        assert (world_site/'peft/__init__.py').read_bytes() == original
        run([primary,'-c',world_code], extension, 'worldsculpt-idempotent-repair')
        native_world = run([primary,'-c',"from pixal3d_extension.worldsculpt_lane import repair; repair()"], extension, 'worldsculpt-production-native-gate', check=False)
        if native_world.returncode:
            assert 'Found no NVIDIA driver' in native_world.stderr and 'DLL load failed' not in native_world.stderr, native_world.stderr
            summary['worldsculpt_native_imports'] = 'BLOCKED'
        else:
            summary['worldsculpt_native_imports'] = 'PASS (imports only; GPU not qualified)'
        # Scene provisioning is the same function called by production repair;
        # it returns runtime_prepared=False until the unchanged GPU probe passes.
        scene_code = "import json; from pixal3d_extension.windows_scene_runtime import provision; print(json.dumps(provision(__import__('pathlib').Path.cwd())))"
        run([primary,'-c',scene_code], extension, 'scene-prep-provision')
        scene_site = extension/'venv-scene-prep/Lib/site-packages'
        scene_original = (scene_site/'sam3/__init__.py').read_bytes()
        (scene_site/'sam3/__init__.py').write_bytes(b'tampered')
        (scene_site/'depth_anything_3/api.py').unlink()
        run([primary,'-c',scene_code], extension, 'scene-prep-source-repair')
        assert (scene_site/'sam3/__init__.py').read_bytes() == scene_original
        run([primary,'-c',scene_code], extension, 'scene-prep-idempotent-repair')
        source_imports = run([primary,'-c',"import json; from pathlib import Path; from pixal3d_extension.windows_scene_runtime import validate_source_imports; print(json.dumps(validate_source_imports(Path.cwd(),allow_no_driver=True)))"], extension, 'scene-prep-source-imports')
        source_receipt = json.loads(source_imports.stdout.strip().splitlines()[-1])
        assert source_receipt['status'] in {'PASS','BLOCKED'} and source_receipt['runtime_prepared'] is False
        summary['scene_prep_source_imports'] = source_receipt['status']
        summary['scene_prep_source_import_checks'] = source_receipt['checks']
        summary['scene_prep_gpu_kernels'] = 'NOT_RUN'
        blocked = run([primary,'-c',"from pixal3d_extension.scene_prepare_lane import repair; repair()"], extension, 'scene-prep-production-gpu-gate', check=False)
        expected_block = HOSTED_GPU_BLOCK if source_receipt['status']=='PASS' else 'Scene Prep source imports BLOCKED:'
        assert blocked.returncode != 0 and expected_block in blocked.stderr, blocked.stderr
        assert not any(value in blocked.stderr for value in ('DLL load failed','ModuleNotFoundError','ImportError','TimeoutExpired')), blocked.stderr
        assert native_snapshot(extension) == baseline, 'Optional provisioning mutated primary native payload'
        for name in ('venv','venv-worldsculpt','venv-scene-prep'):
            run([extension/name/'Scripts/python.exe','-m','pip','check'], extension, name+'-pip-check')
        # The Python 3.12 extracted runtime is verified on every provisioning call.
        assert (extension/'python-scene-prep312/include/Python.h').is_file()
        assert (extension/'python-scene-prep312/libs/python312.lib').is_file()
        assert [p.relative_to(extension/'models').as_posix() for p in (extension/'models').rglob('*') if p.is_file()] == ['pixal3d/readiness.json']
        assert not any(p.is_file() for cache in ('hf-cache','torch-cache') for p in (extension/cache).rglob('*'))
        for pattern in ('test_blackwell_advanced.py', 'test_blackwell_review_regressions.py', 'test_windows_scene_runtime.py', 'test_windows_advanced_artifacts.py',
                        'test_worldsculpt_runtime.py', 'test_scene_prepare_runtime.py'):
            run([primary,'-m','unittest','discover','-s','test','-p',pattern], extension, pattern.removesuffix('.py'))
        summary.update(package_installation='PASS', repair='PASS', scene_prep_native='BLOCKED', hardware='NOT_QUALIFIED')
    except BaseException as exc:
        summary['unexpected_failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (evidence/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
