#!/usr/bin/env python3
"""Reproducible, dependency-free packaging of exact SAM3/DA3 source snapshots.

This is a reviewed metadata-only inference policy, not an upstream general-purpose
wheel. It preserves every byte under each runtime package, including shared train
helpers. No upstream module/function is rewritten or fabricated. No direct_url.json
is invented. The extension must read provenance.json rather than claim a VCS install.

Acquire with GIT_LFS_SKIP_SMUDGE=1, git init/fetch --depth1 --filter=blob:none,
then sparse-checkout SAM3 'sam3' and DA3 'src'. Pass the exact detached commits.
Network is only needed to acquire missing Git objects before running this builder.
The builder itself has no network/import of upstream code, no package installation,
no checkpoint access, and uses only Python's standard library. Requires Python3.11+.
"""
from __future__ import annotations
import argparse, base64, csv, gzip, hashlib, io, json, re, subprocess, tarfile, zipfile
from pathlib import Path

SOURCES={
 'sam3':{'repository':'https://github.com/facebookresearch/sam3','commit':'2345a4ad109ac29c569da749c91d84f10dc08c40','package':'sam3','name':'sam3','version':None,'license':'LicenseRef-SAM','author':'Meta AI Research','python':'>=3.12',
 'requires':['setuptools==80.9.0','torch>=2.7','torchvision>=0.22','timm>=1.0.17','numpy>=1.26,<2','tqdm','ftfy==6.1.1','regex','iopath>=0.1.10','typing-extensions','huggingface-hub','einops','pillow','opencv-python-headless','scipy','pycocotools','psutil','triton-windows==3.3.1.post21; sys_platform == "win32"','triton>=3.3; sys_platform != "win32"'],
 'contract':{'entrypoint':'sam3.model_builder.build_sam3_video_predictor','input':'ordered image directory','excluded_features':['direct video decoding','train entrypoints','notebooks','demo apps'],'upstream_readme_runtime_floor':'Python>=3.12, Torch>=2.7, CUDA>=12.6','distribution_only_license_fix':'Do not reproduce the upstream incorrect MIT classifier; retain custom SAM License.'}},
 'da3':{'repository':'https://github.com/ByteDance-Seed/Depth-Anything-3','commit':'3d835ec1a5802d64a8b8b15f817a1ab54809bfe4','package':'src/depth_anything_3','name':'depth-anything-3','version':'0.0.0','license':'Apache-2.0','author':'ByteDance Ltd. and/or its affiliates','python':'>=3.9,<=3.13',
 'requires':['torch>=2','torchvision','einops','huggingface-hub','imageio','numpy<2','opencv-python-headless','pillow','omegaconf','addict','safetensors','tqdm'],
 'contract':{'entrypoint':'pixal3d_extension.da3_official_adapter.load_depth_anything3','model':'da3-base','export_dir':None,'external_extrinsics':None,'infer_gs':False,'excluded_features':['CLI','web/app','Gaussian splatting','export','external camera alignment'],'metadata_replacements':{'opencv-python':'opencv-python-headless'},'metadata_omissions':['pre-commit','trimesh','xformers','open3d','fastapi','uvicorn','requests','typer','evo','e3nn','moviepy','plyfile','pillow-heif','pycolmap'],'code_policy':'all upstream runtime bytes retained; unused import boundaries are owned and tested by extension adapter, not stubbed in this wheel'}}
}

def digest(b):return hashlib.sha256(b).hexdigest()
def jb(x):return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+'\n').encode()
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args])
def blob_hash(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()

def build(kind,source,out):
 policy=SOURCES[kind];commit=policy['commit'];source=source.resolve();out.mkdir(parents=True,exist_ok=True)
 assert git(source,'rev-parse','HEAD').decode().strip()==commit,'Wrong source commit'
 tree=git(source,'rev-parse',commit+'^{tree}').decode().strip()
 listed=git(source,'ls-tree','-r','-z',commit).split(b'\0');expected={}
 for row in listed:
  if not row:continue
  meta,path=row.split(b'\t',1);mode,typ,sha=meta.split();p=path.decode()
  if typ!=b'blob':continue
  if p.startswith(policy['package']+'/') or p in ('LICENSE','LICENSE.md','NOTICE','NOTICE.md','README.md','pyproject.toml','MANIFEST.in'):
   assert mode==b'100644' or mode==b'100755',f'Unexpected symlink {p}'
   expected[p]=sha.decode()
 assert 'LICENSE' in expected,'Source license missing'
 files={}
 for p,sha in sorted(expected.items()):
  b=(source/p).read_bytes();assert blob_hash(b)==sha,f'Git source mismatch: {p}'
  assert not b.startswith(b'version https://git-lfs.github.com/spec/v1'),f'LFS pointer forbidden {p}'
  assert Path(p).suffix.lower() not in {'.pt','.pth','.ckpt','.safetensors','.onnx','.gguf','.bin','.pyd','.so','.dll'},f'Weights/native payload forbidden {p}'
  files[p]=b
 if policy['version'] is None:
  version=re.search(r'__version__\s*=\s*[\'"]([^\'"]+)',files['sam3/__init__.py'].decode()).group(1)
 else:version=policy['version']
 version+='+'+'modly.inference.1'
 dist=policy['name'].replace('-','_');di=f'{dist}-{version}.dist-info';whl=out/f'{dist}-{version}-py3-none-any.whl'
 snapname=f'{dist}-{commit}.source.tar.gz'
 # Deterministic exact source snapshot contains packaging inputs and full package.
 buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w',format=tarfile.USTAR_FORMAT) as t:
  for p,b in sorted(files.items()):
   i=tarfile.TarInfo(p);i.size=len(b);i.mode=0o644;i.mtime=0;i.uid=i.gid=0;i.uname=i.gname='';t.addfile(i,io.BytesIO(b))
 gb=io.BytesIO()
 with gzip.GzipFile(fileobj=gb,mode='wb',filename='',mtime=0,compresslevel=9) as gz:gz.write(buf.getvalue())
 sb=gb.getvalue();(out/snapname).write_bytes(sb)
 payload={};mf=[]
 prefix='src/' if kind=='da3' else ''
 for p,b in sorted(files.items()):
  if not p.startswith(policy['package']+'/'):continue
  wp=p[len(prefix):];payload[wp]=b;mf.append({'source_path':p,'wheel_path':wp,'git_blob_sha1':expected[p],'sha256':digest(b),'size':len(b)})
 for p,b in sorted(files.items()):
  if p in ('LICENSE','LICENSE.md','NOTICE','NOTICE.md'):payload[f'{di}/licenses/{p}']=b
 manifest=jb(mf);(out/f'{dist}-{commit}.source-manifest.json').write_bytes(manifest)
 prov={'schema':'modly.inference-source-wheel.v1','source':{'repository':policy['repository'],'commit_id':commit,'git_tree_id':tree,'source_snapshot_sha256':digest(sb),'source_snapshot_file':snapname,'source_manifest_sha256':digest(manifest)},'build':{'generator':'modly-inference-wheel/1','metadata_policy':'scene-inference-v1','runtime_source_modified':False,'excluded_paths':[],'inference_contract':policy['contract']},'source_files':mf,'requires_dist':policy['requires']}
 payload[f'{di}/provenance.json']=jb(prov)
 meta=['Metadata-Version: 2.4',f"Name: {policy['name']}",f'Version: {version}',f"Summary: {policy['name']} pinned Modly inference-only metadata distribution",f"Author: {policy['author']}",f"License-Expression: {policy['license']}",f"Requires-Python: {policy['python']}",f"Project-URL: Source, {policy['repository']}"]
 meta += [f'License-File: licenses/{p}' for p in files if p in ('LICENSE','LICENSE.md','NOTICE','NOTICE.md')]
 meta += ['Requires-Dist: '+x for x in policy['requires']]
 meta+=['','This downstream wheel changes distribution metadata only. All packaged runtime files','are byte-identical to the exact upstream source commit. The narrowed dependency','contract is for the Modly ScenePrep inference adapter, not general upstream usage.','']
 payload[f'{di}/METADATA']='\n'.join(meta).encode()
 payload[f'{di}/WHEEL']=b'Wheel-Version: 1.0\nGenerator: modly-inference-wheel/1\nRoot-Is-Purelib: true\nTag: py3-none-any\n'
 rows=[]
 for p,b in sorted(payload.items()):rows.append([p,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(b).digest()).decode().rstrip('='),str(len(b))])
 rows.append([f'{di}/RECORD','','']);rec=io.StringIO(newline='');csv.writer(rec,lineterminator='\n').writerows(rows);payload[f'{di}/RECORD']=rec.getvalue().encode()
 def wheel_bytes():
  buf=io.BytesIO()
  with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
   for p,b in sorted(payload.items()):
    i=zipfile.ZipInfo(p,(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.create_system=3;i.external_attr=0o100644<<16;z.writestr(i,b,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
  return buf.getvalue()
 wb=wheel_bytes();assert wb==wheel_bytes(),'Non-deterministic wheel';whl.write_bytes(wb)
 with zipfile.ZipFile(whl) as z:
  assert set(z.namelist())==set(payload);assert z.testzip() is None
  for p,h,size in csv.reader(io.StringIO(z.read(f'{di}/RECORD').decode())):
   if p.endswith('/RECORD'):assert not h and not size;continue
   b=z.read(p);assert len(b)==int(size);assert 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(b).digest()).decode().rstrip('=')==h
  for m in mf:assert z.read(m['wheel_path'])==files[m['source_path']]
 result={'name':policy['name'],'version':version,'file':whl.name,'path':str(whl),'sha256':digest(wb),'size':len(wb),'source':prov['source'],'provenance_member':f'{di}/provenance.json','runtime_files':len(mf),'record_verified':True,'all_runtime_bytes_match_git':True,'deterministic_rebuild':True,'license':policy['license'],'requires_dist':policy['requires'],'import_status':'NOT RUN: Windows runtime not installed/executed','source_snapshot_path':str(out/snapname)}
 (out/f'{dist}.build-report.json').write_bytes(jb(result));return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--sam3-dir',type=Path,required=True);p.add_argument('--da3-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 results=[build('sam3',a.sam3_dir,a.output),build('da3',a.da3_dir,a.output)];print(json.dumps(results,indent=2))
if __name__=='__main__':main()
