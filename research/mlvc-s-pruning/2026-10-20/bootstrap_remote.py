"""Inspect the actual rented environment and public dataset mounts, read-only.
Does not train or infer successful VMAF/GPU support from package installation.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {'exit_code':p.returncode,'stdout':p.stdout[:16000],'stderr':p.stderr[:2000]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'error':str(e)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result={'platform':platform.platform(),'python':sys.version,'datasets':{}}
    result['nvidia_smi']=command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'])
    for name in ['Vimeo-90k','GOT10k','mot17']:
        root=Path('/root/autodl-pub')/name
        info={'path':str(root),'exists':root.exists(),'files':[]}
        if root.exists():
            for f in sorted(root.iterdir()):
                entry={'name':f.name,'is_directory':f.is_dir()}
                if f.is_file():entry['bytes']=f.stat().st_size
                info['files'].append(entry)
        result['datasets'][name]=info
    result['packages']={}
    for name in ['torch','numpy','pillow','pyyaml','onnx','onnxruntime','msrtc-rans']:
        try:result['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result['packages'][name]=None
    try:
        import torch
        result['cuda_available']=torch.cuda.is_available()
        result['torch_cuda_version']=torch.version.cuda
        if torch.cuda.is_available():
            x=torch.randn(64,64,device='cuda')
            y=x@x
            torch.cuda.synchronize()
            result['cuda_matmul_finite']=bool(y.isfinite().all().item())
    except Exception as e:result['torch_probe_error']=str(e)
    result['ffmpeg_filters']=command(['ffmpeg','-hide_banner','-filters'])
    result['vmaf_filter_available']='libvmaf' in result['ffmpeg_filters'].get('stdout','')
    result['disk']=command(['df','-h','/root/autodl-tmp','/root/autodl-pub'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
