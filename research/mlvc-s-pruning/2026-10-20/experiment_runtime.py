"""Measured MLVC-S research codec, actual rANS streams and independent decode.

Protocol: MLVC-S deployment frame-index offset, LTR8/64 recovery and IDR1024,
with zero-feature initialization and nearest-neighbor chroma. No image codec.
All stream headers and rANS payload bytes count toward bitrate.
"""
import copy
from contextlib import contextmanager
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from pruning_core import compact_ffn

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'vendor/mlvc'
CHECKPOINT_HASH = '1b86b757ddb115342293efb57719d6216c6ee2e459ae796ec41723b5c05ca896'
COLOR = 'BT709 full-range 8-bit 420 canonical; nearest chroma to MLVC; limited-range 420 for VMAF'
MAGIC = b'MLVCPR02'
PROTOCOL = 'MLVCS_deployment_ltr64_idr1024_v2'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def load_model(device='cuda', compact_checkpoint=None):
    sys.path.insert(0, str(SOURCE / 'video'))
    from src.models.dmc_6.dmc_61sb import DMC
    config = yaml.safe_load((SOURCE / 'video/conversion/_full_model/model_configs_example.yaml').read_text())['dmc61sbr_mini_reglu']['params']
    config.pop('disable_feature_reset')
    model = DMC(**config)
    base = ROOT / 'mlvc-s-psnr-v1.ckpt'
    if digest(base) != CHECKPOINT_HASH:
        raise ValueError('Baseline checkpoint digest mismatch')
    model.load_state_dict(torch.load(base, map_location='cpu', weights_only=True), strict=True)
    if compact_checkpoint:
        record = torch.load(compact_checkpoint, map_location='cpu', weights_only=True)
        if record.get('protocol') != PROTOCOL:
            raise ValueError('Compact checkpoint was trained for another experimental protocol')
        if record['baseline_sha256'] != CHECKPOINT_HASH:
            raise ValueError('Compact checkpoint has a different baseline')
        for name, entry in record['layout'].items():
            parent, _, child = name.rpartition('.')
            model.get_submodule(parent)._modules[child] = compact_ffn(model.get_submodule(name), entry['keep'])
        model.load_state_dict(record['state_dict'], strict=True)
    model = model.to(device).eval()
    model._measurement_fingerprint = digest(compact_checkpoint) if compact_checkpoint else CHECKPOINT_HASH
    return model


def save_model(path, model, layout, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'layout': layout, 'baseline_sha256': CHECKPOINT_HASH, 'protocol': PROTOCOL, 'metadata': metadata}, path)
    model._measurement_fingerprint = digest(path)


def canonical_frame(path, size=None):
    """Full-range BT709, box-filtered chroma, then quantized 8-bit 420."""
    with Image.open(path) as image:
        image = image.convert('RGB')
        if size and image.size != tuple(size):
            image = image.resize(tuple(size), Image.Resampling.LANCZOS)
        rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()[None]/255
    h, w = rgb.shape[-2:]
    if h % 2 or w % 2:
        raise ValueError('Even source dimensions required')
    r, g, b = rgb.chunk(3, 1)
    y = .2126*r + .7152*g + .0722*b
    uv = torch.cat(((b-y)/(2*(1-.0722))+.5, (r-y)/(2*(1-.2126))+.5), 1)
    uv = F.avg_pool2d(uv, 2)
    y = (y.clamp(0, 1)*255).round().byte()
    uv = (uv.clamp(0, 1)*255).round().byte()
    raw = y.numpy().tobytes() + uv.numpy().tobytes()
    return raw, unpack_frame(raw, w, h)


def unpack_frame(raw, w, h):
    values = np.frombuffer(raw, dtype=np.uint8).copy()
    if len(values) != w*h*3//2:
        raise ValueError('Invalid canonical frame payload size')
    y = torch.from_numpy(values[:w*h].reshape(1, 1, h, w)).float()/255
    uv = torch.from_numpy(values[w*h:].reshape(1, 2, h//2, w//2)).float()/255
    uv = F.interpolate(uv, scale_factor=2, mode='nearest')
    return F.pad(torch.cat((y, uv), 1), (0, (-w)%16, 0, (-h)%16), mode='replicate')


def metric_frame(x, w, h):
    x = x.detach().float().cpu()[..., :h, :w].clamp(0, 1)
    y = (x[:, :1]*219+16).round().clamp(16, 235).byte()
    uv = (F.avg_pool2d(x[:, 1:], 2)*224+16).round().clamp(16, 240).byte()
    return y.numpy().tobytes()+uv.numpy().tobytes()


def sequence_frames(sequence):
    return [Path(f) for f in sequence['files']]


def frame_index(model, index):
    # conversion/_split_model/_base_split_model.py::_get_fa_idx
    return model.frame_index_map[(index+1) % len(model.frame_index_map)]


def initial_dpb(model, x):
    # The official deployment reference buffer initializes feature and memory
    # to zero. disable_feature_reset=true means RGB is never a reference path.
    return {'ref_frame': torch.zeros_like(x),
            'ref_feature': x.new_zeros(1, 2*model.feature_channels, x.shape[-2]//8, x.shape[-1]//8)}


class DeploymentReferences:
    """Mirror pinned conversion/_frame_loop.py default reference selection.

    IDR resets at 1024. Mark frame8, then multiples of64 as LTR. A later
    LTR-marked frame references the preceding LTR, rather than its predecessor.
    """
    def __init__(self, model):
        self.model = model
        self.previous = None
        self.latest_ltr = None
        self.latest_ltr_index = None
        self.last_reference_index = None

    @staticmethod
    def mark(index):
        current = index%1024
        return current == 8 or (current > 8 and current%64 == 0)

    def reference(self, index, x):
        current = index%1024
        if current == 0:
            self.previous = self.latest_ltr = None
            self.latest_ltr_index = self.last_reference_index = None
            return initial_dpb(self.model, x)
        if self.mark(index) and self.latest_ltr is not None:
            self.last_reference_index = self.latest_ltr_index
            return self.latest_ltr
        self.last_reference_index = index-1
        if self.previous is None:
            raise ValueError('Missing previous reference')
        return self.previous

    def commit(self, index, dpb):
        self.previous = dpb
        if self.mark(index):
            self.latest_ltr = dpb
            self.latest_ltr_index = index


@torch.inference_mode()
def encode(model, sequence, qp, destination):
    with fractional_quantizer(model, qp) as integer_qp:
        return _encode(model, sequence, integer_qp, destination, float(qp))


@contextmanager
def fractional_quantizer(model, qp):
    """Explicit research rate control; interpolate quantizer rows, not bits.

    The entropy table uses floor(qp). Both encoder and decoder use the identical
    interpolated rows. Real file size decides matching; no padding is added.
    """
    if not 0 <= qp <= 63:
        raise ValueError('QP outside [0, 63]')
    lower = int(qp)
    fraction = qp-lower
    changes = []
    try:
        if fraction:
            with torch.no_grad():
                for name in ('q_encoder', 'q_decoder', 'q_feature', 'q_recon'):
                    values = getattr(model, name)
                    for shift in sorted(set(model.qp_shift)):
                        index = lower+shift
                        saved = values[index].clone()
                        changes.append((values, index, saved))
                        values[index].copy_(torch.lerp(saved, values[index+1], fraction))
        yield lower
    finally:
        with torch.no_grad():
            for values, index, saved in changes:
                values[index].copy_(saved)


def _encode(model, sequence, qp, destination, stream_qp):
    """Write actual bytes, then decode from this file with a separate DPB."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    w, h = sequence['width'], sequence['height']
    files = sequence_frames(sequence)
    header = {'protocol': PROTOCOL, 'width': w, 'height': h,
              'frames': len(files), 'fps': sequence['fps'], 'qp': stream_qp,
              'color': COLOR, 'feature_reset': False, 'ltr': True, 'ltr_start_idx': 8,
              'ltr_period': 64, 'iframe_period': 1024, 'proactive_ltr_recovery': True,
              'frame_type_index_offset': 1}
    header_bytes = json.dumps(header, sort_keys=True, separators=(',', ':')).encode()
    reference_path = destination.with_suffix('.reference.yuv')
    start = time.monotonic()
    payload_bytes = []
    encoded_recon = []
    device = next(model.parameters()).device
    with destination.open('wb') as stream, reference_path.open('wb') as reference:
        stream.write(MAGIC+struct.pack('<I', len(header_bytes))+header_bytes)
        references = DeploymentReferences(model)
        for index, file in enumerate(files):
            raw, x = canonical_frame(file, (w, h))
            reference.write(metric_frame(x, w, h))
            x = x.to(device)
            dpb = references.reference(index, x)
            out = model.compress(x, dpb, q_index=int(qp), fa_idx=frame_index(model, index))
            dpb, payload = out['dpb'], out['bit_stream']
            references.commit(index, dpb)
            if not isinstance(payload, bytes):
                raise ValueError('Expected batch-one rANS bytes')
            encoded_recon.append(dpb['ref_frame'].detach().cpu())
            stream.write(struct.pack('<I', len(payload))+payload)
            payload_bytes.append(len(payload))
    decoded_path = destination.with_suffix('.decoded.yuv')
    max_error = 0.
    with destination.open('rb') as stream, decoded_path.open('wb') as decoded:
        if stream.read(8) != MAGIC:
            raise ValueError('Bad research stream signature')
        length = struct.unpack('<I', stream.read(4))[0]
        actual_header = json.loads(stream.read(length))
        if actual_header != header:
            raise ValueError('Research stream header mismatch')
        references = DeploymentReferences(model)
        x = torch.zeros(1, 3, (h+15)//16*16, (w+15)//16*16, device=device)
        for index in range(len(files)):
            length = struct.unpack('<I', stream.read(4))[0]
            payload = stream.read(length)
            if len(payload) != length:
                raise ValueError('Truncated research stream')
            dpb = references.reference(index, x)
            out = model.decompress(payload, dpb, q_index=int(qp), fa_idx=frame_index(model, index),
                                   height=x.shape[-2], width=x.shape[-1])
            dpb = out['dpb']
            references.commit(index, dpb)
            err = (dpb['ref_frame'].cpu()-encoded_recon[index]).abs().max().item()
            max_error = max(max_error, err)
            if err > 1e-5:
                raise ValueError(f'Encoder/independent decoder divergence: {err}')
            decoded.write(metric_frame(dpb['ref_frame'], w, h))
        if stream.read(1):
            raise ValueError('Unexpected trailing bytes')
    expected_raw = len(files)*w*h*3//2
    if decoded_path.stat().st_size != expected_raw or reference_path.stat().st_size != expected_raw:
        raise ValueError('Metric frame count mismatch')
    return {'bitstream_path': str(destination), 'bitstream_sha256': digest(destination),
            'actual_bits': destination.stat().st_size*8,
            'intra_frame_bits': sum(n for i,n in enumerate(payload_bytes) if i%1024 == 0)*8,
            'p_frame_bits': sum(n for i,n in enumerate(payload_bytes) if i%1024 != 0)*8,
            'intra_frames': sum(i%1024 == 0 for i in range(len(files))), 'rate_source': 'actual_bitstream',
            'reference_path': str(reference_path), 'decoded_path': str(decoded_path),
            'source_sha256': digest(reference_path), 'frames': len(files), 'width': w, 'height': h,
            'duration_seconds': len(files)/sequence['fps'], 'color_profile': COLOR,
            'protocol': header['protocol'], 'qp': stream_qp, 'sequence': sequence['id'],
            'independent_decode_max_abs_error': max_error,
            'encode_decode_seconds': time.monotonic()-start}


def vmaf_measure(row, binary, model_path):
    binary, model_path = Path(binary).resolve(), Path(model_path).resolve()
    path = Path(row['bitstream_path']).with_suffix('.vmaf.json')
    args = [str(binary), '-r', row['reference_path'], '-d', row['decoded_path'],
            '-w', str(row['width']), '-h', str(row['height']), '-p', '420', '-b', '8',
            '--model', 'path='+str(model_path), '--threads', '4', '--json', '--output', str(path)]
    completed = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if completed.returncode:
        raise RuntimeError('VMAF failed: '+completed.stderr[-3000:])
    result = json.loads(path.read_text())
    frames = result['frames']
    if len(frames) != row['frames']:
        raise ValueError('VMAF did not score every frame')
    values = np.array([f['metrics']['vmaf'] for f in frames])
    if not np.isfinite(values).all():
        raise ValueError('Non-finite measured VMAF')
    return {**row, 'vmaf': float(result['pooled_metrics']['vmaf']['mean']),
            'vmaf_p05': float(np.quantile(values, .05)), 'vmaf_min': float(values.min()),
            'vmaf_after_first_frame': float(values[1:].mean()),
            'vmaf_json_path': str(path), 'vmaf_model_sha256': digest(model_path),
            'metric_build_sha256': digest(binary)}


def measure(model, sequence, qp, destination, binary, model_path):
    return vmaf_measure(encode(model, sequence, qp, destination), binary, model_path)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


@torch.inference_mode()
def convolution_audit(model, width=640, height=360):
    records, handles = [], []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            def hook(block, inputs, output, name=name):
                records.append({'name': name, 'output_shape': list(output.shape),
                                'macs': int(output.numel()*block.in_channels//block.groups*block.kernel_size[0]*block.kernel_size[1])})
            handles.append(module.register_forward_hook(hook))
    x = torch.zeros(1, 3, (height+15)//16*16, (width+15)//16*16, device=next(model.parameters()).device)
    try:
        model(x, initial_dpb(model, x), q_index=32, fa_idx=0)
    finally:
        for h in handles:
            h.remove()
    decoder = sum(r['macs'] for r in records if not r['name'].startswith(('encoder.', 'hyper_encoder.', 'bit_estimator_z.')))
    return {'parameters': parameter_count(model), 'visible_size': [width, height],
            'padded_size': list(x.shape[-2:][::-1]), 'convolution_macs': sum(r['macs'] for r in records),
            'decoder_convolution_macs': decoder, 'definition': 'Executed Conv2d MACs; excludes bias, activations, arithmetic and rANS; zero-feature initialization', 'layers': records}


def configure_torch(threads=8):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--split', default='selection')
    p.add_argument('--vmaf', required=True)
    p.add_argument('--vmaf-model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    configure_torch()
    manifest = json.loads(a.manifest.read_text())
    sequence = manifest[a.split][0]
    sequence = {**sequence, 'files': sequence['files'][:4]}
    model = load_model(a.device)
    row = measure(model, sequence, 32, a.output/'smoke.mlvcp', a.vmaf, a.vmaf_model)
    save_json(a.output/'smoke.json', row)
    print(json.dumps(row, indent=2))
