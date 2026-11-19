"""Fresh-process MLVC-S stream replay without source video or encoder state."""
import argparse
import json
import struct
from pathlib import Path
import torch
from experiment_runtime import (MAGIC, COLOR, PROTOCOL, DeploymentReferences, load_model, configure_torch, initial_dpb,
                                fractional_quantizer, frame_index, metric_frame, digest, save_json)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--structured', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--expected-sha256')
    p.add_argument('--entropy-tables', type=Path, help='Exact encoder integer tables; required for cross-device replay')
    a = p.parse_args()
    configure_torch()
    if a.structured:
        from structure_search import load_variant
        model = load_variant(a.checkpoint, a.device)
    else:
        model = load_model(a.device, a.checkpoint)
    table_path = a.entropy_tables or Path(__file__).resolve().parent/'entropy/encoder-tables.npz'
    table_receipt = None
    if table_path.exists():
        from entropy_tables import install
        table_receipt = install(model, table_path)
    elif a.device != 'cuda' or a.entropy_tables:
        raise ValueError('Exact encoder entropy tables are required for cross-device decoding')
    with a.stream.open('rb') as stream:
        if stream.read(8) != MAGIC:
            raise ValueError('Invalid stream signature')
        length = struct.unpack('<I', stream.read(4))[0]
        if not 1 <= length <= 65536:
            raise ValueError('Header exceeds format limit')
        header = json.loads(stream.read(length))
        if header['protocol'] != PROTOCOL or header['color'] != COLOR:
            raise ValueError('Unsupported stream protocol')
        width, height, frames = header['width'], header['height'], header['frames']
        if not (isinstance(width, int) and isinstance(height, int) and isinstance(frames, int)
                and 2 <= width <= 8192 and 2 <= height <= 8192 and 1 <= frames <= 100000):
            raise ValueError('Invalid frame dimensions or count')
        a.output.parent.mkdir(parents=True, exist_ok=True)
        x = torch.zeros(1, 3, (height+15)//16*16, (width+15)//16*16, device=a.device)
        references = DeploymentReferences(model)
        with torch.inference_mode(), fractional_quantizer(model, header['qp']) as qp, a.output.open('wb') as output:
            for index in range(frames):
                size_bytes = stream.read(4)
                if len(size_bytes) != 4:
                    raise ValueError('Missing frame length')
                length = struct.unpack('<I', size_bytes)[0]
                if not 1 <= length <= 64*1024*1024:
                    raise ValueError('Invalid frame payload size')
                payload = stream.read(length)
                if len(payload) != length:
                    raise ValueError('Truncated frame')
                dpb = references.reference(index, x)
                result = model.decompress(payload, dpb, q_index=qp, fa_idx=frame_index(model, index),
                                          height=x.shape[-2], width=x.shape[-1])
                dpb = result['dpb']
                references.commit(index, dpb)
                output.write(metric_frame(dpb['ref_frame'], width, height))
            if stream.read(1):
                raise ValueError('Unexpected trailing bytes')
    result = {'status': 'decoded', 'frames': frames, 'stream_sha256': digest(a.stream),
              'decoded_sha256': digest(a.output), 'source_video_needed': False,
              'encoder_state_needed': False, 'checkpoint': str(a.checkpoint) if a.checkpoint else 'baseline'}
    result['frozen_encoder_entropy_tables'] = table_receipt
    if a.expected_sha256 and result['decoded_sha256'] != a.expected_sha256:
        raise ValueError('Fresh-process decoded pixels do not match expected digest')
    result['expected_digest_matches'] = a.expected_sha256 is not None
    save_json(a.output.with_suffix('.replay.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
