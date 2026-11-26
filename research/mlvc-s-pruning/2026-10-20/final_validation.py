"""Frozen held-out confirmation using byte-identical real baseline streams.

Head-only candidates decode the very same rANS files. Their non-head parameters
must be identical to baseline. Worker processes receive no original RGB images
or encoder-side tensors. Acceptance uses zero bitrate tolerance.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
import torch
from experiment_runtime import (ROOT, MAGIC, COLOR, PROTOCOL, DeploymentReferences, load_model, configure_torch, initial_dpb,
                                fractional_quantizer, frame_index, metric_frame, digest,
                                save_json, vmaf_measure, convolution_audit)
from structure_search import load_variant
from run_suite import QPS, cached_measure
from quality_gate import evaluate

RESULTS = ROOT/'results'
FINAL = RESULTS/'heldout-confirmation'
FREEZE = FINAL/'frozen-finalists.json'
TEST_MANIFEST = ROOT/'data/test-full/manifest.json'


def freeze():
    head = json.loads((RESULTS/'head-recovery/index.json').read_text())
    if len(head) != 8 or not (RESULTS/'head-recovery/completion.json').exists():
        raise ValueError('Head structure comparison must finish before test selection')
    passing = [r for r in head if r['gate']['status'] == 'passed' and r['bit_identical_to_baseline']]
    margin = [r for r in passing if r['worst_vmaf_ratio'] >= .98]
    primary = max(margin or passing, key=lambda r: r['decoder_mac_reduction'])
    aggressive = max(passing, key=lambda r: r['decoder_mac_reduction'])
    chosen = []
    def add(record, kind, role):
        if any(r['id'] == record['id'] for r in chosen):
            return
        chosen.append({'id': record['id'], 'checkpoint': record['checkpoint'], 'kind': kind, 'role': role,
                       'checkpoint_sha256': digest(record['checkpoint']),
                       'selection_worst_vmaf_ratio': record['worst_vmaf_ratio'],
                       'decoder_mac_reduction': record['decoder_mac_reduction'],
                       'parameter_reduction': record.get('parameter_reduction'), 'seed': record.get('seed')})
    add(primary, 'extended', 'primary_with_validation_margin')
    add(aggressive, 'extended', 'aggressive_comparison')
    uniform = [r for r in passing if r['id'].startswith('head_ffn_taylor_')]
    if uniform:
        add(max(uniform, key=lambda r: r['decoder_mac_reduction']), 'extended', 'uniform_ffn_comparison')
    # This conservative reference was chosen before seeing any held-out result.
    screen = json.loads((RESULTS/'screen/index.json').read_text())
    conservative = next(r for r in screen if r['id'] == 'reconstruction_taylor_p125')
    add(conservative, 'ffn', 'conservative_reference')
    recovery = json.loads((RESULTS/'recovery/index.json').read_text())
    moderate = next(r for r in recovery if r['id'].startswith('reconstruction_taylor_p500_rd_finetune_'))
    add(moderate, 'ffn', 'uniform_ffn_finetuning_reference')
    for seed in (20261020, 20261021):
        records = json.loads((RESULTS/f'head-seed{str(seed)[-2:]}/index.json').read_text())
        if len(records) != 1 or not (RESULTS/f'head-seed{str(seed)[-2:]}/completion.json').exists():
            raise ValueError('Both additional training seeds must finish before test freeze')
        expected = primary['id'].rsplit('_seed', 1)[0]+f'_seed{seed}'
        if records[0]['id'] != expected:
            raise ValueError('Replica architecture differs from the selected primary')
        add(records[0], 'extended', 'primary_training_seed_replica')
    manifest = json.loads(TEST_MANIFEST.read_text())
    document = {'protocol': PROTOCOL, 'selection_policy': 'Best decoder MAC reduction among full-selection byte-identical heads retaining >=98% VMAF where available; additional comparisons and conservative reference fixed before test',
                'primary': primary['id'], 'candidates': chosen,
                'source_manifest_sha256': digest(ROOT/'data/manifest.json'),
                'test_manifest_sha256': digest(TEST_MANIFEST),
                'test_counts': manifest['counts'],
                'test_sequences': [s['id'] for s in manifest['test']], 'baseline_qps': list(QPS),
                'registered_pairs': [[s['id'], f'q{q}'] for s in manifest['test'] for q in QPS],
                'acceptance': {'vmaf_ratio_min': .95, 'actual_bitrate_tolerance': 0.,
                               'all_registered_pairs_required': True, 'same_actual_bitstream_files': True},
                'test_tuning_forbidden': True}
    document['protocol_qa_note'] = 'MOT17-02 baseline was used only to diagnose and correct omitted official LTR scheduling, QP frame offset and chroma preprocessing in protocol-v1. No MOT candidate quality was used for ranking, training or checkpoint selection; v1 results were archived and excluded.'
    if FREEZE.exists():
        if json.loads(FREEZE.read_text()) != document:
            raise ValueError('Cannot replace frozen test hypotheses')
    else:
        save_json(FREEZE, document)
    return document


def load_candidate(record, device='cuda'):
    if digest(record['checkpoint']) != record['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    return (load_variant(record['checkpoint'], device) if record['kind'] == 'extended'
            else load_model(device, record['checkpoint']))


def coding_state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(('recon_generation_net.', 'q_recon')):
            continue
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def read_stream(path):
    with open(path, 'rb') as f:
        if f.read(8) != MAGIC:
            raise ValueError('Invalid stream')
        size = struct.unpack('<I', f.read(4))[0]
        if not 0 < size <= 65536:
            raise ValueError('Invalid header size')
        header = json.loads(f.read(size))
        if header['protocol'] != PROTOCOL or header['color'] != COLOR:
            raise ValueError('Incompatible protocol')
        if header['feature_reset'] or not header['ltr'] or header['ltr_period'] != 64 or header['iframe_period'] != 1024:
            raise ValueError('Head-only compatibility requires the frozen deployment protocol')
        payloads = []
        for _ in range(header['frames']):
            size_bytes = f.read(4)
            if len(size_bytes) != 4:
                raise ValueError('Missing frame')
            size = struct.unpack('<I', size_bytes)[0]
            if not 0 < size <= 64*1024*1024:
                raise ValueError('Invalid payload size')
            payload = f.read(size)
            if len(payload) != size:
                raise ValueError('Truncated frame')
            payloads.append(payload)
        if f.read(1):
            raise ValueError('Unexpected trailing data')
    return header, payloads


@torch.inference_mode()
def decode(model, header, payloads, output=None):
    w, h = header['width'], header['height']
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, (h+15)//16*16, (w+15)//16*16, device=device)
    references = DeploymentReferences(model)
    with fractional_quantizer(model, header['qp']) as qp:
        for index, payload in enumerate(payloads):
            dpb = references.reference(index, x)
            out = model.decompress(payload, dpb, q_index=qp, fa_idx=frame_index(model, index),
                                   height=x.shape[-2], width=x.shape[-1])
            dpb = out['dpb']
            references.commit(index, dpb)
            if output:
                output.write(metric_frame(dpb['ref_frame'], w, h))
    return dpb['ref_frame']


def baseline(frozen):
    manifest = json.loads(TEST_MANIFEST.read_text())
    if digest(ROOT/'data/manifest.json') != frozen['source_manifest_sha256']:
        raise ValueError('Manifest changed after freeze')
    if digest(TEST_MANIFEST) != frozen['test_manifest_sha256']:
        raise ValueError('Held-out test manifest changed after freeze')
    for record in manifest['files'].values():
        if digest(record['path']) != record['sha256']:
            raise ValueError('Frozen held-out source frame changed')
    model = load_model()
    save_json(FINAL/'coding-state.json', {'baseline_coding_state_sha256': coding_state_digest(model),
                                         'excluded_only': ['recon_generation_net.*', 'q_recon'],
                                         'baseline_checkpoint_sha256': digest(ROOT/'mlvc-s-psnr-v1.ckpt')})
    rows = []
    for s in manifest['test']:
        for qp in QPS:
            row = cached_measure(model, s, qp, FINAL/'baseline', ROOT/'vendor/vmaf/libvmaf/build/tools/vmaf',
                                 ROOT/'vendor/vmaf/model/vmaf_v0.6.1.json', f'q{qp}')
            rows.append(row)
            print(json.dumps({'heldout_baseline': s['id'], 'qp': qp, 'vmaf': row['vmaf'], 'actual_bits': row['actual_bits']}), flush=True)
    save_json(FINAL/'baseline.json', rows)


def variant(frozen, identity):
    record = next(r for r in frozen['candidates'] if r['id'] == identity)
    model = load_candidate(record)
    expected = json.loads((FINAL/'coding-state.json').read_text())['baseline_coding_state_sha256']
    actual = coding_state_digest(model)
    if expected != actual:
        raise ValueError('Candidate changed coding, context, memory or entropy parameters; same-stream validation invalid')
    save_json(FINAL/identity/'coding-compatibility.json', {'outside_head_state_identical': True,
                 'coding_state_sha256': actual, 'checkpoint_sha256': digest(record['checkpoint']),
                 'worker_pid': os.getpid(), 'encoder_invoked_in_worker': False,
                 'original_rgb_inputs_read_in_worker': False, 'deployment_ltr64_idr1024_protocol': True})
    baseline_rows = json.loads((FINAL/'baseline.json').read_text())
    rows = []
    for source in baseline_rows:
        key = source['sequence']+'_'+source['target']
        output = FINAL/identity/(key+'.decoded.yuv')
        stream = FINAL/identity/(key+'.mlvcp')
        if not stream.exists():
            try:
                os.link(source['bitstream_path'], stream)
            except OSError:
                shutil.copyfile(source['bitstream_path'], stream)
        if digest(stream) != source['bitstream_sha256'] or digest(source['reference_path']) != source['source_sha256']:
            raise ValueError('Actual coded bytes or source reference changed')
        header, payloads = read_stream(stream)
        start = time.monotonic()
        with output.open('wb') as f:
            decode(model, header, payloads, f)
        elapsed = time.monotonic()-start
        if output.stat().st_size != header['frames']*header['width']*header['height']*3//2:
            raise ValueError('Decoded frame count mismatch')
        row = {k: source[k] for k in ('sequence', 'target', 'source_sha256', 'frames', 'width', 'height',
                    'duration_seconds', 'color_profile', 'protocol', 'source_manifest_sha256', 'qp')}
        row.update({'actual_bits': stream.stat().st_size*8, 'rate_source': 'actual_bitstream',
                    'checkpoint_sha256': record['checkpoint_sha256'],
                    'bitstream_path': str(stream), 'bitstream_sha256': digest(stream),
                    'decoded_path': str(output), 'reference_path': source['reference_path'],
                    'decode_seconds_with_output_io': elapsed, 'same_bytes_as_baseline': True,
                    'decoded_sha256': digest(output)})
        row = vmaf_measure(row, ROOT/'vendor/vmaf/libvmaf/build/tools/vmaf', ROOT/'vendor/vmaf/model/vmaf_v0.6.1.json')
        rows.append(row)
        save_json(FINAL/identity/(key+'.json'), row)
        print(json.dumps({'heldout_candidate': identity, 'sequence': row['sequence'], 'target': row['target'],
                          'vmaf': row['vmaf'], 'baseline_vmaf': source['vmaf'], 'rate_error': 0}), flush=True)
    document = {'registered_pairs': frozen['registered_pairs'], 'baseline': baseline_rows, 'candidate': rows}
    checked = evaluate(document, rate_tolerance=0.)
    save_json(FINAL/identity/'measurements.json', document)
    save_json(FINAL/identity/'gate.json', checked)
    save_json(FINAL/identity/'audit.json', convolution_audit(model))
    summary = {**record, 'gate_status': checked['status'], 'test_pairs': len(rows),
               'worst_vmaf_ratio': min(r['vmaf']/b['vmaf'] for r, b in zip(rows, baseline_rows)),
               'worst_actual_rate_relative_error': max(abs(r['actual_bits']/b['actual_bits']-1) for r, b in zip(rows, baseline_rows)),
               'all_bitstreams_identical': all(r['bitstream_sha256'] == b['bitstream_sha256'] for r, b in zip(rows, baseline_rows)),
               'checkpoint_parameters_verified': True}
    save_json(FINAL/identity/'summary.json', summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('freeze', 'baseline', 'variant', 'all'), default='all')
    p.add_argument('--variant')
    a = p.parse_args()
    configure_torch()
    FINAL.mkdir(parents=True, exist_ok=True)
    frozen = freeze() if a.stage in ('freeze', 'all') else json.loads(FREEZE.read_text())
    if a.stage in ('baseline', 'all'):
        baseline(frozen)
    if a.stage == 'variant':
        variant(frozen, a.variant)
    elif a.stage == 'all':
        for record in frozen['candidates']:
            target = FINAL/record['id']/'summary.json'
            if target.exists():
                continue
            # Fresh decoder process: no live encoder DPB/latents are shared.
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stage', 'variant', '--variant', record['id']],
                           check=True, timeout=3600)
        reports = [json.loads((FINAL/r['id']/'summary.json').read_text()) for r in frozen['candidates']]
        save_json(FINAL/'results.json', {'status': 'completed', 'frozen_finalists_sha256': digest(FREEZE),
                   'primary': frozen['primary'], 'all_test_pairs_per_candidate': len(frozen['registered_pairs']),
                   'acceptance_bitrate_tolerance': 0., 'results': reports,
                   'weights_or_structure_tuned_on_test': False})
        print(json.dumps({'status': 'completed', 'candidates': len(reports),
                          'passed': [r['id'] for r in reports if r['gate_status'] == 'passed']}), flush=True)


if __name__ == '__main__':
    main()
