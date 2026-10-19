"""Independent local audit of delivered bytes, frame metrics and acceptance.

No torch/model inference or training code is imported. Run after the full bundle
manifest verification. The audit does not substitute for real decoder replay.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_stream(path):
    with path.open('rb') as f:
        if f.read(8) != b'MLVCPR02':
            raise ValueError('Non-v2 actual stream')
        length = struct.unpack('<I', f.read(4))[0]
        if not 0 < length <= 65536:
            raise ValueError('Invalid header')
        header = json.loads(f.read(length))
        for i in range(header['frames']):
            raw = f.read(4)
            if len(raw) != 4:
                raise ValueError('Missing frame')
            size = struct.unpack('<I', raw)[0]
            if not 0 < size <= 64*1024*1024 or len(f.read(size)) != size:
                raise ValueError('Invalid actual rANS payload')
        if f.read(1):
            raise ValueError('Unexpected bytes')
    if header['protocol'] != 'MLVCS_deployment_ltr64_idr1024_v2':
        raise ValueError('Wrong protocol')
    return header


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    index = read(a.bundle/'artifact-index.json')
    summaries = index['summary']['results']
    frozen = read(a.bundle/'evidence/results/heldout-confirmation/frozen-finalists.json')
    if sha(a.bundle/'data/test-full/manifest.json') != frozen['test_manifest_sha256']:
        raise ValueError('Final test manifest changed')
    pairs = {tuple(x) for x in frozen['registered_pairs']}
    if len(pairs) != 69 or len(frozen['registered_pairs']) != 69:
        raise ValueError('Unexpected acceptance coverage')
    streams, metric_files, records = {}, set(), []
    for result in summaries:
        model = a.bundle/index['models'][result['id']]['file']
        if sha(model) != result['checkpoint_sha256']:
            raise ValueError('Delivered finalist weight mismatch')
        prefix = ('decoder-drift/heldout' if result['id'].startswith('drift_') else 'heldout-confirmation')
        data = read(a.bundle/'evidence/results'/prefix/result['id']/'measurements.json')
        baseline = {(r['sequence'], r['target']): r for r in data['baseline']}
        candidate = {(r['sequence'], r['target']): r for r in data['candidate']}
        if (len(data['baseline']) != 69 or len(data['candidate']) != 69 or
            set(baseline) != pairs or set(candidate) != pairs):
            raise ValueError('Missing/duplicate delivered test point')
        ratios = []
        for key in sorted(pairs):
            b, c = baseline[key], candidate[key]
            for field in ('source_sha256', 'color_profile', 'protocol', 'vmaf_model_sha256', 'metric_build_sha256'):
                if b[field] != c[field]:
                    raise ValueError('Mismatched source/evaluation provenance')
            if b['vmaf_model_sha256'] != sha(a.bundle/'vmaf/vmaf_v0.6.1.json'):
                raise ValueError('Different delivered VMAF model')
            if b['bitstream_sha256'] != c['bitstream_sha256'] or b['actual_bits'] != c['actual_bits']:
                raise ValueError('Bitrate is not exactly identical')
            source = a.bundle/'streams'/(b['bitstream_sha256']+'.mlvcp')
            if b['bitstream_sha256'] not in streams:
                if sha(source) != b['bitstream_sha256']:
                    raise ValueError('Actual stream SHA mismatch')
                streams[b['bitstream_sha256']] = inspect_stream(source)
            header = streams[b['bitstream_sha256']]
            if source.stat().st_size*8 != b['actual_bits']:
                raise ValueError('Actual file bits differ from metric metadata')
            for row in (b, c):
                if any(header[k] != row[k] for k in ('width', 'height', 'frames', 'qp')):
                    raise ValueError('Actual stream geometry/quantizer mismatch')
                if abs(row['duration_seconds']-header['frames']/header['fps']) > 1e-9:
                    raise ValueError('Wrong duration for actual-bitrate calculation')
                path = a.bundle/index['path_map'][row['vmaf_json_path']]
                vmaf = read(path)
                values = [frame['metrics']['vmaf'] for frame in vmaf['frames']]
                if len(values) != row['frames']:
                    raise ValueError('Wrong VMAF frame count')
                if abs(sum(values)/len(values)-row['vmaf']) > 2e-6:
                    raise ValueError('Pooled VMAF disagrees with delivered per-frame JSON')
                if abs(vmaf['pooled_metrics']['vmaf']['mean']-row['vmaf']) > 1e-9:
                    raise ValueError('Wrong measured VMAF')
                metric_files.add(str(path))
            ratios.append(c['vmaf']/b['vmaf'])
        worst = min(ratios)
        status = 'passed' if all(r >= .95 for r in ratios) else 'not_passed'
        if status != result['gate_status'] or abs(worst-result['worst_vmaf_ratio']) > 1e-12:
            raise ValueError('Independent acceptance disagrees')
        records.append({'id': result['id'], 'status': status, 'pairs': 69,
                        'worst_vmaf_ratio': worst, 'actual_bitrate_error': 0.})
    primary = next(r for r in records if r['id'] == index['summary']['primary'])
    if primary['status'] != 'passed':
        raise ValueError('Pretest selected primary failed')
    proof = read(a.bundle/'evidence/results/standalone-replay/results.json')
    if proof['status'] != 'passed' or not all(r['expected_digest_matches'] for r in proof['models']):
        raise ValueError('Standalone replay failed')
    result = {'status': 'passed', 'unique_actual_streams_fully_parsed': len(streams),
              'per_frame_metric_files_recomputed': len(metric_files), 'candidates': records,
              'exact_same_actual_file_bytes_verified': True, 'all69_pairs_per_finalist_verified': True,
              'training_or_inference_code_imported': False, 'standalone_replay_receipt_verified': True}
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
