"""Strict per-sequence VMAF retention gate using actual, matched-rate encodes.
Input JSON includes registered_pairs, baseline and candidate lists. Each row requires
sequence, target, source_sha256, frames, width, height, color_profile,
vmaf_model_sha256, metric_build_sha256, actual_bits, duration_seconds, vmaf.
Estimated bits, missing rows and mismatched sources cannot pass this gate.
"""
import argparse
import json
import math
import re
from pathlib import Path


def evaluate(document, threshold=.95, rate_tolerance=.01):
    fields = ['sequence', 'target', 'source_sha256', 'frames', 'width', 'height',
              'color_profile', 'vmaf_model_sha256', 'metric_build_sha256', 'protocol']
    def index(rows):
        result = {}
        for r in rows:
            for f in fields + ['actual_bits', 'duration_seconds', 'vmaf']:
                if f not in r:
                    raise ValueError(f'Missing field {f}')
            key = (r['sequence'], r['target'])
            if key in result:
                raise ValueError(f'Duplicate measurement {key}')
            if any(not isinstance(r[f], (int, float)) or isinstance(r[f], bool)
                   or not math.isfinite(r[f]) for f in ['actual_bits', 'duration_seconds', 'vmaf']):
                raise ValueError(f'Invalid numeric measurement {key}')
            if r['actual_bits'] <= 0 or r['duration_seconds'] <= 0 or r['frames'] <= 0:
                raise ValueError(f'Invalid measurement {key}')
            if any(not isinstance(r[f], int) or isinstance(r[f], bool) or r[f] <= 0
                   for f in ['frames','width','height','actual_bits']):
                raise ValueError(f'Invalid count or dimensions {key}')
            if any(not isinstance(r[f], str) or not re.fullmatch('[0-9a-fA-F]{64}',r[f])
                   for f in ['source_sha256','vmaf_model_sha256','metric_build_sha256']):
                raise ValueError(f'Invalid provenance digest {key}')
            if r.get('rate_source') != 'actual_bitstream':
                raise ValueError(f'Actual bitstream evidence required: {key}')
            if not 0 <= r['vmaf'] <= 100:
                raise ValueError(f'Invalid clipped VMAF {key}')
            result[key] = r
        return result
    baseline, candidate = index(document['baseline']), index(document['candidate'])
    registered = {tuple(x) for x in document['registered_pairs']}
    if not baseline or baseline.keys() != candidate.keys() or set(baseline) != registered:
        raise ValueError('Incomplete or unexpected sequence/target coverage')
    checks = []
    for key, b in baseline.items():
        c = candidate[key]
        if any(b[f] != c[f] for f in fields):
            raise ValueError(f'Source or evaluation protocol mismatch: {key}')
        if not math.isclose(b['duration_seconds'], c['duration_seconds'], rel_tol=0, abs_tol=1e-6):
            raise ValueError(f'Duration mismatch: {key}')
        rate_error = abs(c['actual_bits']/b['actual_bits']-1)
        quality_ok = c['vmaf'] >= threshold*b['vmaf']
        checks.append({'sequence':key[0], 'target':key[1], 'baseline_vmaf':b['vmaf'],
                       'candidate_vmaf':c['vmaf'], 'required_vmaf':threshold*b['vmaf'],
                       'actual_rate_relative_error':rate_error,
                       'status':'rate_mismatch' if rate_error > rate_tolerance else ('passed' if quality_ok else 'quality_failed')})
    return {'status':'passed' if all(c['status']=='passed' for c in checks) else 'not_passed',
            'threshold':threshold, 'rate_tolerance':rate_tolerance,
            'aggregation':'all registered sequence/target pairs must pass', 'checks':checks}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('measurements', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        result = evaluate(json.loads(a.measurements.read_text(encoding='utf-8')))
    except (ValueError, KeyError, TypeError) as e:
        result = {'status':'incomplete_not_passed','error':str(e)}
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['status'])
    return 0 if result['status']=='passed' else 2


if __name__=='__main__':
    raise SystemExit(main())
