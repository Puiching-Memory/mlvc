"""Standalone CLI replay of completed finalists, checking exact decoded pixels."""
import json
import subprocess
import sys
from pathlib import Path
from experiment_runtime import ROOT, digest, save_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    final = ROOT/'results/heldout-confirmation'
    head = read(final/'results.json')
    drift = read(ROOT/'results/decoder-drift/results.json')
    passed = [r for r in head['results']+drift['results'] if r['gate_status'] == 'passed']
    primary = next(r for r in passed if r['id'] == head['primary'])
    best = max(passed, key=lambda r: r['decoder_mac_reduction'])
    chosen = {r['id']: r for r in (primary, best)}
    proof = []
    for identity, record in chosen.items():
        directory = (ROOT/'results/decoder-drift/heldout' if identity.startswith('drift_') else final)/identity
        # A full600-frame video exercises numerous LTR transitions.
        row = read(directory/'MOT17-02-FRCNN_q32.json')
        output = ROOT/'results/standalone-replay'/(identity+'.yuv')
        command = [sys.executable, 'decode_stream.py', '--stream', row['bitstream_path'],
                   '--checkpoint', record['checkpoint'], '--output', str(output),
                   '--expected-sha256', row['decoded_sha256']]
        if record['kind'] == 'extended':
            command.append('--structured')
        subprocess.run(command, cwd=ROOT, check=True, timeout=600)
        result = read(output.with_suffix('.replay.json'))
        result.update({'id': identity, 'checkpoint_sha256': digest(record['checkpoint']),
                       'protocol': row['protocol'], 'reference_expected_from_frozen_test': True})
        proof.append(result)
    save_json(ROOT/'results/standalone-replay/results.json', {'status': 'passed', 'models': proof,
                  'fresh_process_per_model': True, 'original_rgb_needed': False})


if __name__ == '__main__':
    main()
