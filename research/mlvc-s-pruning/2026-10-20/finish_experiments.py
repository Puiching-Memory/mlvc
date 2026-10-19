"""Serialize post-training benchmarks, standalone replay and evidence export."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(stage, script, *args):
    with (ROOT/'logs'/(stage+'.log')).open('w') as log:
        print(json.dumps({'stage': stage, 'status': 'started'}), flush=True)
        subprocess.run(['bash', 'run_stage.sh', stage, sys.executable, '-u', script, *args],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({'stage': stage, 'status': 'completed'}), flush=True)


def main():
    exitfile = ROOT/'logs/queue-drift-v2.exitcode'
    deadline = time.monotonic()+3*3600
    print(json.dumps({'status': 'waiting_for_drift_experiments', 'additional_gpu_processes': 0}), flush=True)
    while not exitfile.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Drift queue did not finish')
        time.sleep(5)
    if exitfile.read_text().strip() != '0':
        raise RuntimeError('Drift experiments failed; inspect before exporting')
    if not (ROOT/'results/decoder-benchmark-combined.json').exists():
        run('benchmark-combined-v2', 'benchmark_decoder.py', '--include-drift', '--output',
            'results/decoder-benchmark-combined.json')
    run('benchmark-cpu-v2', 'benchmark_decoder.py', '--include-drift', '--device', 'cpu',
        '--threads', '8', '--rounds', '5', '--frames', '70', '--output', 'results/decoder-benchmark-cpu.json')
    run('replay-final-v2', 'replay_final_models.py')
    run('export-v2', 'export_artifacts.py')
    print(json.dumps({'status': 'ready_for_retrieval_full_digest_verification_and_platform_shutdown'}), flush=True)


if __name__ == '__main__':
    main()
