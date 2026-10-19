"""Wait for the current two-process pipeline before exclusive GPU drift work."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    exitfile = ROOT/'logs/pipeline-v2.exitcode'
    print(json.dumps({'status': 'waiting_for_existing_pipeline', 'additional_gpu_processes': 0}), flush=True)
    deadline = time.monotonic()+4*3600
    while not exitfile.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Existing pipeline has not finished; no extra GPU process started')
        time.sleep(5)
    for path in ('results/heldout-confirmation/results.json', 'results/recovery/completion.json'):
        if not (ROOT/path).exists():
            raise ValueError('Earlier experiments did not finish; inspect before starting another stage')
    print(json.dumps({'status': 'starting_exclusive_decoder_drift',
                      'earlier_pipeline_exit': int(exitfile.read_text()), 'max_gpu_processes': 1}), flush=True)
    with (ROOT/'logs/decoder-drift-v2.log').open('w') as log:
        subprocess.run(['bash', 'run_stage.sh', 'decoder-drift-v2', sys.executable, '-u',
                        'decoder_drift.py', '--stage', 'all'], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
    with (ROOT/'logs/benchmark-combined-v2.log').open('w') as log:
        subprocess.run(['bash', 'run_stage.sh', 'benchmark-combined-v2', sys.executable, '-u',
                        'benchmark_decoder.py', '--include-drift', '--output',
                        'results/decoder-benchmark-combined.json'], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'status': 'decoder_drift_complete_artifact_retrieval_still_required'}), flush=True)


if __name__ == '__main__':
    main()
