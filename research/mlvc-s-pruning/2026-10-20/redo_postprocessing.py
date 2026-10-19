"""Quarantine native CPU-PMF timing and rebuild immutable final postprocessing."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from experiment_runtime import ROOT, save_json


def run(stage, script, *args):
    print(json.dumps({'stage': stage, 'status': 'started'}), flush=True)
    with (ROOT/'logs'/(stage+'.log')).open('w') as log:
        subprocess.run(['bash', 'run_stage.sh', stage, sys.executable, '-u', script, *args],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'stage': stage, 'status': 'completed'}), flush=True)


def main():
    deadline = time.monotonic()+600
    while not (ROOT/'logs/finish-v2.exitcode').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Earlier exporter has not stopped')
        time.sleep(3)
    quarantine = ROOT/'quarantine-native-cpu-pmf'
    if quarantine.exists():
        raise FileExistsError('Inspect existing quarantine before retry')
    quarantine.mkdir()
    sources = [ROOT/'export-v2', ROOT/'mlvc-s-pruning-results-v2.tar.gz', ROOT/'export-receipt.json',
               ROOT/'results/decoder-benchmark-cpu.json', ROOT/'results/decoder-benchmark-combined.json']
    for prefix in ('benchmark-cpu-v2.', 'benchmark-combined-v2.', 'finish-v2.', 'export-v2.'):
        sources.extend((ROOT/'logs').glob(prefix+'*'))
    moved = []
    for source in sources:
        if not source.exists():
            continue
        if not source.resolve().is_relative_to(ROOT.resolve()) or source.resolve() == ROOT.resolve():
            raise ValueError('Quarantine path outside task directory')
        target = quarantine/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        moved.append(str(source.relative_to(ROOT)))
    save_json(ROOT/'results/cpu-pmf-correction.json', {
        'invalid_native_cpu_timing_quarantined': True, 'files': moved,
        'reason': 'CPU/CUDA build different integer entropy tables; older queued CPU process loaded code before new guard was added. Reject old CPU timings and interim bundle.',
        'resolution': 'Freeze actual CUDA encoder integer tables, verify SHA256 and neural entropy weights/geometry, install identical tables for CPU; rerun all final timing from fresh immutable sources.',
        'gpu_vmaf_confirmation_unchanged': True, 'gpu_encoded_bytes_unchanged': True})
    run('benchmark-combined-fixed-v2', 'benchmark_decoder.py', '--include-drift', '--output',
        'results/decoder-benchmark-combined.json')
    run('benchmark-cpu-fixed-v2', 'benchmark_decoder.py', '--include-drift', '--device', 'cpu',
        '--threads', '8', '--rounds', '5', '--frames', '70', '--output', 'results/decoder-benchmark-cpu.json')
    run('replay-final-fixed-v2', 'replay_final_models.py')
    run('export-fixed-v2', 'export_artifacts.py')
    print(json.dumps({'status': 'ready_for_full_local_retrieval_verification_and_shutdown'}), flush=True)


if __name__ == '__main__':
    main()
