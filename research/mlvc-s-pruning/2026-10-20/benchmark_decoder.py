"""Cross-ordered, batch-one real rANS decoder timing on the rented H800.

Run only after training and validation finish. No disk writes or VMAF inside the
timed interval. These measurements establish no CPU/ARM/mobile latency claim.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
import torch
from experiment_runtime import ROOT, configure_torch, load_model, save_json, digest
from final_validation import FINAL, FREEZE, load_candidate, read_stream, decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--frames', type=int, default=120)
    p.add_argument('--include-drift', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'results/decoder-benchmark.json')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    source_files = ('benchmark_decoder.py', 'experiment_runtime.py', 'final_validation.py', 'entropy_tables.py')
    source_hashes = {name: digest(ROOT/name) for name in source_files}
    configure_torch(a.threads)
    if a.device == 'cpu':
        compatibility = json.loads((ROOT/'results/cpu-entropy-compatibility.json').read_text())
        if compatibility['status'] != 'passed':
            raise ValueError('CPU entropy compatibility not verified')
    def synchronize():
        if a.device == 'cuda':
            torch.cuda.synchronize()
    confirmation = json.loads((FINAL/'results.json').read_text())
    frozen = json.loads(FREEZE.read_text())
    passed = [r for r in confirmation['results'] if r['gate_status'] == 'passed']
    primary = next((r for r in passed if r['id'] == frozen['primary']), None)
    if primary is None:
        raise ValueError('Primary did not pass; do not silently substitute another architecture')
    compare = [primary]
    best = max(passed, key=lambda r: r['decoder_mac_reduction'])
    if best['id'] != primary['id']:
        compare.append(best)
    if a.include_drift:
        drift = json.loads((ROOT/'results/decoder-drift/results.json').read_text())
        for scope in ('main_decoder', 'active_decoder'):
            candidates = [r for r in drift['results'] if r['gate_status'] == 'passed' and r['scope'] == scope]
            if candidates:
                compare.append(max(candidates, key=lambda r: r['decoder_mac_reduction']))
    baseline_rows = json.loads((FINAL/'baseline.json').read_text())
    sample = next(r for r in baseline_rows if r['sequence'] == 'MOT17-02-FRCNN' and r['target'] == 'q32')
    header, payloads = read_stream(sample['bitstream_path'])
    payloads = payloads[:a.frames]
    models = {'baseline': load_model(a.device)}
    for record in compare:
        models[record['id']] = load_candidate(record, a.device)
    if a.device == 'cpu':
        from entropy_tables import install
        for model in models.values():
            install(model, ROOT/'entropy/encoder-tables.npz')
    hardware = None
    if a.device == 'cuda':
        hardware = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,pstate,clocks.sm,clocks.mem,power.limit',
                                   '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
        if hardware.returncode:
            raise RuntimeError('GPU hardware query failed')
    timings = {name: [] for name in models}
    last = {}
    # Build entropy tables and warm convolution kernels before timing.
    for name, model in models.items():
        for _ in range(2):
            decode(model, header, payloads)
        synchronize()
    order_log = []
    names = list(models)
    for round_index in range(a.rounds):
        shift = round_index%len(names)
        order = names[shift:]+names[:shift]
        if round_index%2:
            order.reverse()
        order_log.append(order)
        for name in order:
            synchronize()
            start = time.perf_counter()
            output = decode(models[name], header, payloads)
            synchronize()
            elapsed = time.perf_counter()-start
            if not bool(output.isfinite().all()):
                raise ValueError('Non-finite benchmark output')
            timings[name].append(elapsed)
            last[name] = float(output.sum().cpu())
    summary = {}
    for name, times in timings.items():
        summary[name] = {'median_seconds': statistics.median(times),
                         'median_ms_per_frame': statistics.median(times)*1000/len(payloads),
                         'median_fps': len(payloads)/statistics.median(times),
                         'min_seconds': min(times), 'max_seconds': max(times), 'runs_seconds': times}
    for name in list(models)[1:]:
        summary[name]['speedup_vs_baseline'] = summary['baseline']['median_seconds']/summary[name]['median_seconds']
        summary[name]['latency_reduction'] = 1-summary[name]['median_seconds']/summary['baseline']['median_seconds']
    cpu_models = sorted(set(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')))
    result = {'backend': ('H800 CUDA FP32; TF32 disabled; no sparse kernels' if a.device == 'cuda' else platform.machine()+' CPU FP32; no sparse kernels'),
              'hardware': hardware.stdout.strip() if a.device == 'cuda' else cpu_models,
              'logical_cpu_affinity_count': len(os.sched_getaffinity(0)),
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'batch_size': 1, 'cpu_threads': a.threads,
              'sequence': sample['sequence'], 'qp': header['qp'], 'visible_size': [header['width'], header['height']],
              'frames_per_run': len(payloads), 'rounds': a.rounds, 'warmup_runs': 2, 'order': order_log,
              'interval_includes': 'rANS CPU entropy decode, tensor transfers, context, recurrent feature and reconstruction inference; input payloads already in RAM',
              'interval_excludes': 'model loading, file IO, metric computation, output pixel transfers',
              'stream_sha256': digest(sample['bitstream_path']), 'results': summary,
              'cpu_arm_mobile_latency_verified': False, 'x86_cpu_latency_verified': a.device == 'cpu',
              'arm_mobile_latency_verified': False, 'final_output_sum_sanity': last}
    if {name: digest(ROOT/name) for name in source_files} != source_hashes:
        raise ValueError('Benchmark source changed during execution; reject the measurement')
    result['source_hashes_at_start_and_finish'] = source_hashes
    result['benchmark_source_sha256'] = source_hashes['benchmark_decoder.py']
    if a.device == 'cpu':
        result['entropy_table_compatibility'] = compatibility
        result['full69_pair_cpu_vmaf_confirmation_performed'] = False
    save_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
