"""Run the verified v2 stages with at most two concurrent GPU processes.

The agent retains AutoDL lifecycle control; this script cannot power instances
off or claim successful artifact retrieval. Failures stop only owned children.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT/'logs'
PROTOCOL = 'MLVCS_deployment_ltr64_idr1024_v2'
children = []
handles = []


def event(stage, **fields):
    print(json.dumps({'stage': stage, **fields}), flush=True)


def spawn(stage, script, *args):
    log = (LOGS/(stage+'.log')).open('w')
    handles.append(log)
    command = ['bash', 'run_stage.sh', stage, sys.executable, '-u', script, *args]
    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    children.append(child)
    event(stage, status='started', pid=child.pid, command=command)
    return child


def wait(child, stage, watch=None):
    while child.poll() is None:
        if watch is not None and watch.poll() not in (None, 0):
            raise RuntimeError('Concurrent recovery process failed; inspect its log')
        time.sleep(2)
    event(stage, status='completed' if child.returncode == 0 else 'failed', exitcode=child.returncode)
    if child.returncode:
        raise RuntimeError(f'{stage} failed')


def run(stage, script, *args):
    child = spawn(stage, script, *args)
    wait(child, stage)


def main():
    LOGS.mkdir(exist_ok=True)
    qa = json.loads((ROOT/'results/baseline-protocol-qa/report.json').read_text())
    if qa['protocol'] != PROTOCOL or qa['independent_decode_max_abs_error'] > 1e-5:
        raise ValueError('Corrected baseline QA missing or invalid')
    event('pipeline', protocol=PROTOCOL, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          baseline_qa_vmaf=qa['vmaf'], max_gpu_processes=2)
    deadline = time.monotonic()+900
    while not (ROOT/'results/screen/completion.json').exists():
        exitfile = LOGS/'screen-v2.exitcode'
        if exitfile.exists() and exitfile.read_text().strip() != '0':
            raise RuntimeError('Existing v2 screen failed')
        if time.monotonic() > deadline:
            raise TimeoutError('Timed out waiting for v2 screen')
        time.sleep(3)
    env = json.loads((ROOT/'results/screen-environment.json').read_text())
    if env.get('protocol') != PROTOCOL:
        raise ValueError('Screen protocol mismatch')
    run('structure-v2', 'structure_search.py')
    recovery = spawn('recovery-v2', 'run_suite.py', '--stage', 'recover', '--steps', '1000')
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    expected = [ROOT/'results/selection/baseline'/(s['id']+f'_q{qp}.json')
                for s in manifest['selection'] for qp in (12, 32, 52)]
    deadline = time.monotonic()+600
    while not all(path.exists() for path in expected):
        if recovery.poll() not in (None, 0):
            raise RuntimeError('Recovery baseline preparation failed')
        if time.monotonic() > deadline:
            raise TimeoutError('Selection baseline preparation timeout')
        time.sleep(2)
    for path in expected:
        if json.loads(path.read_text()).get('protocol') != PROTOCOL:
            raise ValueError('Shared selection cache protocol mismatch')
    head = spawn('head-v2', 'head_recovery.py', '--steps', '2000')
    wait(head, 'head-v2', watch=recovery)
    for seed in (20261020, 20261021):
        replica = spawn(f'head-seed{str(seed)[-2:]}-v2', 'head_recovery.py', '--steps', '2000',
                        '--seed', str(seed), '--only', 'auto_primary',
                        '--teacher-bank', 'results/head-recovery/teacher-bank.pt',
                        '--output', f'results/head-seed{str(seed)[-2:]}')
        wait(replica, f'head-seed{str(seed)[-2:]}-v2', watch=recovery)
    heldout = spawn('heldout-v2', 'final_validation.py', '--stage', 'all')
    wait(heldout, 'heldout-v2', watch=recovery)
    wait(recovery, 'recovery-v2')
    run('benchmark-v2', 'benchmark_decoder.py')
    event('pipeline', status='experiments_completed_artifact_retrieval_required')


if __name__ == '__main__':
    try:
        main()
    finally:
        for child in children:
            if child.poll() is None:
                # Terminate only the owned stage and its immediate Python child.
                records = subprocess.run(['ps', '-eo', 'pid=,ppid='], capture_output=True, text=True, check=True)
                for line in records.stdout.splitlines():
                    pid, parent = map(int, line.split())
                    if parent == child.pid:
                        subprocess.run(['kill', '-TERM', str(pid)], check=False)
                child.terminate()
        for handle in handles:
            handle.close()
