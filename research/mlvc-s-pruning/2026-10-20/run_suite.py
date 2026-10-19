"""Finite, auditable pruning suite. Only completed measurements are results.

The recovery schedules are explicit research implementations, not claims of
faithfully reproducing named papers. Test data never drives pruning or training.
"""
import argparse
import copy
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
from experiment_runtime import (ROOT, PROTOCOL, load_model, save_model, save_json, digest, canonical_frame,
                                initial_dpb, frame_index, configure_torch, measure, encode,
                                vmaf_measure, parameter_count, convolution_audit)
from pruning_core import prune_model, gated_ffns, compact_ffn, ChannelCalibration
from quality_gate import evaluate

RANKINGS = ('l2', 'l1', 'fpgm', 'activation_rms', 'taylor', 'random')
SCOPES = ('reconstruction', 'main_decoder', 'active_decoder')
FRACTIONS = (.125, .25, .375, .5, .625, .75)
QPS = (12, 32, 52)
RECOVERY = ('rd_finetune', 'rd_teacher_distillation', 'gradual_channel_gating',
            'mask_decay_schedule', 'gradient_decay_staged_distillation')


def weighted_mse(a, b):
    e = (a-b).square().mean((0, 2, 3))
    return (6*e[0]+e[1]+e[2])/8


def frame_tensor(path, sequence, device):
    return canonical_frame(path, (sequence['width'], sequence['height']))[1].to(device)


def collect_calibration(base, manifest, output):
    collector = ChannelCalibration(base, scope='active_decoder')
    base.eval().requires_grad_(True)
    records = []
    try:
        for index, sequence in enumerate(manifest['calibration']):
            qp = QPS[index % len(QPS)]
            dpb = None
            for f, path in enumerate(sequence['files'][:3]):
                x = frame_tensor(path, sequence, 'cuda')
                if dpb is None:
                    dpb = initial_dpb(base, x)
                base.zero_grad(set_to_none=True)
                out = base(x, dpb, q_index=qp, fa_idx=frame_index(base, f))
                loss = out['bpp'].mean() + 100*weighted_mse(out['x_hat'], x)
                loss.backward()
                dpb = {k: v.detach() for k, v in out['dpb'].items()}
            records.append({'sequence': sequence['id'], 'qp': qp, 'frames': 3})
        values = collector.result()
        expected = {name for name, _ in gated_ffns(base, 'active_decoder')}
        if set(values) != expected or any('taylor' not in row for row in values.values()):
            raise ValueError('Every pruned active block needs real RMS and Taylor evidence')
        save_json(output, {'scores': values, 'inputs': records, 'loss': 'estimated bpp + 100 * 6:1:1 YUV MSE',
                           'scope': 'active_decoder; inactive feature_adaptor_i excluded',
                           'manifest_sha256': digest(ROOT/'data/manifest.json')})
        return values
    finally:
        collector.close()
        base.zero_grad(set_to_none=True)


def cached_measure(model, sequence, qp, directory, binary, vmaf_model, target):
    key = f"{sequence['id']}_q{qp:g}"
    path = directory/(key+'.json')
    if path.exists():
        result = json.loads(path.read_text())
        if result.get('protocol') != PROTOCOL or result.get('checkpoint_sha256') != model._measurement_fingerprint:
            raise ValueError('Cached measurement belongs to another protocol or checkpoint')
        if result['source_manifest_sha256'] != digest(ROOT/'data/manifest.json'):
            raise ValueError('Cached measurement belongs to another frozen manifest')
        if not Path(result['bitstream_path']).exists() or digest(result['bitstream_path']) != result['bitstream_sha256']:
            raise ValueError('Cached bitstream missing or modified')
        return result
    result = measure(model, sequence, qp, directory/(key+'.mlvcp'), binary, vmaf_model)
    result.update({'target': target, 'source_manifest_sha256': digest(ROOT/'data/manifest.json'),
                   'checkpoint_sha256': model._measurement_fingerprint})
    save_json(path, result)
    return result


def gate(baseline, candidate):
    document = {'registered_pairs': [[r['sequence'], r['target']] for r in baseline],
                'baseline': baseline, 'candidate': candidate}
    return evaluate(document)


def screen(a, manifest):
    result_root = a.output/'screen'
    result_root.mkdir(parents=True, exist_ok=True)
    # Keep the ranking/pruning source free of non-copyable C++ entropy handles.
    base = load_model()
    baseline_codec = load_model()
    audit = convolution_audit(base)
    save_json(a.output/'baseline-audit.json', audit)
    calibration_path = a.output/'calibration.json'
    calibration = (json.loads(calibration_path.read_text())['scores'] if calibration_path.exists()
                   else collect_calibration(base, manifest, calibration_path))
    base.eval()
    selection = manifest['selection'][:a.screen_clips]
    baseline = [cached_measure(baseline_codec, sequence, qp, result_root/'baseline', a.vmaf, a.vmaf_model, f'q{qp}')
                for sequence in selection for qp in QPS]
    save_json(result_root/'baseline.json', baseline)
    index_path = result_root/'index.json'
    results = json.loads(index_path.read_text()) if index_path.exists() else []
    done = {r['id'] for r in results}
    for scope in SCOPES:
        for method in RANKINGS:
            for fraction in FRACTIONS:
                identity = f'{scope}_{method}_p{int(fraction*1000):03}'
                if identity in done:
                    continue
                start = time.monotonic()
                model, layout = prune_model(base, method, fraction, scope, calibration, seed=20261019)
                model.eval()
                path = a.output/'models'/(identity+'.pt')
                metadata = {'id': identity, 'scope': scope, 'ranking': method, 'fraction': fraction,
                            'recovery': 'none', 'seed': 20261019,
                            'source_manifest_sha256': digest(a.manifest)}
                save_model(path, model, layout, metadata)
                rows = [cached_measure(model, sequence, qp, result_root/identity, a.vmaf, a.vmaf_model, f'q{qp}')
                        for sequence in selection for qp in QPS]
                checked = gate(baseline, rows)
                compact_audit = convolution_audit(model)
                save_json(result_root/identity/'audit.json', compact_audit)
                record = {**metadata, 'checkpoint': str(path), 'layout': layout, 'gate': checked,
                          'parameters': parameter_count(model),
                          'parameter_reduction': 1-parameter_count(model)/audit['parameters'],
                          'decoder_mac_reduction': 1-compact_audit['decoder_convolution_macs']/audit['decoder_convolution_macs'],
                          'conv_mac_reduction': 1-compact_audit['convolution_macs']/audit['convolution_macs'],
                          'worst_vmaf_ratio': min(r['vmaf']/b['vmaf'] for r, b in zip(rows, baseline)),
                          'worst_rate_error': max(abs(r['actual_bits']/b['actual_bits']-1) for r, b in zip(rows, baseline)),
                          'bit_identical_to_baseline': all(r['bitstream_sha256'] == b['bitstream_sha256'] for r, b in zip(rows, baseline)),
                          'screen_sequences': [s['id'] for s in selection], 'seconds': time.monotonic()-start}
                results.append(record)
                save_json(index_path, results)
                print(json.dumps({k: record[k] for k in ('id', 'parameter_reduction', 'decoder_mac_reduction', 'worst_vmaf_ratio', 'worst_rate_error')}, ensure_ascii=False), flush=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()
    save_json(result_root/'completion.json', {'status': 'completed', 'variants': len(results),
                                            'expected': len(SCOPES)*len(RANKINGS)*len(FRACTIONS),
                                            'screen_only': True, 'untouched_test_not_evaluated': True})


def set_training_parameters(model, scope):
    prefixes = {'reconstruction': ('recon_generation_net.', 'q_recon'),
                'main_decoder': ('feature_extractor.', 'decoder.', 'recon_generation_net.', 'q_feature', 'q_decoder', 'q_recon'),
                'active_decoder': ('feature_adaptor_p.', 'feature_extractor.', 'hyper_decoder.',
                                   'temporal_prior_encoder.', 'y_prior_fusion.', 'y_spatial_prior.',
                                   'decoder.', 'recon_generation_net.', 'q_feature', 'q_decoder', 'q_recon')}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes[scope]))
    return [p for p in model.parameters() if p.requires_grad]


def training_batch(manifest, rng, batch_size, patch_size, frames, device):
    items = []
    for _ in range(batch_size):
        sequence = rng.choice(manifest['train'])
        start = rng.randrange(len(sequence['files'])-frames+1)
        x = [frame_tensor(path, sequence, device) for path in sequence['files'][start:start+frames]]
        h, w = x[0].shape[-2:]
        top = rng.randrange((h-patch_size)//2+1)*2
        left = rng.randrange((w-patch_size)//2+1)*2
        items.append([frame[..., top:top+patch_size, left:left+patch_size] for frame in x])
    return [torch.cat([item[i] for item in items], 0) for i in range(frames)]


def training_dpb(model, x):
    state = initial_dpb(model, x[:1])
    return {k: v.expand(x.shape[0], -1, -1, -1).clone() for k, v in state.items()}


def recover_one(base, teacher, record, method, manifest, steps, output, seed):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    dense = method in ('gradual_channel_gating', 'mask_decay_schedule', 'gradient_decay_staged_distillation')
    if dense:
        student = copy.deepcopy(base)
        layout = record['layout']
    else:
        student = load_model(compact_checkpoint=record['checkpoint'])
        layout = record['layout']
    parameters = set_training_parameters(student, record['scope'])
    student.train()
    optimizer = torch.optim.Adam(parameters, lr=3e-5)
    handles = []
    control = {'factor': 1., 'gradient_factor': 1.}
    if dense:
        for name, entry in layout.items():
            block = student.get_submodule(name)
            width = block.ffn[2].in_channels
            keep = torch.zeros(width, device='cuda')
            keep[entry['keep']] = 1
            def mask_hook(module, inputs, value, keep=keep):
                return value*(keep+(1-keep)*control['factor'])[None, :, None, None]
            handles.append(block.ffn[1].register_forward_hook(mask_hook))
            if method == 'gradient_decay_staged_distillation':
                paired = torch.cat((keep, keep))
                if block.ffn[0].weight.requires_grad:
                    handles.append(block.ffn[0].weight.register_hook(lambda g, mask=paired: g*(mask+(1-mask)*control['gradient_factor'])[:, None, None, None]))
                if block.ffn[2].weight.requires_grad:
                    handles.append(block.ffn[2].weight.register_hook(lambda g, mask=keep: g*(mask+(1-mask)*control['gradient_factor'])[None, :, None, None]))
    log = []
    try:
        for step in range(steps):
            progress = (step+1)/steps
            control['factor'] = (1-progress)**2 if method == 'mask_decay_schedule' else max(0., 1-progress/.8)
            control['gradient_factor'] = 1-progress
            frames = training_batch(manifest, rng, 4, 128, 3, 'cuda')
            qp = rng.choice(QPS)
            lam = math.exp(math.log(768)*qp/63)
            dpb = training_dpb(student, frames[0])
            teacher_dpb = training_dpb(teacher, frames[0])
            loss = 0.
            distortion = 0.
            rate = 0.
            for index, x in enumerate(frames):
                result = student(x, dpb, q_index=qp, fa_idx=frame_index(student, index))
                dpb = result['dpb']
                mse = weighted_mse(result['x_hat'], x)
                bpp = result['bpp'].mean()
                frame_loss = bpp+lam*mse
                if method in ('rd_teacher_distillation', 'gradient_decay_staged_distillation'):
                    with torch.no_grad():
                        target = teacher(x, teacher_dpb, q_index=qp, fa_idx=frame_index(teacher, index))
                        teacher_dpb = target['dpb']
                    kd = weighted_mse(result['x_hat'], target['x_hat'])
                    feature_kd = (dpb['ref_feature']-teacher_dpb['ref_feature']).square().mean()
                    weight = 1. if method == 'rd_teacher_distillation' else min(1., progress*2)
                    frame_loss = frame_loss+weight*lam*(kd+.05*feature_kd)
                loss = loss+frame_loss/len(frames)
                distortion += float(mse.detach())/len(frames)
                rate += float(bpp.detach())/len(frames)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            if step == 0 or (step+1)%100 == 0:
                row = {'step': step+1, 'loss': float(loss.detach()), 'mse': distortion, 'estimated_bpp': rate,
                       'qp': qp, 'gate_factor': control['factor'], 'gradient_norm': float(norm)}
                log.append(row)
                save_json(output.with_suffix('.training.json'), {'seed': seed, 'method': method, 'log': log,
                          'training_only_estimated_rate': True, 'input_manifest_sha256': digest(ROOT/'data/manifest.json')})
                torch.save({'step': step+1, 'model': student.state_dict(), 'optimizer': optimizer.state_dict(),
                            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                            'python_rng': rng.getstate(), 'method': method, 'layout': layout}, output.with_suffix('.resume.pt'))
                print(json.dumps({'training': output.stem, **row}), flush=True)
        for h in handles:
            h.remove()
        handles = []
        if dense:
            for name, entry in layout.items():
                parent, _, child = name.rpartition('.')
                student.get_submodule(parent)._modules[child] = compact_ffn(student.get_submodule(name), entry['keep'])
        student.eval()
        metadata = {**record, 'id': output.stem, 'recovery': method, 'steps': steps, 'seed': seed,
                    'recipe': 'batch4 patch128 frames3 BPTT; Adam3e-5; 6:1:1 YUV MSE with estimated bpp; QP12/32/52; optional output+feature KD',
                    'scope_constraint': 'Only named decoder scope and its quantizers trained; encoder and entropy tables frozen'}
        save_model(output, student, layout, metadata)
        return student, metadata
    finally:
        for h in handles:
            h.remove()


def recover(a, manifest):
    screened = json.loads((a.output/'screen/index.json').read_text())
    selected = []
    for scope in SCOPES:
        candidates = [r for r in screened if r['scope'] == scope]
        passing = [r for r in candidates if r['gate']['status'] == 'passed']
        chosen = (max(passing, key=lambda r: r['decoder_mac_reduction']) if passing else
                  max([r for r in candidates if r['fraction'] == min(FRACTIONS)], key=lambda r: r['worst_vmaf_ratio']))
        selected.append(chosen)
    save_json(a.output/'recovery-selection.json', {'selection_only': True, 'candidates': selected,
                                                 'test_accessed': False})
    index_path = a.output/'recovery/index.json'
    index_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(index_path.read_text()) if index_path.exists() else []
    done = {r['id'] for r in results}
    base = load_model()
    teacher = load_model().eval().requires_grad_(False)
    baseline_codec = load_model()
    selection = manifest['selection']
    baseline = [cached_measure(baseline_codec, sequence, qp, a.output/'selection/baseline', a.vmaf, a.vmaf_model, f'q{qp}')
                for sequence in selection for qp in QPS]
    for source in selected:
        for method in RECOVERY:
            identity = source['id']+'_'+method+f'_seed{a.seed}'
            if identity in done:
                continue
            path = a.output/'models'/(identity+'.pt')
            model, metadata = recover_one(base, teacher, source, method, manifest, a.steps, path, a.seed)
            rows = [cached_measure(model, sequence, qp, a.output/'selection'/identity, a.vmaf, a.vmaf_model, f'q{qp}')
                    for sequence in selection for qp in QPS]
            record = {**metadata, 'checkpoint': str(path), 'gate': gate(baseline, rows),
                      'worst_vmaf_ratio': min(r['vmaf']/b['vmaf'] for r, b in zip(rows, baseline)),
                      'worst_rate_error': max(abs(r['actual_bits']/b['actual_bits']-1) for r, b in zip(rows, baseline)),
                      'parameters': parameter_count(model)}
            results.append(record)
            save_json(index_path, results)
            print(json.dumps({'recovery_finished': identity, 'gate': record['gate']['status'], 'worst_vmaf_ratio': record['worst_vmaf_ratio']}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    save_json(a.output/'recovery/completion.json', {'status': 'completed', 'variants': len(results),
                                                  'expected_this_seed': len(selected)*len(RECOVERY)})


def matched_measure(model, sequence, qp, target_row, directory, a):
    """Measured size only. QP search cannot use VMAF to select the rate."""
    directory.mkdir(parents=True, exist_ok=True)
    target_bits = target_row['actual_bits']
    attempts = []
    def trial(value):
        path = directory/(sequence['id']+f'_target{qp}_trial{len(attempts):02}.mlvcp')
        row = encode(model, sequence, value, path)
        attempts.append(row)
        return row
    middle = trial(float(qp))
    if abs(middle['actual_bits']/target_bits-1) > .01:
        low, high = max(0., qp-8.), min(63., qp+8.)
        left, right = trial(low), trial(high)
        increasing = right['actual_bits'] > left['actual_bits']
        for _ in range(12):
            value = (low+high)/2
            row = trial(value)
            if abs(row['actual_bits']/target_bits-1) <= .01:
                break
            if (row['actual_bits'] < target_bits) == increasing:
                low = value
            else:
                high = value
    best = min(attempts, key=lambda r: abs(r['actual_bits']/target_bits-1))
    result = vmaf_measure(best, a.vmaf, a.vmaf_model)
    result.update({'target': target_row['target'], 'source_manifest_sha256': digest(a.manifest),
                   'rate_search': [{'qp': r['qp'], 'actual_bits': r['actual_bits']} for r in attempts],
                   'rate_match_tolerance': .01, 'rate_search_uses_quality': False})
    save_json(directory/(sequence['id']+f'_target{qp}.json'), result)
    return result


def final(a, manifest):
    screen_rows = json.loads((a.output/'screen/index.json').read_text())
    recovery_path = a.output/'recovery/index.json'
    recovered = json.loads(recovery_path.read_text()) if recovery_path.exists() else []
    candidates = sorted(screen_rows+recovered, key=lambda r: r['decoder_mac_reduction'], reverse=True)
    baseline_codec = load_model()
    validation_baseline = [cached_measure(baseline_codec, s, qp, a.output/'selection/baseline', a.vmaf, a.vmaf_model, f'q{qp}')
                           for s in manifest['selection'] for qp in QPS]
    finalists = []
    # Test data remains untouched until the finalist identities are frozen below.
    for scope in SCOPES:
        for candidate in [r for r in candidates if r['scope'] == scope]:
            if candidate['worst_vmaf_ratio'] < .93:
                continue
            model = load_model(compact_checkpoint=candidate['checkpoint'])
            rows = [matched_measure(model, s, qp, b, a.output/'matched-selection'/candidate['id'], a)
                    for (s, qp), b in zip([(s, q) for s in manifest['selection'] for q in QPS], validation_baseline)]
            checked = gate(validation_baseline, rows)
            save_json(a.output/'matched-selection'/candidate['id']/'gate.json', checked)
            del model
            if checked['status'] == 'passed':
                finalists.append(candidate)
                break
    if not finalists:
        save_json(a.output/'final/status.json', {'status': 'no_validation_candidate_passed', 'test_accessed': False})
        return
    freeze = {'protocol': PROTOCOL, 'manifest_sha256': digest(a.manifest),
              'finalists': finalists, 'test_results_used_for_selection': False,
              'warning': 'No tuning or replacement after held-out test results'}
    freeze_path = a.output/'final/frozen-finalists.json'
    if freeze_path.exists():
        if json.loads(freeze_path.read_text()) != freeze:
            raise ValueError('Cannot replace frozen finalists after testing')
    else:
        save_json(freeze_path, freeze)
    test_baseline = [cached_measure(baseline_codec, s, qp, a.output/'test/baseline', a.vmaf, a.vmaf_model, f'q{qp}')
                     for s in manifest['test'] for qp in QPS]
    save_json(a.output/'final/baseline.json', test_baseline)
    reports = []
    for candidate in finalists:
        model = load_model(compact_checkpoint=candidate['checkpoint'])
        rows = [matched_measure(model, s, qp, b, a.output/'test'/candidate['id'], a)
                for (s, qp), b in zip([(s, q) for s in manifest['test'] for q in QPS], test_baseline)]
        checked = gate(test_baseline, rows)
        save_json(a.output/'final'/(candidate['id']+'.measurements.json'),
                  {'registered_pairs': [[r['sequence'], r['target']] for r in test_baseline],
                   'baseline': test_baseline, 'candidate': rows})
        save_json(a.output/'final'/(candidate['id']+'.gate.json'), checked)
        reports.append({'id': candidate['id'], 'checkpoint': candidate['checkpoint'],
                        'parameters': parameter_count(model), 'decoder_mac_reduction': candidate['decoder_mac_reduction'],
                        'gate': checked, 'test_pairs': len(rows),
                        'worst_vmaf_ratio': min(r['vmaf']/b['vmaf'] for r, b in zip(rows, test_baseline)),
                        'worst_rate_error': max(abs(r['actual_bits']/b['actual_bits']-1) for r, b in zip(rows, test_baseline)),
                        'bit_identical_to_baseline': all(r['bitstream_sha256'] == b['bitstream_sha256'] for r, b in zip(rows, test_baseline))})
        del model
    save_json(a.output/'final/results.json', {'status': 'completed', 'finalists_frozen_sha256': digest(freeze_path),
                                            'results': reports, 'test_data_tuned': False})
    print(json.dumps(reports, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['screen', 'recover', 'final'], required=True)
    p.add_argument('--manifest', type=Path, default=ROOT/'data/manifest.json')
    p.add_argument('--output', type=Path, default=ROOT/'results')
    p.add_argument('--vmaf', default=str(ROOT/'vendor/vmaf/libvmaf/build/tools/vmaf'))
    p.add_argument('--vmaf-model', default=str(ROOT/'vendor/vmaf/model/vmaf_v0.6.1.json'))
    p.add_argument('--screen-clips', type=int, default=2)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261019)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    configure_torch()
    manifest = json.loads(a.manifest.read_text())
    for record in manifest['files'].values():
        if digest(record['path']) != record['sha256']:
            raise ValueError('Frozen dataset file modified')
    environment = {'protocol': PROTOCOL, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(), 'manifest_sha256': digest(a.manifest),
                   'source_scripts': {path.name: digest(path) for path in ROOT.glob('*.py')},
                   'vmaf_binary_sha256': digest(a.vmaf), 'vmaf_model_sha256': digest(a.vmaf_model)}
    save_json(a.output/(a.stage+'-environment.json'), environment)
    {'screen': screen, 'recover': recover, 'final': final}[a.stage](a, manifest)


if __name__ == '__main__':
    main()
