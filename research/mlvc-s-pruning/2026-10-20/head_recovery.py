"""Cache frozen teacher features, then recover more aggressive head structures.

Only reconstruction weights/quantizers change, leaving recurrent coding and
all entropy payloads invariant. Real bitstream identity is still verified.
"""
import argparse
import copy
import json
import random
from pathlib import Path
import torch

from experiment_runtime import (ROOT, PROTOCOL, load_model, save_json, digest, configure_torch,
                                frame_index, convolution_audit, parameter_count)
from structure_search import apply_definition, save_variant, load_variant
from run_suite import QPS, training_batch, training_dpb, weighted_mse, cached_measure, gate


def build_bank(manifest, path):
    if path.exists():
        record = torch.load(path, map_location='cpu', weights_only=True)
        if record['manifest_sha256'] != digest(ROOT/'data/manifest.json') or record.get('protocol') != PROTOCOL:
            raise ValueError('Teacher cache belongs to another data split')
        return record
    teacher = load_model().eval().requires_grad_(False)
    rng = random.Random(20261019)
    features, targets, teacher_targets, indices = [], [], [], []
    with torch.inference_mode():
        for batch in range(128):
            frames = training_batch(manifest, rng, 4, 128, 3, 'cuda')
            qp = QPS[batch%len(QPS)]
            dpb = training_dpb(teacher, frames[0])
            for index, x in enumerate(frames):
                fa = frame_index(teacher, index)
                result = teacher(x, dpb, q_index=qp, fa_idx=fa)
                dpb = result['dpb']
                features.append(dpb['ref_feature'][:, :teacher.feature_channels].cpu())
                targets.append(x.cpu())
                teacher_targets.append(result['x_hat'].cpu())
                indices.extend([teacher.shift_qp(qp, fa)]*len(x))
            if (batch+1)%32 == 0:
                print(json.dumps({'teacher_cache_batches': batch+1}), flush=True)
    record = {'features': torch.cat(features), 'target': torch.cat(targets),
              'teacher_target': torch.cat(teacher_targets), 'q_index': torch.tensor(indices),
              'manifest_sha256': digest(ROOT/'data/manifest.json'),
              'protocol': PROTOCOL,
              'recipe': '128 batches of4 random training clips,3 frames,128x128; seed20261019; QP12/32/52; float32',
              'split': 'train_only'}
    torch.save(record, path)
    del teacher
    torch.cuda.empty_cache()
    return record


def candidates():
    screened = json.loads((ROOT/'results/screen/index.json').read_text())
    taylor = {r['fraction']: r for r in screened if r['scope'] == 'reconstruction' and r['ranking'] == 'taylor'}
    structure = json.loads((ROOT/'results/structure/index.json').read_text())
    structural = {r['id']: r for r in structure}
    ffn50 = {name: row['keep'] for name, row in taylor[.5]['layout'].items()}
    for fraction in (.625, .75):
        yield f'head_ffn_taylor_p{int(fraction*1000)}', {'ffn': {name: row['keep'] for name, row in taylor[fraction]['layout'].items()}}
    for fraction in (.25, .5):
        definition = copy.deepcopy(structural[f'reconstruction_outer_p{int(fraction*1000)}']['definition'])
        definition['ffn'] = copy.deepcopy(ffn50)
        yield f'head_outer{int(fraction*100)}_ffn50', definition
    for depth in (1, 2):
        definition = copy.deepcopy(structural[f'reconstruction_remove_{depth}_blocks']['definition'])
        removed = definition['bypass']
        definition['ffn'] = {name: keep for name, keep in ffn50.items() if name not in removed}
        yield f'head_remove{depth}_ffn50', definition
    for target in (15, 20):
        yield f'head_nonuniform_mac{target}', structural[f'nonuniform_reconstruction_mac{target}']['definition']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20261019)
    p.add_argument('--only', help='One preregistered structure name, without seed suffix')
    p.add_argument('--teacher-bank', type=Path, help='Reuse the same frozen training feature bank')
    p.add_argument('--output', type=Path, default=ROOT/'results/head-recovery')
    a = p.parse_args()
    if a.only == 'auto_primary':
        records = json.loads((ROOT/'results/head-recovery/index.json').read_text())
        if len(records) != 8 or not (ROOT/'results/head-recovery/completion.json').exists():
            raise ValueError('Head comparison must finish before primary selection')
        passing = [r for r in records if r['gate']['status'] == 'passed' and r['bit_identical_to_baseline']]
        margin = [r for r in passing if r['worst_vmaf_ratio'] >= .98]
        chosen = max(margin or passing, key=lambda r: r['decoder_mac_reduction'])
        a.only = chosen['id'].rsplit('_kd_seed', 1)[0]
    a.output.mkdir(parents=True, exist_ok=True)
    configure_torch()
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    save_json(a.output/'environment.json', {'protocol': PROTOCOL, 'torch': torch.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(), 'manifest_sha256': digest(ROOT/'data/manifest.json'),
               'source_sha256': digest(Path(__file__)), 'selection_only': True, 'test_accessed': False})
    bank_path = a.teacher_bank or a.output/'teacher-bank.pt'
    bank = build_bank(manifest, bank_path)
    save_json(a.output/'teacher-bank.json', {'sha256': digest(bank_path),
               'frames': len(bank['features']), 'split': bank['split'], 'recipe': bank['recipe'],
               'manifest_sha256': bank['manifest_sha256']})
    bank = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in bank.items()}
    baseline_codec = load_model()
    baseline_audit = convolution_audit(baseline_codec)
    baseline = [cached_measure(baseline_codec, s, q, ROOT/'results/selection/baseline',
                                ROOT/'vendor/vmaf/libvmaf/build/tools/vmaf', ROOT/'vendor/vmaf/model/vmaf_v0.6.1.json', f'q{q}')
                for s in manifest['selection'] for q in QPS]
    path = a.output/'index.json'
    results = json.loads(path.read_text()) if path.exists() else []
    done = {r['id'] for r in results}
    for name, definition in candidates():
        if a.only and name != a.only:
            continue
        identity = name+f'_kd_seed{a.seed}'
        if identity in done:
            continue
        model = apply_definition(load_model(), definition)
        model.train()
        for pname, parameter in model.named_parameters():
            parameter.requires_grad_(pname.startswith(('recon_generation_net.', 'q_recon')))
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-4)
        generator = torch.Generator(device='cuda').manual_seed(a.seed)
        logs = []
        for step in range(a.steps):
            sample = torch.randint(len(bank['features']), (16,), generator=generator, device='cuda')
            prediction = model.unshift_output(model.get_recon(bank['features'][sample], bank['q_index'][sample]))
            mse = weighted_mse(prediction, bank['target'][sample])
            teacher_mse = weighted_mse(prediction, bank['teacher_target'][sample])
            loss = .3*mse+.7*teacher_mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            if step == 0 or (step+1)%250 == 0:
                row = {'step': step+1, 'loss': float(loss.detach()), 'gt_mse': float(mse.detach()),
                       'teacher_mse': float(teacher_mse.detach())}
                logs.append(row)
                save_json(a.output/(identity+'.training.json'), {'loss': '.3 GT MSE + .7 teacher MSE; 6:1:1 YUV',
                           'seed': a.seed, 'batch': 16, 'lr': 1e-4, 'logs': logs,
                           'teacher_bank_sha256': json.loads((a.output/'teacher-bank.json').read_text())['sha256']})
                torch.save({'step': step+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'sampler_rng': generator.get_state(), 'definition': definition}, a.output/(identity+'.resume.pt'))
                print(json.dumps({'training': identity, **row}), flush=True)
        model.eval()
        checkpoint = a.output/(identity+'.pt')
        save_variant(checkpoint, model, definition, {'id': identity, 'steps': a.steps, 'seed': a.seed,
                                                    'recovery': 'cached_teacher_head_distillation'})
        del model
        model = load_variant(checkpoint)
        rows = [cached_measure(model, s, q, a.output/identity,
                                ROOT/'vendor/vmaf/libvmaf/build/tools/vmaf', ROOT/'vendor/vmaf/model/vmaf_v0.6.1.json', f'q{q}')
                for s in manifest['selection'] for q in QPS]
        audit = convolution_audit(model)
        record = {'id': identity, 'checkpoint': str(checkpoint), 'definition': definition, 'seed': a.seed,
                  'steps': a.steps, 'parameters': parameter_count(model),
                  'gate': gate(baseline, rows),
                  'decoder_mac_reduction': 1-audit['decoder_convolution_macs']/baseline_audit['decoder_convolution_macs'],
                  'parameter_reduction': 1-parameter_count(model)/baseline_audit['parameters'],
                  'worst_vmaf_ratio': min(r['vmaf']/b['vmaf'] for r, b in zip(rows, baseline)),
                  'worst_rate_error': max(abs(r['actual_bits']/b['actual_bits']-1) for r, b in zip(rows, baseline)),
                  'bit_identical_to_baseline': all(r['bitstream_sha256'] == b['bitstream_sha256'] for r, b in zip(rows, baseline))}
        results.append(record)
        save_json(path, results)
        print(json.dumps({'finished': identity, 'gate': record['gate']['status'], 'worst_vmaf_ratio': record['worst_vmaf_ratio'],
                          'decoder_mac_reduction': record['decoder_mac_reduction']}), flush=True)
        del model
        torch.cuda.empty_cache()
    save_json(a.output/'completion.json', {'status': 'completed', 'variants': len(results), 'test_accessed': False})


if __name__ == '__main__':
    main()
