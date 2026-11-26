"""Fixed original encoder, compact decoder, recursive latent distillation.

This is an additional preregistered research route, independent of head-only
confirmation. No pruned encoder claim is made. Actual baseline bytes are used
for both long-video selection and held-out confirmation.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from experiment_runtime import (ROOT, PROTOCOL, CHECKPOINT_HASH, load_model, save_model,
    digest, save_json, configure_torch, canonical_frame, frame_index,
    DeploymentReferences, convolution_audit, parameter_count, vmaf_measure)
from final_validation import read_stream, decode, FINAL
from pruning_core import prune_model
from run_suite import QPS, cached_measure, weighted_mse, set_training_parameters
from quality_gate import evaluate

OUT = ROOT/'results/decoder-drift'
MANIFEST = ROOT/'data/got-long/manifest.json'
PLAN = ROOT/'decoder-drift-plan.json'


def read(path):
    return json.loads(Path(path).read_text())


def tensor_hash(values):
    h = hashlib.sha256()
    for name, value in values:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous().numpy()
        value = np.asarray(value)
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def entropy_signature(model):
    """Exact PMF tables, source functions, scale/latent geometry and encoder."""
    import inspect
    tables = []
    for name in ('bit_estimator_z', 'gaussian_encoder'):
        module = getattr(model, name)
        module.build_pmf()
        tables.extend((f'{name}.{i}', v) for i, v in enumerate(module.get_pmf()))
    keys = ('y_channels', 'z_channels', 'y_scale_repeat', 'y_downscale_factor',
            'z_downscale_factor', 'feature_channels', 'force_zero_thres')
    geometry = {key: getattr(model, key, None) for key in keys}
    g = model.gaussian_encoder
    geometry.update({key: getattr(g, key) for key in
        ('distribution', 'scale_min', 'scale_max', 'scale_level',
         'scale_step_in_log_space', 'scale_input_in_index_space')})
    geometry['qp_shift'] = list(model.qp_shift)
    geometry['frame_index_map'] = list(model.frame_index_map)
    geometry['scale_function_source'] = hashlib.sha256(inspect.getsource(type(model).get_y_scales).encode()).hexdigest()
    frozen = [(name, value) for name, value in sorted(model.state_dict().items())
              if name.startswith(('encoder.', 'hyper_encoder.', 'q_', 'bit_estimator_z.'))]
    return {'pmf_tables_sha256': tensor_hash(tables), 'geometry': geometry,
            'original_encoder_quantizers_entropy_sha256': tensor_hash(frozen)}


def compatible(model, expected):
    actual = entropy_signature(model)
    if actual != expected:
        raise ValueError('Changed original encoder, quantizers, entropy tables or latent geometry')
    return actual


def variants():
    scores = read(ROOT/'results/calibration.json')['scores']
    for spec in read(PLAN)['variants']:
        name = f"drift_{spec['scope']}_taylor_p{int(spec['fraction']*1000)}"
        yield name, spec, scores


def make_variant(spec, scores):
    return prune_model(load_model(), 'taylor', spec['fraction'], spec['scope'], scores, seed=20261019)


def detach_references(references):
    for key in ('previous', 'latest_ltr'):
        dpb = getattr(references, key)
        if dpb is not None:
            setattr(references, key, {k: v.detach() for k, v in dpb.items()})


def bank():
    output = OUT/'bank'
    output.mkdir(parents=True, exist_ok=True)
    index = output/'index.json'
    if index.exists():
        record = read(index)
        if record['manifest_sha256'] != digest(MANIFEST) or record['protocol'] != PROTOCOL:
            raise ValueError('Incompatible latent bank')
        for item in record['items']:
            if digest(item['path']) != item['sha256']:
                raise ValueError('Latent bank changed')
        return record
    model = load_model().requires_grad_(False).eval()
    signature = entropy_signature(model)
    rng = random.Random(20261019)
    items = []
    with torch.no_grad():
        for sequence in read(MANIFEST)['train']:
            for qp in QPS:
                top, left = rng.randrange(129), rng.randrange(129)
                references = DeploymentReferences(model)
                frames = []
                for i, file in enumerate(sequence['files']):
                    _, x = canonical_frame(file, (256, 256))
                    x = x[..., top:top+128, left:left+128].cuda()
                    dpb = references.reference(i, x)
                    result = model.compress_core(x, dpb, qp, frame_index(model, i))
                    references.commit(i, result['dpb'])
                    row = {'x': x.cpu(), 'teacher_x': result['dpb']['ref_frame'].cpu(),
                           'teacher_feature': result['dpb']['ref_feature'].cpu(),
                           'z': result['z_hat'].cpu(), 'y0': result['y_q'][0][0].cpu(),
                           'y1': result['y_q'][1][0].cpu()}
                    frames.append(row)
                path = output/f"{sequence['id']}_q{qp}.pt"
                torch.save({'frames': frames, 'qp': qp, 'sequence': sequence['id'],
                            'crop': [top, left, 128, 128], 'protocol': PROTOCOL}, path)
                items.append({'path': str(path), 'sha256': digest(path), 'qp': qp, 'frames': len(frames)})
                print(json.dumps({'latent_bank': sequence['id'], 'qp': qp, 'frames': len(frames)}), flush=True)
    record = {'protocol': PROTOCOL, 'manifest_sha256': digest(MANIFEST),
              'encoder_signature': signature, 'items': items, 'training_only': True}
    save_json(index, record)
    return record


def proof():
    """Real rANS decode must reproduce latent-based decode for own recurrence."""
    teacher = load_model().requires_grad_(False).eval()
    signature = entropy_signature(teacher)
    generator = torch.Generator(device='cuda').manual_seed(20261019)
    recordings = {}
    with torch.no_grad():
        for qp in QPS:
            ref = DeploymentReferences(teacher)
            rows = []
            base = torch.rand(1, 3, 128, 128, generator=generator, device='cuda')
            for i in range(70):
                x = torch.roll(base, i%9, -1)
                dpb = ref.reference(i, x)
                core = teacher.compress_core(x, dpb, qp, frame_index(teacher, i))
                actual = teacher.compress(x, dpb, q_index=qp, fa_idx=frame_index(teacher, i))
                if not torch.equal(core['dpb']['ref_feature'], actual['dpb']['ref_feature']):
                    raise ValueError('Core and real encoder disagree')
                ref.commit(i, core['dpb'])
                rows.append((actual['bit_stream'], core['z_hat'], core['y_q']))
            recordings[qp] = rows
        results = []
        for name, spec, scores in variants():
            model, _ = make_variant(spec, scores)
            compatible(model, signature)
            max_error = 0.
            for qp, rows in recordings.items():
                actual_refs, latent_refs = DeploymentReferences(model), DeploymentReferences(model)
                for i, (payload, z, y) in enumerate(rows):
                    actual_dpb = actual_refs.reference(i, base)
                    actual = model.decompress(payload, actual_dpb, q_index=qp,
                               fa_idx=frame_index(model, i), height=128, width=128)['dpb']
                    latent_dpb = latent_refs.reference(i, base)
                    latent = model.decode_core(latent_dpb, qp, frame_index(model, i),
                                               z_hat=z, y_res_q=y)[2]
                    for key in actual:
                        err = float((actual[key]-latent[key]).abs().max())
                        max_error = max(max_error, err)
                        if err > 1e-5 or not bool(actual[key].isfinite().all()):
                            raise ValueError('Real rANS and fixed-latent decoder differ')
                    actual_refs.commit(i, actual); latent_refs.commit(i, latent)
            results.append({'id': name, 'real_decoded_frames': 210, 'max_error': max_error,
                            'entropy_geometry_encoder_unchanged': True})
            print(json.dumps({'proof': name, 'max_error': max_error}), flush=True)
            del model
    save_json(OUT/'compatibility-proof.json', {'protocol': PROTOCOL, 'plan_sha256': digest(PLAN),
               'baseline': signature, 'variants': results, 'ltr64_crossed': True,
               'same_original_encoder_streams': True})


def train():
    proof_doc = read(OUT/'compatibility-proof.json')
    if proof_doc['plan_sha256'] != digest(PLAN):
        raise ValueError('Plan changed after proof')
    cache = bank()
    index = OUT/'training-index.json'
    records = read(index) if index.exists() else []
    done = {r['id'] for r in records}
    for name, spec, scores in variants():
        if name in done:
            continue
        model, layout = make_variant(spec, scores)
        parameters = set_training_parameters(model, spec['scope'])
        # All quantizer rows remain frozen, unlike naive RD fine tuning.
        for pname, p in model.named_parameters():
            if pname.startswith('q_'):
                p.requires_grad_(False)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=3e-5)
        rng = random.Random(20261019)
        logs = []
        # eval mode deliberately keeps deployment clipping. Autograd remains on.
        model.eval()
        for iteration in range(100):
            item = rng.choice(cache['items'])
            record = torch.load(item['path'], map_location='cuda', weights_only=True)
            student_refs, teacher_refs = DeploymentReferences(model), DeploymentReferences(model)
            loss_sum, total = 0., 0
            for start in range(0, len(record['frames']), 16):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for i in range(start, min(start+16, len(record['frames']))):
                    row = record['frames'][i]
                    teacher_dpb = teacher_refs.reference(i, row['x'])
                    student_dpb = student_refs.reference(i, row['x'])
                    reference = teacher_dpb if iteration < 25 else student_dpb
                    xhat, feature, dpb, _, _ = model.decode_core(reference, record['qp'], frame_index(model, i),
                            z_hat=row['z'], y_res_q=((row['y0'], None), (row['y1'], None)))
                    loss = .9*weighted_mse(xhat, row['teacher_x']) + .1*weighted_mse(xhat, row['x'])
                    loss = loss + .05*(feature-row['teacher_feature']).square().mean()
                    if not bool(loss.isfinite()):
                        raise ValueError('Non-finite drift training loss')
                    losses.append(loss)
                    student_refs.commit(i, dpb)
                    teacher_refs.commit(i, {'ref_frame': row['teacher_x'], 'ref_feature': row['teacher_feature']})
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, .5)
                optimizer.step()
                detach_references(student_refs)
                loss_sum += float(loss.detach())*len(losses); total += len(losses)
            if iteration == 0 or (iteration+1)%10 == 0:
                row = {'iteration': iteration+1, 'mean_loss': loss_sum/total,
                       'teacher_reference_warmup': iteration < 25, 'video_qp': item['path']}
                logs.append(row)
                save_json(OUT/(name+'.training.json'), {'logs': logs, 'plan_sha256': digest(PLAN),
                         'bank_index_sha256': digest(OUT/'bank/index.json')})
                torch.save({'iteration': iteration+1, 'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(), 'sampler_rng': rng.getstate(), 'layout': layout,
                       'protocol': PROTOCOL, 'plan_sha256': digest(PLAN)}, OUT/(name+'.resume.pt'))
                print(json.dumps({'drift_training': name, **row}), flush=True)
        compatible(model, cache['encoder_signature'])
        path = OUT/(name+'.pt')
        save_model(path, model, layout, {'plan_sha256': digest(PLAN), 'iterations': 100,
                   'method': 'fixed_original_encoder_recursive_latent_distillation', 'spec': spec})
        audit = convolution_audit(model)
        base = load_model()
        base_audit = convolution_audit(base)
        records.append({'id': name, 'checkpoint': str(path), 'checkpoint_sha256': digest(path),
                        'kind': 'ffn', 'scope': spec['scope'], 'fraction': spec['fraction'],
                        'parameters': parameter_count(model),
                        'parameter_reduction': 1-parameter_count(model)/parameter_count(base),
                        'decoder_mac_reduction': 1-audit['decoder_convolution_macs']/base_audit['decoder_convolution_macs']})
        save_json(index, records)
        del model, base, record
        torch.cuda.empty_cache()


def measure(model, sources, directory, signature):
    compatible(model, signature)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sources:
        key = source['sequence']+'_'+source['target']
        path = directory/(key+'.json')
        if path.exists():
            row = read(path)
            if (row['checkpoint_sha256'] != model._measurement_fingerprint or
                digest(source['bitstream_path']) != row['bitstream_sha256'] or
                row['protocol'] != PROTOCOL):
                raise ValueError('Stale fixed-stream measurement')
            rows.append(row)
            continue
        output = directory/(key+'.decoded.yuv')
        stream = directory/(key+'.mlvcp')
        if not stream.exists():
            try:
                os.link(source['bitstream_path'], stream)
            except OSError:
                shutil.copyfile(source['bitstream_path'], stream)
        if digest(stream) != source['bitstream_sha256'] or digest(source['reference_path']) != source['source_sha256']:
            raise ValueError('Actual stream/reference changed')
        header, payloads = read_stream(stream)
        started = time.monotonic()
        with output.open('wb') as f:
            decode(model, header, payloads, f)
        if output.stat().st_size != header['frames']*header['width']*header['height']*3//2:
            raise ValueError('Wrong frame count')
        row = {k: source[k] for k in ('sequence', 'target', 'source_sha256', 'frames', 'width', 'height',
                     'duration_seconds', 'color_profile', 'protocol', 'source_manifest_sha256', 'qp')}
        row.update({'actual_bits': stream.stat().st_size*8, 'rate_source': 'actual_bitstream',
                    'checkpoint_sha256': model._measurement_fingerprint,
                    'bitstream_path': str(stream), 'bitstream_sha256': digest(stream),
                    'decoded_path': str(output), 'reference_path': source['reference_path'],
                    'decode_seconds_with_output_io': time.monotonic()-started, 'same_bytes_as_baseline': True,
                    'decoded_sha256': digest(output), 'worker_pid': os.getpid(),
                    'original_rgb_inputs_read_in_worker': False, 'encoder_invoked_in_worker': False})
        row = vmaf_measure(row, ROOT/'vendor/vmaf/libvmaf/build/tools/vmaf', ROOT/'vendor/vmaf/model/vmaf_v0.6.1.json')
        rows.append(row)
        save_json(path, row)
        print(json.dumps({'fixed_stream_decoder': directory.name, 'sequence': row['sequence'],
                          'target': row['target'], 'vmaf_ratio': row['vmaf']/source['vmaf']}), flush=True)
    pairs = [[r['sequence'], r['target']] for r in sources]
    document = {'registered_pairs': pairs, 'baseline': sources, 'candidate': rows}
    checked = evaluate(document, rate_tolerance=0.)
    save_json(directory/'measurements.json', document)
    save_json(directory/'gate.json', checked)
    return {'gate_status': checked['status'], 'worst_vmaf_ratio': min(r['vmaf']/b['vmaf'] for r, b in zip(rows, sources)),
            'worst_actual_rate_relative_error': 0., 'all_bitstreams_identical': True, 'test_pairs': len(rows)}


def selection_baseline():
    model = load_model()
    sources = []
    for sequence in read(MANIFEST)['selection']:
        for qp in QPS:
            sources.append(cached_measure(model, sequence, qp, OUT/'selection/baseline',
                ROOT/'vendor/vmaf/libvmaf/build/tools/vmaf', ROOT/'vendor/vmaf/model/vmaf_v0.6.1.json', f'q{qp}'))
    save_json(OUT/'selection/baseline.json', sources)


def worker(stage, identity):
    records = read(OUT/'training-index.json') if stage == 'select-worker' else read(OUT/'frozen-finalists.json')['candidates']
    record = next(r for r in records if r['id'] == identity)
    if digest(record['checkpoint']) != record['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    model = load_model(compact_checkpoint=record['checkpoint'])
    signature = read(OUT/'compatibility-proof.json')['baseline']
    if stage == 'select-worker':
        sources = read(OUT/'selection/baseline.json')
        directory = OUT/'selection'/identity
    else:
        frozen = read(OUT/'frozen-finalists.json')
        if digest(ROOT/'data/test-full/manifest.json') != frozen['test_manifest_sha256']:
            raise ValueError('Held-out manifest changed')
        sources = read(FINAL/'baseline.json')
        directory = OUT/'heldout'/identity
    result = {**record, **measure(model, sources, directory, signature)}
    save_json(directory/'summary.json', result)


def select():
    selection_baseline()
    records = read(OUT/'training-index.json')
    if len(records) != 6:
        raise ValueError('All six preregistered variants must finish')
    summaries = []
    for record in records:
        subprocess.run([sys.executable, str(Path(__file__)), '--stage', 'select-worker', '--variant', record['id']], check=True)
        summaries.append(read(OUT/'selection'/record['id']/'summary.json'))
    save_json(OUT/'selection-index.json', summaries)
    chosen = []
    for scope in ('main_decoder', 'active_decoder'):
        passing = [r for r in summaries if r['scope'] == scope and r['gate_status'] == 'passed']
        if passing:
            chosen.append(max(passing, key=lambda r: r['decoder_mac_reduction']))
    frozen = {'protocol': PROTOCOL, 'plan_sha256': digest(PLAN), 'selection_manifest_sha256': digest(MANIFEST),
              'test_manifest_sha256': digest(ROOT/'data/test-full/manifest.json'), 'candidates': chosen,
              'registered_pairs': read(FINAL/'frozen-finalists.json')['registered_pairs'],
              'selection_policy': 'Best MAC reduction per scope among all12 long GOT selection pairs passing95% at exactly identical original encoder stream bytes',
              'no_mot_candidate_metrics_used_for_selection': True}
    path = OUT/'frozen-finalists.json'
    if path.exists() and read(path) != frozen:
        raise ValueError('Cannot modify frozen finalists')
    save_json(path, frozen)


def test():
    frozen = read(OUT/'frozen-finalists.json')
    reports = []
    for record in frozen['candidates']:
        subprocess.run([sys.executable, str(Path(__file__)), '--stage', 'test-worker', '--variant', record['id']], check=True)
        reports.append(read(OUT/'heldout'/record['id']/'summary.json'))
    save_json(OUT/'results.json', {'status': 'completed', 'frozen_finalists_sha256': digest(OUT/'frozen-finalists.json'),
               'results': reports, 'selected_candidates': len(reports), 'all_test_pairs_per_candidate': len(frozen['registered_pairs']),
               'no_selection_candidate_passed': not bool(reports), 'actual_bitrate_tolerance': 0.,
               'original_encoder_required': True, 'test_tuning_performed': False})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('proof', 'bank', 'train', 'select', 'test', 'select-worker', 'test-worker', 'all'), default='all')
    p.add_argument('--variant')
    a = p.parse_args()
    configure_torch()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage.endswith('-worker'):
        worker(a.stage, a.variant)
    elif a.stage == 'all':
        save_json(OUT/'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(), 'protocol': PROTOCOL, 'source_sha256': digest(Path(__file__)),
                'plan_sha256': digest(PLAN), 'manifest_sha256': digest(MANIFEST)})
        proof(); bank(); train(); select(); test()
    else:
        globals()[a.stage]()


if __name__ == '__main__':
    main()
