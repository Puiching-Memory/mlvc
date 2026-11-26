"""Additional physical structure experiments and dense sparse controls.

No sparse speedup is inferred from zero-valued dense weights. Extended layouts
are replayable with load_variant(), and retain the codec's external interfaces.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from experiment_runtime import (ROOT, CHECKPOINT_HASH, PROTOCOL, load_model, save_json,
                                configure_torch, convolution_audit, parameter_count, digest)
from pruning_core import gated_ffns, select_channels, compact_ffn


class Bypass(nn.Module):
    def forward(self, x, q_index=None):
        return x


def slice_output(conv, keep):
    keep = torch.as_tensor(keep, device=conv.weight.device, dtype=torch.long)
    conv.weight = nn.Parameter(conv.weight.detach()[keep].clone(), requires_grad=conv.weight.requires_grad)
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.detach()[keep].clone(), requires_grad=conv.bias.requires_grad)
    conv.out_channels = len(keep)


def slice_input(conv, keep):
    if conv.groups != 1:
        raise ValueError('Input projection must be ungrouped')
    keep = torch.as_tensor(keep, device=conv.weight.device, dtype=torch.long)
    conv.weight = nn.Parameter(conv.weight.detach()[:, keep].clone(), requires_grad=conv.weight.requires_grad)
    conv.in_channels = len(keep)


def compact_dc(block, keep):
    result = copy.deepcopy(block)
    first, depth, last = result.dc[0], result.dc[2], result.dc[3]
    if first.groups != 1 or depth.groups != depth.in_channels or depth.in_channels != depth.out_channels:
        raise ValueError('Unexpected depthwise structure')
    slice_output(first, keep)
    slice_output(depth, keep)
    depth.in_channels = depth.groups = len(keep)
    slice_input(last, keep)
    return result


def outer_compact(model, keep):
    model = copy.deepcopy(model)
    old_width = model.q_recon.shape[1]
    if not keep or min(keep) < 0 or max(keep) >= old_width or len(set(keep)) != len(keep):
        raise ValueError('Invalid outer channel indices')
    for index, block in enumerate(model.recon_generation_net.conv):
        if index == 0:
            if block.adaptor is None:
                raise ValueError('Expected first reconstruction adaptor')
            slice_output(block.adaptor, keep)
        elif block.adaptor is not None:
            raise ValueError('Unexpected reconstruction width conversion')
        slice_input(block.dc[0], keep)
        slice_output(block.dc[3], keep)
        slice_input(block.ffn[0], keep)
        slice_output(block.ffn[2], keep)
    slice_input(model.recon_generation_net.head, keep)
    model.q_recon = nn.Parameter(model.q_recon.detach()[:, keep].clone(), requires_grad=model.q_recon.requires_grad)
    return model


def apply_definition(model, definition):
    for name, keep in definition.get('ffn', {}).items():
        parent, _, child = name.rpartition('.')
        model.get_submodule(parent)._modules[child] = compact_ffn(model.get_submodule(name), keep)
    for name, keep in definition.get('dc', {}).items():
        parent, _, child = name.rpartition('.')
        model.get_submodule(parent)._modules[child] = compact_dc(model.get_submodule(name), keep)
    for name in definition.get('bypass', []):
        block = model.get_submodule(name)
        if block.adaptor is not None or block.shortcut:
            raise ValueError('Only a same-width residual block can be removed')
        parent, _, child = name.rpartition('.')
        model.get_submodule(parent)._modules[child] = Bypass()
    if 'outer_keep' in definition:
        model = outer_compact(model, definition['outer_keep'])
    return model


def load_variant(path, device='cuda'):
    record = torch.load(path, map_location='cpu', weights_only=True)
    if record.get('protocol') != PROTOCOL:
        raise ValueError('Structure checkpoint belongs to another protocol')
    if record['baseline_sha256'] != CHECKPOINT_HASH:
        raise ValueError('Baseline mismatch')
    model = apply_definition(load_model('cpu'), record['definition'])
    model.load_state_dict(record['state_dict'], strict=True)
    model._measurement_fingerprint = digest(path)
    return model.to(device).eval()


def save_variant(path, model, definition, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'definition': definition, 'metadata': metadata, 'baseline_sha256': CHECKPOINT_HASH, 'protocol': PROTOCOL}, path)
    model._measurement_fingerprint = digest(path)


def nonuniform_layout(base, calibration, scope, target_fraction, audit):
    """Greedy groups of 8 by measured Taylor saliency per executed Conv MAC."""
    layer_info = {row['name']: row for row in audit['layers']}
    candidates = []
    keep_by_name = {}
    for name, block in gated_ffns(base, scope):
        scores = torch.tensor(calibration[name]['taylor'])
        width = len(scores)
        keep_by_name[name] = set(range(width))
        shape = layer_info[name+'.ffn.0']['output_shape']
        per_channel_macs = shape[-2]*shape[-1]*(2*block.ffn[0].in_channels+block.ffn[2].out_channels)
        order = torch.argsort(scores, stable=True).tolist()
        for index in range(0, width-8, 8):
            channels = order[index:index+8]
            loss = sum(float(scores[channel]) for channel in channels)
            cost = len(channels)*per_channel_macs
            candidates.append((loss/cost, name, channels, cost))
    target = target_fraction*audit['decoder_convolution_macs']
    saved = 0
    for _, name, channels, cost in sorted(candidates, key=lambda x: x[0]):
        if saved >= target:
            break
        if len(keep_by_name[name])-len(channels) < 8:
            continue
        keep_by_name[name].difference_update(channels)
        saved += cost
    if saved < .99*target:
        raise ValueError('Requested nonuniform MAC budget cannot be reached')
    return {name: sorted(keep) for name, keep in keep_by_name.items()}


def sparse_control(base, fraction=None, nm=False):
    model = copy.deepcopy(base)
    names = {name for name, _ in gated_ffns(model, 'active_decoder')}
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Conv2d) and any(name.startswith(parent+'.') for parent in names)]
    before_nonzero = sum(int(torch.count_nonzero(module.weight)) for _, module in modules)
    eligible, removed = 0, 0
    with torch.no_grad():
        if nm:
            for _, module in modules:
                matrix = module.weight.reshape(module.out_channels, -1)
                if matrix.shape[1] % 4:
                    continue
                grouped = matrix.reshape(module.out_channels, -1, 4)
                indices = grouped.abs().topk(2, dim=-1).indices
                mask = torch.zeros_like(grouped).scatter_(-1, indices, 1)
                grouped.mul_(mask)
                eligible += grouped.numel()
                removed += grouped.numel()//2
        else:
            weights = torch.cat([module.weight.detach().abs().flatten() for _, module in modules])
            amount = int(len(weights)*fraction)
            indices = weights.argsort(stable=True)[:amount]
            mask = torch.ones(len(weights), device=weights.device)
            mask[indices] = 0
            offset = 0
            for _, module in modules:
                count = module.weight.numel()
                module.weight.mul_(mask[offset:offset+count].reshape_as(module.weight))
                offset += count
            eligible, removed = len(weights), amount
    return model, {'kind': '2:4_dense_mask_control' if nm else 'global_magnitude_dense_mask_control',
                   'eligible_weights': eligible, 'removed_weights': removed,
                   'nonzero_before': before_nonzero,
                   'parameter_shapes_reduced': False, 'sparse_kernel_used': False,
                   'speedup_claim': 'none; zeros are evaluated with ordinary dense Conv2d'}


def definitions(base, calibration, audit):
    for fraction in (.125, .25, .5):
        dc = {}
        for name, block in gated_ffns(base, 'active_decoder'):
            first, depth, last = block.dc[0], block.dc[2], block.dc[3]
            scores = first.weight.detach().flatten(1).norm(dim=1)*depth.weight.detach().flatten(1).norm(dim=1)*last.weight.detach().transpose(0, 1).flatten(1).norm(dim=1)
            dc[name] = select_channels(scores, fraction).cpu().tolist()
        yield f'dc_internal_p{int(fraction*1000)}', {'dc': dc}
    scores = base.recon_generation_net.head.weight.detach().transpose(0, 1).flatten(1).norm(dim=1)
    for fraction in (.125, .25, .375, .5):
        keep = select_channels(scores, fraction).cpu().tolist()
        yield f'reconstruction_outer_p{int(fraction*1000)}', {'outer_keep': keep}
    available = [f'recon_generation_net.conv.{index}' for index, block in enumerate(base.recon_generation_net.conv)
                 if block.adaptor is None and not block.shortcut]
    for depth in range(1, len(available)+1):
        yield f'reconstruction_remove_{depth}_blocks', {'bypass': available[-depth:]}
    for scope in ('reconstruction', 'main_decoder', 'active_decoder'):
        for fraction in (.05, .1, .15, .2):
            layout = nonuniform_layout(base, calibration, scope, fraction, audit)
            yield f'nonuniform_{scope}_mac{int(fraction*100)}', {'ffn': layout}


def numerical_checks(output):
    configure_torch(4)
    base = load_model('cpu')
    checks = []
    with torch.no_grad():
        for name, block in gated_ffns(base, 'active_decoder'):
            width = block.dc[0].out_channels
            keep = list(range(0, width, 2))
            compact = compact_dc(block, keep)
            dense = copy.deepcopy(block)
            mask = torch.zeros(width); mask[keep] = 1
            handle = dense.dc[2].register_forward_hook(lambda module, inp, value, mask=mask: value*mask[None, :, None, None])
            x = torch.randn(1, block.dc[0].in_channels, 4, 6)*.1
            if block.adaptor is not None:
                x = torch.randn(1, block.adaptor.in_channels, 4, 6)*.1
            expected, actual = dense(x), compact(x)
            handle.remove()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            checks.append({'kind': 'depthwise_hidden_mask_equivalence', 'block': name,
                           'max_error': float((actual-expected).abs().max())})
        width = base.q_recon.shape[1]
        keep = list(range(0, width, 2))
        compact = outer_compact(base, keep)
        dense = copy.deepcopy(base)
        mask = torch.zeros(width); mask[keep] = 1
        handles = []
        for block in dense.recon_generation_net.conv:
            for layer in ([block.adaptor] if block.adaptor is not None else [])+[block.dc[3], block.ffn[2]]:
                handles.append(layer.register_forward_hook(lambda module, inp, value: value*mask[None, :, None, None]))
        x = torch.randn(1, base.feature_channels, 4, 6)*.1
        expected = dense.recon_generation_net(x, dense.q_recon[32])
        actual = compact.recon_generation_net(x, compact.q_recon[32])
        for h in handles:
            h.remove()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        checks.append({'kind': 'outer_width_mask_equivalence', 'max_error': float((actual-expected).abs().max())})
        path = output.with_suffix('.roundtrip.pt')
        save_variant(path, compact, {'outer_keep': keep}, {'test_only': True})
        loaded = load_variant(path, 'cpu')
        torch.testing.assert_close(loaded.recon_generation_net(x, loaded.q_recon[32]), actual, rtol=0, atol=0)
        checks.append({'kind': 'extended_checkpoint_strict_roundtrip', 'passed': True})
    save_json(output, {'status': 'passed', 'synthetic_only': True, 'checks': checks})
    print(json.dumps({'status': 'passed', 'checks': len(checks)}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'results/structure')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.verify:
        return numerical_checks(a.output/'numerical-verification.json')
    from run_suite import cached_measure, gate, QPS
    configure_torch()
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    calibration = json.loads((ROOT/'results/calibration.json').read_text())['scores']
    save_json(a.output/'environment.json', {'protocol': PROTOCOL, 'torch': torch.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(), 'manifest_sha256': digest(ROOT/'data/manifest.json'),
               'calibration_sha256': digest(ROOT/'results/calibration.json'),
               'source_sha256': digest(Path(__file__)), 'selection_only': True, 'test_accessed': False})
    base = load_model()
    baseline_codec = load_model()
    audit = convolution_audit(base)
    sequences = manifest['selection'][:2]
    baseline = [cached_measure(baseline_codec, s, q, ROOT/'results/screen/baseline',
                                ROOT/'vendor/vmaf/libvmaf/build/tools/vmaf', ROOT/'vendor/vmaf/model/vmaf_v0.6.1.json', f'q{q}')
                for s in sequences for q in QPS]
    path = a.output/'index.json'
    results = json.loads(path.read_text()) if path.exists() else []
    done = {r['id'] for r in results}
    variants = list(definitions(base, calibration, audit))
    variants += [(f'unstructured_p{int(f*100)}', {'sparse_fraction': f}) for f in (.25, .5, .75)]
    variants += [('nm_2_4_dense_control', {'nm_dense_control': True})]
    for identity, definition in variants:
        if identity in done:
            continue
        sparse = None
        if 'sparse_fraction' in definition or 'nm_dense_control' in definition:
            model, sparse = sparse_control(base, definition.get('sparse_fraction'), definition.get('nm_dense_control', False))
        else:
            model = apply_definition(copy.deepcopy(base), definition)
        checkpoint = a.output/(identity+'.pt')
        save_variant(checkpoint, model, definition, {'id': identity, 'recovery': 'none', 'sparse': sparse})
        # Strict replay before any measured encode, including physically changed shapes.
        del model
        model = load_variant(checkpoint)
        rows = [cached_measure(model, s, q, a.output/identity,
                                ROOT/'vendor/vmaf/libvmaf/build/tools/vmaf', ROOT/'vendor/vmaf/model/vmaf_v0.6.1.json', f'q{q}')
                for s in sequences for q in QPS]
        compact_audit = convolution_audit(model)
        record = {'id': identity, 'definition': definition, 'checkpoint': str(checkpoint),
                  'gate': gate(baseline, rows), 'sparse': sparse,
                  'parameter_reduction': 1-parameter_count(model)/audit['parameters'],
                  'decoder_mac_reduction': 1-compact_audit['decoder_convolution_macs']/audit['decoder_convolution_macs'],
                  'worst_vmaf_ratio': min(r['vmaf']/b['vmaf'] for r, b in zip(rows, baseline)),
                  'worst_rate_error': max(abs(r['actual_bits']/b['actual_bits']-1) for r, b in zip(rows, baseline))}
        results.append(record)
        save_json(path, results)
        print(json.dumps({k: record[k] for k in ('id', 'parameter_reduction', 'decoder_mac_reduction', 'worst_vmaf_ratio', 'worst_rate_error')}), flush=True)
        del model
        torch.cuda.empty_cache()
    save_json(a.output/'completion.json', {'status': 'completed', 'variants': len(results), 'screen_only': True})


if __name__ == '__main__':
    main()
