"""Exercise six rankings, paired compaction, save/reload and the VMAF gate.
Synthetic numerical checks are not training, RD evaluation, or hardware results.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
import torch
import yaml
from pruning_core import gated_ffns, channel_scores, select_channels, compact_ffn, prune_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261019)
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == '1b86b757ddb115342293efb57719d6216c6ee2e459ae796ec41723b5c05ca896'
    sys.path.insert(0, str(a.source / 'video'))
    from src.models.dmc_6.dmc_61sb import DMC
    cfg = yaml.safe_load((a.source / 'video/conversion/_full_model/model_configs_example.yaml').read_text())['dmc61sbr_mini_reglu']['params']
    cfg.pop('disable_feature_reset')
    model = DMC(**cfg).eval()
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True), strict=True)
    records = []
    with torch.no_grad():
        for name, block in gated_ffns(model, 'all'):
            width = block.ffn[2].in_channels
            x = torch.randn(1, block.ffn[0].in_channels, 4, 6) * .1
            evidence = {'activation_rms': torch.linspace(.1, 1., width),
                        'taylor': torch.linspace(1., .1, width)}
            for method in ['l1', 'l2', 'fpgm', 'random', 'activation_rms', 'taylor']:
                scores = channel_scores(block, method, evidence, seed=7)
                keep = select_channels(scores, .25)
                compact = compact_ffn(block, keep)
                dense_ffn = copy.deepcopy(block.ffn)
                mask = torch.zeros(width); mask[keep.cpu()] = 1
                hook = dense_ffn[1].register_forward_hook(lambda m, inp, z: z*mask[None,:,None,None])
                expected, actual = dense_ffn(x), compact.ffn(x)
                hook.remove()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                records.append({'block': name, 'ranking': method,
                                'old_width': width, 'new_width': len(keep),
                                'max_abs_error': (actual-expected).abs().max().item()})
    # Verify that a recorded compact layout can instantiate the saved tensor shapes.
    compact, layout = prune_model(model, 'l2', .25, 'decoder')
    reloaded = copy.deepcopy(model)
    for name, entry in layout.items():
        parent_name, _, child_name = name.rpartition('.')
        parent = reloaded.get_submodule(parent_name) if parent_name else reloaded
        parent._modules[child_name] = compact_ffn(reloaded.get_submodule(name), entry['keep'])
    reloaded.load_state_dict(compact.state_dict(), strict=True)
    try:
        channel_scores(next(gated_ffns(model, 'all'))[1], 'taylor')
        raise AssertionError('Missing calibration was silently accepted')
    except ValueError:
        pass
    result = {'status': 'passed', 'checks': records, 'ranking_input_note':
              'Weight rankings use official weights. Activation/Taylor inputs here are synthetic fixtures, not calibrated research scores.',
              'baseline_parameter_count': sum(p.numel() for p in model.parameters()),
              'decoder_ffn25_parameter_count': sum(p.numel() for p in compact.parameters()),
              'compact_layout_reload': 'passed', 'missing_calibration_rejected': True,
              'scope': 'CPU FP32 numerical and shape checks only; no video quality or GPU training'}
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}, indent=2))
    print('numerical_cases', len(records), 'max_error', max(r['max_abs_error'] for r in records))


if __name__ == '__main__':
    main()
