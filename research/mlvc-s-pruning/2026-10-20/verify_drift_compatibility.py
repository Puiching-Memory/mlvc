"""Focused compatibility guard checks; not measured video-quality evidence."""
import copy
import json
import torch
from decoder_drift import entropy_signature, compatible
from experiment_runtime import load_model, configure_torch, save_json, ROOT
from pruning_core import prune_model


def main():
    configure_torch()
    base = load_model('cpu')
    expected = json.loads(json.dumps(entropy_signature(base)))
    # Construct from a fresh model: entropy C++ handles are not deepcopyable.
    compact, _ = prune_model(load_model('cpu'), 'l1', .125, 'active_decoder')
    compatible(compact, expected)
    checked = ['compact_decoder_with_original_entropy_is_compatible', 'json_roundtrip']
    for kind in ('encoder', 'quantizer', 'entropy', 'geometry'):
        model = load_model('cpu')
        with torch.no_grad():
            if kind == 'geometry':
                model.y_scale_repeat += 1
            else:
                prefix = {'encoder': 'encoder.', 'quantizer': 'q_decoder', 'entropy': 'bit_estimator_z.'}[kind]
                value = next(p for n, p in model.named_parameters() if n.startswith(prefix))
                value.view(-1)[0] += .1
        try:
            compatible(model, expected)
        except ValueError:
            checked.append(kind+'_change_rejected')
        else:
            raise AssertionError('Mutation escaped guard: '+kind)
    path = ROOT/'drift-compatibility-verification.json'
    save_json(path, {'status': 'passed', 'checks': checked, 'backend': 'CPU guard tests only',
                     'real_gpu_rans_and_vmaf_tests_still_required': True})
    print(path.read_text())


if __name__ == '__main__':
    main()
