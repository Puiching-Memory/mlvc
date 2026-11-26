"""Regression checks anchored to pinned deployment source, not a hand-written offset.

Optional codec check crosses the first proactive LTR recovery with real rANS.
"""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from experiment_runtime import (ROOT, SOURCE, PROTOCOL, DeploymentReferences, frame_index,
                                unpack_frame, load_model, encode, configure_torch, save_json, digest)
from pruning_core import prune_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--codec', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, default=ROOT/'results/protocol-verification')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    configure_torch(4)
    source = SOURCE/'video/conversion/_split_model/_base_split_model.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseSplitModel')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_fa_idx')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    model = load_model(a.device)
    wrapper = SimpleNamespace(_full_model=SimpleNamespace(model_params=SimpleNamespace(frame_index_map=model.frame_index_map)))
    for index in range(2050):
        if frame_index(model, index) != namespace['_get_fa_idx'](wrapper, index):
            raise ValueError('Frame-type mapping differs from actual pinned deployment method')
    expected = {0: None, 1: 0, 8: 7, 9: 8, 63: 62, 64: 8, 65: 64, 127: 126,
                128: 64, 192: 128, 1023: 1022, 1024: None, 1032: 1031, 1088: 1032}
    refs = DeploymentReferences(model)
    x = torch.zeros(1, 3, 16, 16, device=a.device)
    events = []
    for index in range(1090):
        refs.reference(index, x)
        if index in expected:
            if refs.last_reference_index != expected[index]:
                raise ValueError('IDR/LTR reference schedule mismatch')
            events.append({'frame': index, 'reference': refs.last_reference_index,
                           'fa_idx': frame_index(model, index), 'mark_ltr': refs.mark(index)})
        refs.commit(index, {'test_index': index})
    raw = bytes([0]*16+[0, 64, 128, 255]+[255, 128, 64, 0])
    expanded = unpack_frame(raw, 4, 4)[0, 1:, :4, :4].numpy()
    expected_uv = np.repeat(np.repeat(np.array([[0,64],[128,255]])[None], 2, 1), 2, 2)/255
    np.testing.assert_allclose(expanded[:1], expected_uv, rtol=0, atol=1e-7)
    result = {'status': 'passed', 'protocol': PROTOCOL, 'source_fa_method_sha256': digest(source),
              'frame_loop_sha256': digest(SOURCE/'video/conversion/_frame_loop.py'),
              'fa_indices_checked_against_source': 2050, 'reference_schedule_checks': events,
              'nearest_chroma_matches_deployment_repeat': True}
    if a.codec:
        rng = np.random.default_rng(20261019)
        base = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        files = []
        for index in range(70):
            path = a.output/f'frame{index:03}.png'
            Image.fromarray(np.roll(base, index, axis=1)).save(path)
            files.append(str(path))
        sequence = {'id': 'synthetic_ltr_regression', 'files': files, 'width': 64, 'height': 64, 'fps': 30.}
        compact, _ = prune_model(model, 'l2', .25, 'reconstruction')
        original = encode(model, sequence, 32, a.output/'baseline.mlvcp')
        pruned = encode(compact, sequence, 32, a.output/'compact.mlvcp')
        if original['bitstream_sha256'] != pruned['bitstream_sha256']:
            raise ValueError('Head-only candidate changed actual bytes across LTR transition')
        result['codec_synthetic_only'] = {'frames': 70, 'head_only_actual_stream_identical': True,
               'max_decode_error': max(original['independent_decode_max_abs_error'], pruned['independent_decode_max_abs_error'])}
    save_json(a.output/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
