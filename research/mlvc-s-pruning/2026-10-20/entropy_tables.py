"""Freeze the actual encoder's integer rANS tables for cross-device replay.

Neural PMF construction can differ between CPU and CUDA. Installing the exact
encoder tables changes neither model weights nor any previously coded bytes.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from experiment_runtime import ROOT, CHECKPOINT_HASH, PROTOCOL, digest, load_model, save_json, configure_torch
from decoder_drift import entropy_signature, tensor_hash

NAMES = ('bit_estimator_z', 'gaussian_encoder')


def state_hash(model):
    return tensor_hash((name, value) for name, value in sorted(model.state_dict().items())
                       if name.startswith('bit_estimator_z.'))


def installed_hash(model):
    return tensor_hash((f'{name}.{i}', value) for name in NAMES
                      for i, value in enumerate(getattr(model, name).get_pmf()))


def geometry_signature(model):
    geometry = {key: getattr(model, key, None) for key in
                ('y_channels', 'z_channels', 'y_scale_repeat', 'y_downscale_factor',
                 'z_downscale_factor', 'feature_channels', 'force_zero_thres')}
    geometry.update({key: getattr(model.gaussian_encoder, key) for key in
                    ('distribution', 'scale_min', 'scale_max', 'scale_level',
                     'scale_step_in_log_space', 'scale_input_in_index_space')})
    geometry['qp_shift'] = list(model.qp_shift)
    geometry['frame_index_map'] = list(model.frame_index_map)
    geometry['scale_function_source'] = hashlib.sha256(inspect.getsource(type(model).get_y_scales).encode()).hexdigest()
    return geometry


def export(path):
    path = Path(path)
    model = load_model()
    signature = entropy_signature(model)
    expected = json.loads((ROOT/'results/decoder-drift/compatibility-proof.json').read_text())['baseline']
    if signature != expected:
        raise ValueError('Current actual CUDA encoder does not reproduce experiment entropy tables')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        install(model, path)
        if installed_hash(model) != expected['pmf_tables_sha256']:
            raise ValueError('Existing table export differs')
        return
    arrays = {f'{name}_{i}': np.asarray(value).copy() for name in NAMES
              for i, value in enumerate(getattr(model, name).get_pmf())}
    np.savez_compressed(path, **arrays)
    save_json(path.with_suffix('.json'), {'protocol': PROTOCOL, 'baseline_sha256': CHECKPOINT_HASH,
               'file_sha256': digest(path), 'pmf_tables_sha256': signature['pmf_tables_sha256'],
               'entropy_weight_sha256': state_hash(model), 'source_backend': 'actual H800 CUDA encoder',
               'geometry': signature['geometry'], 'reason': 'CPU and CUDA neural PMF generation differ; use exact original integer tables',
               'model_weights_changed': False, 'coded_bytes_changed': False})
    install(model, path)
    print(json.dumps({'exported': str(path), 'file_sha256': digest(path),
                      'pmf_tables_sha256': installed_hash(model)}))


def install(model, path):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if (metadata['protocol'] != PROTOCOL or metadata['baseline_sha256'] != CHECKPOINT_HASH or
        metadata['file_sha256'] != digest(path) or metadata['entropy_weight_sha256'] != state_hash(model) or
        metadata['geometry'] != geometry_signature(model)):
        raise ValueError('Entropy table identity or model entropy weights mismatch')
    values = []
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {f'{name}_{i}' for name in NAMES for i in range(3)}:
            raise ValueError('Unexpected entropy table arrays')
        for name in NAMES:
            table = [arrays[f'{name}_{i}'].copy() for i in range(3)]
            if any(not np.issubdtype(a.dtype, np.integer) for a in table):
                raise ValueError('Integer entropy tables required')
            values.extend((f'{name}.{i}', a) for i, a in enumerate(table))
            getattr(model, name).set_pmf(*table)
    if tensor_hash(values) != metadata['pmf_tables_sha256']:
        raise ValueError('Installed entropy tables differ from actual encoder')
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'entropy/encoder-tables.npz')
    a = p.parse_args()
    configure_torch()
    export(a.output)


if __name__ == '__main__':
    main()
