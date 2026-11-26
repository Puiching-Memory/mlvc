"""Check CPU rANS probability tables against the actual CUDA encoder."""
import json
from experiment_runtime import ROOT, load_model, save_json, configure_torch
from decoder_drift import entropy_signature
from entropy_tables import install, installed_hash


def main():
    configure_torch()
    expected = json.loads((ROOT/'results/decoder-drift/compatibility-proof.json').read_text())['baseline']
    model = load_model('cpu')
    actual = entropy_signature(model)
    if actual['geometry'] != expected['geometry'] or actual['original_encoder_quantizers_entropy_sha256'] != expected['original_encoder_quantizers_entropy_sha256']:
        raise ValueError('CPU latent geometry or weights differ from original encoder')
    native_identical = actual['pmf_tables_sha256'] == expected['pmf_tables_sha256']
    install(model, ROOT/'entropy/encoder-tables.npz')
    if installed_hash(model) != expected['pmf_tables_sha256']:
        raise ValueError('Frozen CPU entropy tables differ from actual encoder')
    result = {'status': 'passed', 'cpu_entropy_tables_and_geometry_match_cuda_encoder': True,
              'native_cpu_tables_identical_to_cuda': native_identical,
              'native_cpu_tables_sha256': actual['pmf_tables_sha256'],
              'frozen_actual_encoder_tables_installed': True,
              'pmf_tables_sha256': installed_hash(model),
              'full69_pair_cpu_vmaf_confirmation_performed': False}
    save_json(ROOT/'results/cpu-entropy-compatibility.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
