"""Export completed v2 research evidence, trained states and actual streams.

Exclude source image datasets, raw YUV intermediates, private keys and v1
diagnostics. All trained checkpoints/optimizer states are included; uniform
untrained screening checkpoints are reproducible from baseline+layout+scores.
"""
import csv
import hashlib
import html
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
FINAL = RESULTS/'heldout-confirmation'
BUNDLE = ROOT/'export-v2'
PROTOCOL = 'MLVCS_deployment_ltr64_idr1024_v2'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    summary = read(FINAL/'results.json')
    frozen = read(FINAL/'frozen-finalists.json')
    if summary['status'] != 'completed' or frozen['protocol'] != PROTOCOL:
        raise ValueError('Completed v2 confirmation is required')
    drift = read(RESULTS/'decoder-drift/results.json')
    if drift['status'] != 'completed':
        raise ValueError('Additional registered drift experiments must finish')
    summary['head_only_results_count'] = len(summary['results'])
    summary['results'] += drift['results']
    summary['original_encoder_required_for_drift_models'] = True
    if not (RESULTS/'decoder-benchmark-combined.json').exists():
        raise ValueError('Benchmark must finish before packaging')
    if BUNDLE.exists():
        raise FileExistsError('Export directory already exists; inspect it before replacing anything')
    BUNDLE.mkdir()
    mapping, files = {}, []
    def copy(source, relative):
        source, target = Path(source), BUNDLE/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(source)
        if target.exists():
            if sha(target) != expected:
                raise ValueError('Export filename collision')
        else:
            shutil.copy2(source, target)
            if sha(target) != expected:
                raise ValueError('Export copy digest mismatch')
        mapping[str(source.resolve())] = str(Path(relative).as_posix())
        return target
    for source in sorted(ROOT.iterdir()):
        if source.is_file() and (source.suffix in ('.py', '.sh') or source.name in
             ('requirements-runtime.txt', 'requirements-report.txt', 'MLVC-LICENSE', 'mlvc-source-64b10d4.zip', 'mlvc-s-psnr-v1.ckpt', 'source-integrity.json', 'protocol-correction.json', 'decoder-drift-plan.json', 'drift-compatibility-verification.json', 'plugin-feedback.json')):
            copy(source, source.name)
    for source in (ROOT/'data/manifest.json', ROOT/'data/test-full/manifest.json', ROOT/'data/got-long/manifest.json'):
        copy(source, source.relative_to(ROOT))
    for source in sorted((ROOT/'entropy').glob('encoder-tables.*')):
        copy(source, source.relative_to(ROOT))
    if not (BUNDLE/'entropy/encoder-tables.npz').exists() or not (BUNDLE/'entropy/encoder-tables.json').exists():
        raise ValueError('Exact actual encoder entropy tables must be included')
    for source in sorted(RESULTS.rglob('*.json')):
        copy(source, Path('evidence')/source.relative_to(ROOT))
    for source in sorted(RESULTS.rglob('*.mlvcp')):
        # Byte-identical candidate streams are stored once by digest.
        copy(source, Path('streams')/(sha(source)+'.mlvcp'))
    # Supply the exact VMAF model and immutable source/build provenance. The
    # user rebuilds the executable with build_vmaf.sh rather than trusting a
    # remotely linked dynamic binary on an arbitrary operating system.
    copy(ROOT/'vendor/vmaf/model/vmaf_v0.6.1.json', 'vmaf/vmaf_v0.6.1.json')
    copy(ROOT/'vendor/vmaf/LICENSE', 'vmaf/LICENSE')
    wanted_logs = ('bootstrap', 'data', 'extend-test', 'baseline-protocol-qa', 'vmaf-', 'environment-installed')
    for source in sorted((ROOT/'logs').iterdir()):
        if source.is_file() and ('-v2.' in source.name or source.name.startswith(wanted_logs)):
            copy(source, Path('logs')/source.name)
    for source in (ROOT/'results/protocol-verification/verification.json',):
        if source.exists():
            copy(source, Path('evidence')/source.relative_to(ROOT))
    checkpoints = {}
    for path in (RESULTS/'recovery/index.json', RESULTS/'head-recovery/index.json',
                 RESULTS/'head-seed20/index.json', RESULTS/'head-seed21/index.json',
                 RESULTS/'decoder-drift/training-index.json'):
        for record in read(path):
            checkpoints[record['id']] = Path(record['checkpoint'])
    for record in frozen['candidates']:
        checkpoints[record['id']] = Path(record['checkpoint'])
    checkpoint_index = {}
    for identity, source in sorted(checkpoints.items()):
        target = copy(source, Path('models')/(identity+'.pt'))
        checkpoint_index[identity] = {'file': str(target.relative_to(BUNDLE).as_posix()), 'sha256': sha(target)}
        for suffix in ('.resume.pt', '.training.json'):
            extra = source.with_suffix(suffix)
            if extra.exists():
                copy(extra, Path('training')/(identity+suffix))
    for source in sorted(RESULTS.rglob('*.training.json')):
        copy(source, Path('evidence')/source.relative_to(ROOT))
    matrix = []
    for label, path in [('uniform_screen', RESULTS/'screen/index.json'),
                        ('structure_screen', RESULTS/'structure/index.json'),
                        ('recovery_selection', RESULTS/'recovery/index.json'),
                        ('head_selection', RESULTS/'head-recovery/index.json'),
                        ('replica_selection', RESULTS/'head-seed20/index.json'),
                        ('replica_selection', RESULTS/'head-seed21/index.json')]:
        for row in read(path):
            matrix.append({'stage': label, 'id': row['id'], 'scope': row.get('scope', ''),
                           'gate': row['gate']['status'], 'worst_vmaf_ratio': row['worst_vmaf_ratio'],
                           'worst_actual_rate_relative_error': row['worst_rate_error'],
                           'decoder_mac_reduction': row['decoder_mac_reduction'],
                           'parameter_reduction': row.get('parameter_reduction', ''),
                           'note': 'screen/selection result only; not final held-out acceptance'})
    for row in read(RESULTS/'decoder-drift/selection-index.json'):
        matrix.append({'stage': 'fixed_original_encoder_drift_selection', 'id': row['id'], 'scope': row['scope'],
                       'gate': row['gate_status'], 'worst_vmaf_ratio': row['worst_vmaf_ratio'],
                       'worst_actual_rate_relative_error': row['worst_actual_rate_relative_error'],
                       'decoder_mac_reduction': row['decoder_mac_reduction'],
                       'parameter_reduction': row['parameter_reduction'],
                       'note': 'GOT training-only selection; no MOT metrics used for model selection'})
    with (BUNDLE/'method-matrix.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix[0]))
        writer.writeheader(); writer.writerows(matrix)
    points = []
    for result in summary['results']:
        directory = (RESULTS/'decoder-drift/heldout' if result['id'].startswith('drift_') else FINAL)
        document = read(directory/result['id']/'measurements.json')
        for base, candidate in zip(document['baseline'], document['candidate']):
            if base['actual_bits'] != candidate['actual_bits'] or base['bitstream_sha256'] != candidate['bitstream_sha256']:
                raise ValueError('Final exported candidate is not byte-identical')
            points.append({'candidate': result['id'], 'sequence': base['sequence'], 'target': base['target'],
                           'frames': base['frames'], 'width': base['width'], 'height': base['height'],
                           'baseline_vmaf': base['vmaf'], 'candidate_vmaf': candidate['vmaf'],
                           'vmaf_ratio': candidate['vmaf']/base['vmaf'], 'actual_bits': base['actual_bits'],
                           'actual_bitrate_bps': base['actual_bits']/base['duration_seconds'],
                           'actual_rate_relative_error': 0., 'bitstream_sha256': base['bitstream_sha256'],
                           'candidate_vmaf_p05': candidate['vmaf_p05'], 'candidate_vmaf_min': candidate['vmaf_min'],
                           'status': 'passed' if candidate['vmaf'] >= .95*base['vmaf'] else 'quality_failed'})
    with (BUNDLE/'heldout-points.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0]))
        writer.writeheader(); writer.writerows(points)
    portable = {'protocol': PROTOCOL, 'summary': summary, 'models': checkpoint_index,
                'trained_or_final_checkpoints': len(checkpoints), 'configuration_runs': len(matrix),
                'all_actual_streams_included': True, 'raw_source_images_included': False,
                'raw_decoded_yuv_included': False, 'v1_results_included': False,
                'source_archive_and_baseline_included': True,
                'model_and_training_states_scope': 'All completed trained candidates and frozen final models; untrained uniform-screen weights omitted but layouts and calibration scores included',
                'path_map': mapping,
                'replay_note': 'Run prepare_source.py, build/install msrtc-rans and VMAF, then decode_stream.py with --structured for extended models. Stream paths are addressed by their actual SHA256 under streams/.'}
    (BUNDLE/'artifact-index.json').write_text(json.dumps(portable, indent=2, ensure_ascii=False), encoding='utf-8')
    columns = ['模型', '角色', '最差VMAF保留', '解码卷积MAC减少', '相同码流', '验收']
    body = []
    for row in summary['results']:
        body.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
             (row['id'], row.get('role', 'fixed_original_encoder_drift_comparison'), f"{row['worst_vmaf_ratio']*100:.3f}%",
              f"{row['decoder_mac_reduction']*100:.3f}%", row['all_bitstreams_identical'], row['gate_status']))+'</tr>')
    doc = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MLVC-S 剪枝实验</title><style>body{max-width:1300px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#182431}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d8e0e8;padding:8px;text-align:left}th{background:#edf3f7}code{background:#eff3f7;padding:3px}aside{padding:16px;background:#fff4d9}a{color:#1263a1}</style><h1>MLVC-S 剪枝实测</h1>'
    doc += '<p>验收：每个保留视频、每个码率点，候选 VMAF 分数 ≥ 基线的95%；实际码流字节完全相同，码率容差为0。</p>'
    doc += '<aside>本包仅包含修正后的部署协议 v2。初版遗漏 LTR 恢复、帧索引偏移及色度处理的结果已归档排除。MOT17-02 原始模型曾用于协议实现 QA，没有用于候选排序、训练或检查点选择。</aside>'
    doc += '<p>测试：16段 Vimeo 官方测试片段，加7个完整 MOT17 视频，共5428帧、3个码率点、69个逐视频验收点。输入最大640×360，保持MOT17比例；Vimeo片段为原生448×256。H800、PyTorch2.8.0+cu128、CUDA12.8、FP32、关闭TF32。该结果不等同于所有分辨率、所有码率或ARM性能保证。</p>'
    doc += '<table><thead><tr>'+''.join('<th>'+c+'</th>' for c in columns)+'</tr></thead><tbody>'+''.join(body)+'</tbody></table>'
    doc += '<p><a href="heldout-points.csv">逐视频实际码率/VMAF</a> · <a href="method-matrix.csv">全部方法配置比较</a> · <a href="artifact-index.json">检查点、真实码流及路径映射</a> · <a href="evidence/results/decoder-benchmark-combined.json">真实解码计时</a></p>'
    doc += '<p>解码器读取同一实际rANS文件；独立进程不读取原始RGB输入或编码器状态。重建头模型的所有非重建头权重须与基线摘要一致；drift系列固定原编码器、量化器、熵表和潜变量几何，使用GOT10k的16段训练视频和4段选型视频恢复自身递归误差，实际码流解码必须等价于固定潜变量解码。不宣称剪枝编码器可以生成相同码流。码率包含研究容器头、帧长度和rANS载荷，不含预装模型权重。</p>'
    doc += '<h2>重放</h2><pre>python prepare_source.py\n# 依照 bootstrap.sh 安装固定环境和 msrtc-rans\npython decode_stream.py --stream streams/STREAM_SHA256.mlvcp --checkpoint models/MODEL.pt --structured --entropy-tables entropy/encoder-tables.npz --output decoded.yuv</pre><p>传统FFN检查点去掉 <code>--structured</code>。原生CPU和CUDA生成的量化熵表不同；必须保留随包实际编码器整数表和JSON哈希。旧CPU结果已隔离，只交付冻结表后重跑的CPU计时；完整69点VMAF仅在H800确认。原始训练特征缓存不打包；可按固定数据和对应脚本重新生成。训练状态保存在 training/，可供按配方恢复；当前CLI默认从预训练权重重跑。</p>'
    doc += '<p>主要依据：<a href="https://github.com/microsoft/mlvc/tree/64b10d4c0108106e66b7c54d6e5e470200a57abf">固定 MLVC 源码</a> · <a href="https://github.com/Netflix/vmaf/tree/v3.0.0">VMAF3.0.0</a> · <a href="https://www.autodl.com/console/public-data">AutoDL公共数据目录</a>。性能、失败结果和方法覆盖以本包CSV/JSON实测为准。</p></html>'
    (BUNDLE/'report.html').write_text(doc, encoding='utf-8')
    for path in sorted(BUNDLE.rglob('*')):
        if path.is_file():
            files.append({'path': str(path.relative_to(BUNDLE).as_posix()), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    (BUNDLE/'bundle-manifest.json').write_text(json.dumps({'protocol': PROTOCOL, 'files': files}, indent=2), encoding='utf-8')
    archive = ROOT/'mlvc-s-pruning-results-v2.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        tar.add(BUNDLE, arcname='mlvc-s-pruning-v2')
    receipt = {'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive),
               'files': len(files)+1, 'configuration_runs': len(matrix), 'checkpoints': len(checkpoints),
               'note': 'Archive generated on remote; local retrieval and AutoDL shutdown still required'}
    (ROOT/'export-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
