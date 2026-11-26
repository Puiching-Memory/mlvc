"""Render standalone scientific plots and a Chinese report from exported evidence."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def label(name):
    if name.startswith('drift_'):
        return name.replace('drift_', '').replace('_taylor_', ' ').replace('active_decoder', 'Full decoder').replace('main_decoder', 'Main decoder')
    if 'outer50' in name:
        return 'Head outer50 + FFN50'
    if '_p750_' in name:
        return 'Head FFN75 + KD'
    if '_p625_' in name:
        return 'Head FFN62.5 + KD ('+name[-2:]+')'
    if '_p500_' in name:
        return 'Head FFN50 + RD'
    if name == 'reconstruction_taylor_p125':
        return 'Head FFN12.5'
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    index = read(a.bundle/'artifact-index.json')
    summary = index['summary']
    rows = summary['results']
    passed = [r for r in rows if r['gate_status'] == 'passed']
    if not passed:
        raise ValueError('No measured candidate passed')
    best = max(passed, key=lambda r: r['decoder_mac_reduction'])
    primary = next(r for r in rows if r['id'] == summary['primary'])
    with (a.bundle/'heldout-points.csv').open(encoding='utf-8-sig', newline='') as f:
        points = list(csv.DictReader(f))
    with (a.bundle/'method-matrix.csv').open(encoding='utf-8-sig', newline='') as f:
        methods = list(csv.DictReader(f))
    benchmark = read(a.bundle/'evidence/results/decoder-benchmark-combined.json')
    cpu_benchmark = read(a.bundle/'evidence/results/decoder-benchmark-cpu.json')
    drift_selection = read(a.bundle/'evidence/results/decoder-drift/selection-index.json')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 140, 'savefig.dpi': 180})
    figure, axes = plt.subplots(1, 2, figsize=(14, max(4.5, len(rows)*.48)), constrained_layout=True)
    y = np.arange(len(rows))
    colors = ['#157f6d' if r['gate_status'] == 'passed' else '#c24c3a' for r in rows]
    quality = [r['worst_vmaf_ratio']*100 for r in rows]
    axes[0].barh(y, quality, color=colors)
    axes[0].axvline(95, color='#212f3d', ls='--', lw=1.4, label='95% acceptance')
    axes[0].set(yticks=y, yticklabels=[label(r['id']) for r in rows], xlim=(min(90, min(quality)-1), 101),
                xlabel='Worst sequence/rate VMAF retention (%)', title='All 69 held-out points must pass')
    axes[0].invert_yaxis()
    for i, value in enumerate(quality):
        axes[0].text(value+.12, i, f'{value:.2f}', va='center', fontsize=9)
    macs = [r['decoder_mac_reduction']*100 for r in rows]
    axes[1].barh(y, macs, color=colors)
    axes[1].set(yticks=y, yticklabels=[], xlabel='Executed decoder Conv2d MAC reduction (%)',
                xlim=(0, max(macs)*1.18), title='Physical compact channels; fixed original streams')
    axes[1].invert_yaxis()
    for i, value in enumerate(macs):
        axes[1].text(value+.25, i, f'{value:.2f}', va='center', fontsize=9)
    figure.savefig(a.output/'quality-compute.png'); figure.savefig(a.output/'quality-compute.svg'); plt.close(figure)
    selected = [r for r in points if r['candidate'] == best['id']]
    sequences = list(dict.fromkeys(r['sequence'] for r in selected))
    targets = ['q12', 'q32', 'q52']
    values = np.array([[float(next(r['vmaf_ratio'] for r in selected if r['sequence'] == s and r['target'] == t))*100
                        for t in targets] for s in sequences])
    fig, ax = plt.subplots(figsize=(7, max(8, len(sequences)*.36)), constrained_layout=True)
    img = ax.imshow(values, vmin=95, vmax=100, cmap='YlGn', aspect='auto')
    ax.set(xticks=range(3), xticklabels=targets, yticks=range(len(sequences)), yticklabels=sequences,
           title='VMAF retention at identical actual bytes\n'+label(best['id']))
    for i in range(len(sequences)):
        for j in range(3):
            ax.text(j, i, f'{values[i,j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if values[i,j] >= 98.5 else '#18332c')
    fig.colorbar(img, ax=ax, label='VMAF retention (%)', shrink=.6)
    fig.savefig(a.output/'per-sequence-retention.png'); fig.savefig(a.output/'per-sequence-retention.svg'); plt.close(fig)
    timings = benchmark['results']
    fig, ax = plt.subplots(figsize=(9, 4.3), constrained_layout=True)
    for i, (name, row) in enumerate(timings.items()):
        series = np.array(row['runs_seconds'])*1000/benchmark['frames_per_run']
        ax.scatter(np.full(len(series), i)+np.linspace(-.12,.12,len(series)), series, alpha=.65, color='#177e8b', s=22)
        ax.plot([i-.25, i+.25], [row['median_ms_per_frame']]*2, color='#212f3d', lw=2)
    ax.set(xticks=range(len(timings)), xticklabels=['Baseline' if n == 'baseline' else label(n) for n in timings],
           ylabel='Real rANS decode latency (ms/frame)', title='H800 PCIe FP32; 10 cross-ordered runs; TF32 off')
    ax.tick_params(axis='x', labelrotation=15)
    fig.savefig(a.output/'decoder-latency.png'); fig.savefig(a.output/'decoder-latency.svg'); plt.close(fig)
    cpu_timings = cpu_benchmark['results']
    fig, ax = plt.subplots(figsize=(9, 4.3), constrained_layout=True)
    for i, (name, row) in enumerate(cpu_timings.items()):
        series = np.array(row['runs_seconds'])*1000/cpu_benchmark['frames_per_run']
        ax.scatter(np.full(len(series), i)+np.linspace(-.1,.1,len(series)), series, color='#85763b', s=25)
        ax.plot([i-.25, i+.25], [row['median_ms_per_frame']]*2, color='#212f3d', lw=2)
    ax.set(xticks=range(len(cpu_timings)), xticklabels=['Baseline' if n == 'baseline' else label(n) for n in cpu_timings],
           ylabel='Real rANS decode latency (ms/frame)', title='Host x86 CPU FP32; 8 threads; 5 cross-ordered runs')
    ax.tick_params(axis='x', labelrotation=15)
    fig.savefig(a.output/'decoder-latency-cpu.png'); fig.savefig(a.output/'decoder-latency-cpu.svg'); plt.close(fig)
    worst = min(selected, key=lambda r: float(r['vmaf_ratio']))
    result = {'status': 'completed', 'configuration_runs': len(methods), 'heldout_candidates': len(rows),
              'test_sequences': len(sequences), 'pairs_per_candidate': 69, 'frames_once': 5428,
              'acceptance': 'All69 sequence/rate pairs; VMAF>=95% baseline; exactly identical actual stream bytes',
              'maximum_validated_mac_reduction': best, 'pretest_selected_margin_model': primary,
              'maximum_mac_candidate_worst_point': worst, 'benchmark': benchmark, 'cpu_benchmark': cpu_benchmark,
              'universal_all_video_or_all_bitrate_guarantee': False}
    (a.output/'results-summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    import hashlib
    (a.output/'report-render-environment.json').write_text(json.dumps({'matplotlib': matplotlib.__version__,
         'numpy': np.__version__, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'training_runtime_modified': False}, indent=2), encoding='utf-8')
    model_file = index['models'][best['id']]['file']
    structured = ' --structured' if best['kind'] == 'extended' else ''
    instruction = ('MLVC-S 剪枝实测交付\n\n'
       f'最大通过模型：{best["id"]}\n'
       f'全部69点最差VMAF保留：{best["worst_vmaf_ratio"]*100:.4f}%\n'
       f'整体解码卷积MAC减少：{best["decoder_mac_reduction"]*100:.4f}%\n'
       '候选与基线实际码流完全相同，码率差为0。\n\n'
       '完整包解压后进入 mlvc-s-pruning-v2 目录。先执行 python prepare_source.py，\n'
       '依照 bootstrap.sh 准备 PyTorch2.8.0+cu128/Python3.12 和 msrtc-rans。\n'
       '下面命令只解码，不需要原视频或编码器状态：\n\n'
       f'python decode_stream.py --stream streams/{worst["bitstream_sha256"]}.mlvcp --checkpoint {model_file}{structured} --entropy-tables entropy/encoder-tables.npz --output decoded.yuv\n\n'
       '输出为有限范围8位YUV420，宽高/帧数见码流头和逐视频CSV。\n'
       'CPU和CUDA原生生成的量化熵表不同；必须保留随包冻结的实际编码器整数表和JSON哈希。\n'
       '默认CUDA；加 --device cpu 可在CPU读取同一流，但本次完整69点VMAF仅在H800确认。\n'
       'drift系列固定原编码器，只剪枝解码器；不要用改过上下文的compress路径声称生成相同流。\n'
       '训练状态可供按配方恢复，当前CLI默认从预训练权重重跑，并无自动恢复入口。\n'
       '详见 report.html、heldout-points.csv、method-matrix.csv、artifact-index.json。\n')
    (a.output/'README.txt').write_text(instruction, encoding='utf-8')
    body = []
    for row in rows:
        timing = timings.get(row['id'], {})
        cpu_timing = cpu_timings.get(row['id'], {})
        vals = [row['id'], f"{row['worst_vmaf_ratio']*100:.3f}%", f"{row['decoder_mac_reduction']*100:.3f}%",
                f"{row['parameter_reduction']*100:.3f}%", str(row['test_pairs']),
                '通过' if row['gate_status'] == 'passed' else '未通过',
                f"{timing['median_ms_per_frame']:.3f}" if timing else '未单独计时',
                f"{cpu_timing['median_ms_per_frame']:.3f}" if cpu_timing else '未单独计时']
        body.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
    b = best
    report = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MLVC-S剪枝实测报告</title><style>body{max-width:1300px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#182431}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d8e0e8;padding:8px;text-align:left}th{background:#edf3f7}aside{padding:16px;background:#fff4d9}a{color:#1263a1}img{max-width:100%;height:auto}code{overflow-wrap:anywhere}</style><h1>MLVC-S 剪枝实测报告</h1>'
    report += f'<p><b>已验证可剪枝。</b>通过全部69个测试点的最大解码卷积MAC降幅为 <b>{b["decoder_mac_reduction"]*100:.2f}%</b>，最差VMAF保留 <b>{b["worst_vmaf_ratio"]*100:.2f}%</b>；总参数减少 {b["parameter_reduction"]*100:.2f}%。候选与基线读取相同rANS文件，码流SHA256一致，实际码率差为0。</p>'
    report += '<p>预先选定且质量余量更大的主模型为 <code>'+html.escape(primary['id'])+f'</code>，MAC减少{primary["decoder_mac_reduction"]*100:.2f}%，最差保留{primary["worst_vmaf_ratio"]*100:.2f}%。更高压缩版本是否适合部署，需结合下表中的失败点和计时结果判断。</p>'
    report += '<aside>验收单位是每个视频在每个目标码率下的整段平均VMAF，69点全部要求≥95%，不是只比较全局平均。证明范围为本包固定分辨率、数据、码率和部署协议；不构成所有视频、所有码率的普遍保证。</aside>'
    report += '<h2>测试和计时结果</h2><img src="quality-compute.png" alt="质量与计算量"><table><thead><tr>'+''.join('<th>'+c+'</th>' for c in ['模型','最差VMAF保留','解码MAC减少','总参数减少','测试点','验收','H800 ms/帧','x86 CPU ms/帧'])+'</tr></thead><tbody>'+''.join(body)+'</tbody></table>'
    report += '<p>MAC仅统计实际执行的Conv2d乘加，不含偏置、激活、熵编解码和数据传输。参数减少指整个已加载模型的参数数目。真实延迟包含rANS、上下文、递归特征和重建；不含文件读取、指标和输出像素拷贝。H800 FP32计时不代表ARM或手机加速。</p><img src="decoder-latency.png" alt="真实解码计时">'
    report += '<p>另测同一服务器x86 CPU后端，固定8线程，70帧跨越LTR64；5轮交错顺序，训练和GPU验证全部停止后执行。CPU硬件：'+html.escape(str(cpu_benchmark['hardware']))+'。CPU熵表已与实际GPU编码器核对；CPU侧仅计时，没有重跑完整69点VMAF确认，也不等同于ARM手机实测。</p><img src="decoder-latency-cpu.png" alt="x86 CPU真实解码计时">'
    report += '<aside>发现原生CPU与CUDA生成的量化熵表不相同。旧CPU计时和临时包已经隔离，本报告只读取修复后重新运行的结果。交付的 <code>entropy/encoder-tables.npz</code> 冻结实际H800编码器的整数表，JSON记录文件SHA256、表内容哈希、熵模型权重和潜变量几何；跨设备解码必须使用该表。冻结表不改权重、不改码流，独立CUDA重放仍要求像素SHA256完全一致。</aside>'
    report += '<h2>逐视频质量</h2><img src="per-sequence-retention.png" alt="69点质量保留热图">'
    report += '<h2>数据和方法</h2><p>Vimeo-90k：官方训练集512段训练、32段校准、8段选型；官方测试集16段。七个MOT17 FRCNN完整视频外部留出测试，排除DPM/SDP重复图像。测试5428帧，QP12/32/52，共69点。Vimeo448×256；MOT保持比例限制640×360，其中MOT05为480×360。MOT04的1050帧覆盖IDR1024切换。GOT10k仅使用官方训练档案，16段×96帧用于新增递归恢复，4段×96帧独立选型；未使用其官方测试/验证档案优化。</p>'
    report += f'<p>共{len(methods)}个配置：六种通道排序、三种剪枝范围、六档比例；深度卷积内部宽度、重建头整体宽度、删除块、按MAC非均匀分配；非结构化和2:4掩码作为无稀疏内核对照；五类恢复日程；重建头蒸馏、三个训练种子；固定原编码器的长序列递归潜变量蒸馏。日程是本研究实现，不宣称复现同名论文；方法覆盖有明确清单，不宣称穷尽所有可能方法。</p>'
    report += '<p>初筛使用2个Vimeo短片×3个码率点；恢复选型使用8个Vimeo短片×3点。探索矩阵的实际码率容差为1%，仅用于选型，不作为最终验收。最终重建头和新增GOT固定码流实验都要求字节完全一致，容差为0。码率不匹配的探索结果不能用于证明同码率质量达标。</p>'
    report += '<p>新增drift系列不改原编码器、量化器、熵表及潜变量几何；训练剪枝解码器读取固定教师残差并维护自己的参考状态。先核对真实rANS解码与潜变量解码在跨LTR64时一致，再用GOT选型，冻结后才读取69点测试结果。重建头系列还额外要求所有非重建头权重与基线完全一致。最终验证为独立进程从实际文件解码，不共享编码器张量。</p>'
    report += '<h3>新增递归恢复路线的GOT长视频选型</h3><p>六个固定配置各训练100个96帧视频/码率样本，前25个样本采用教师参考预热，之后维护自己的参考状态；截断反向传播16帧。下表是4段独立GOT训练视频×3码率的选型结果，不是MOT测试结果。未通过者不进入最终保留集；失败仅界定这次配方和训练规模，不能推出整个剪枝方向不可行。</p><table><thead><tr><th>模型</th><th>解码MAC减少</th><th>最差VMAF保留</th><th>选型</th></tr></thead><tbody>'
    for r in drift_selection:
        report += '<tr><td>'+html.escape(r['id'])+f'</td><td>{r["decoder_mac_reduction"]*100:.2f}%</td><td>{r["worst_vmaf_ratio"]*100:.2f}%</td><td>'+('通过' if r['gate_status'] == 'passed' else '未通过')+'</td></tr>'
    report += '</tbody></table>'
    report += '<h2>固定协议与实现核对</h2><p>基线：MLVC-S PSNR v1，源版本64b10d4c0108106e66b7c54d6e5e470200a57abf；检查点SHA256 '+CHECKPOINT_SHA+'.</p>'
    report += '<p>采用固定源码部署默认LTR8/64、主动长期参考恢复、IDR1024、帧类型索引偏移+1、零特征初始化和最近邻色度展开。BT709全范围8位420规范化输入，VMAF使用有限范围420、VMAF3.0.0和vmaf_v0.6.1模型。研究容器计入头、帧长度和rANS载荷；没有填充字节匹配码率，预装模型权重不计入传输。</p><aside>初版v1遗漏LTR恢复、帧索引偏移和色度处理，结果已隔离排除。本包仅交付修正后全部重跑的v2。MOT17-02基线曾用于实现QA；没有用MOT候选指标选择通道、训练或修改检查点。</aside>'
    report += '<h2>交付与重放</h2><p>压缩包包含固定源码、原始和训练后检查点、物理剪枝布局、训练状态、数据文件哈希清单、实际rANS流、逐帧VMAF JSON、失败结果、环境和任务日志。原视频、训练特征缓存和原始YUV不打包，可按公共数据清单再生成。</p><p><a href="mlvc-s-pruning-results-v2.tar.gz">完整实验包</a> · <a href="results-summary.json">结果JSON</a> · <a href="heldout-points.csv">逐视频实际码率/VMAF CSV</a> · <a href="method-matrix.csv">方法配置CSV</a> · <a href="artifact-index.json">模型与码流路径映射</a></p>'
    report += '<p>独立重放：先执行prepare_source.py并按bootstrap.sh准备固定环境。扩展模型：<code>python decode_stream.py --stream streams/码流SHA256.mlvcp --checkpoint models/模型.pt --structured --entropy-tables entropy/encoder-tables.npz --output decoded.yuv</code>；普通FFN/drift模型去掉--structured。drift模型继续使用原始编码器产生码流，不使用其修改过上下文的compress路径。</p></html>'
    (a.output/'report.html').write_text(report, encoding='utf-8')
    for filename in ('method-matrix.csv', 'heldout-points.csv', 'artifact-index.json'):
        (a.output/filename).write_bytes((a.bundle/filename).read_bytes())
    print(json.dumps({'output': str(a.output), 'best': best['id'], 'worst_retention': best['worst_vmaf_ratio'], 'configurations': len(methods)}))


CHECKPOINT_SHA = '1b86b757ddb115342293efb57719d6216c6ee2e459ae796ec41723b5c05ca896'


if __name__ == '__main__':
    main()
