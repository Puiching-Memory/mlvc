"""Baseline protocol QA only; never chooses a pruned model from MOT metrics."""
import json
from experiment_runtime import ROOT, PROTOCOL, load_model, configure_torch, measure, save_json

configure_torch()
manifest = json.loads((ROOT/'data/test-full/manifest.json').read_text())
sequence = next(s for s in manifest['test'] if s['id'] == 'MOT17-02-FRCNN')
model = load_model()
row = measure(model, sequence, 32, ROOT/'results/baseline-protocol-qa/mot02_q32.mlvcp',
              ROOT/'vendor/vmaf/libvmaf/build/tools/vmaf', ROOT/'vendor/vmaf/model/vmaf_v0.6.1.json')
metrics = json.loads(__import__('pathlib').Path(row['vmaf_json_path']).read_text())
scores = [f['metrics']['vmaf'] for f in metrics['frames']]
summary = {**row, 'protocol': PROTOCOL,
           'qa_only_not_pruning_selection': True,
           'temporal_means_20frames': [{'start': start, 'vmaf_mean': sum(scores[start:start+20])/len(scores[start:start+20])}
                                      for start in (0,20,60,100,300,580)]}
save_json(ROOT/'results/baseline-protocol-qa/report.json', summary)
print(json.dumps(summary, indent=2))
