"""Negative controls for the quality contract. Fixture scores are synthetic."""
import copy
import json
from pathlib import Path
from quality_gate import evaluate

b = dict(sequence='fixture', target='rate1', source_sha256='0'*64, frames=96,
         width=640, height=360, color_profile='BT709-full-YUV420',
         vmaf_model_sha256='1'*64, metric_build_sha256='2'*64,
         actual_bits=100000, duration_seconds=3.2, vmaf=80., rate_source='actual_bitstream', protocol='fixture_protocol')
doc = dict(registered_pairs=[['fixture','rate1']], baseline=[b],candidate=[copy.deepcopy(b)])
checks=[]
for label, field, value, expected in [
    ('exact_boundary','vmaf',76.,'passed'),
    ('quality_failure','vmaf',75.99,'not_passed'),
    ('rate_inflation','actual_bits',110000,'not_passed'),
    ('estimated_rate','rate_source','estimated','error'),
    ('frame_mismatch','frames',95,'error'),
    ('different_reference','source_sha256','3'*64,'error'),
    ('different_metric_model','vmaf_model_sha256','4'*64,'error'),
    ('different_codec_protocol','protocol','different_protocol','error'),
    ('nan_metric','vmaf',float('nan'),'error'),
    ('fractional_frames','frames',95.5,'error')]:
    d=copy.deepcopy(doc);d['candidate'][0][field]=value
    try: status=evaluate(d)['status']
    except ValueError: status='error'
    assert status==expected,(label,status,expected)
    checks.append(dict(case=label,status='passed'))
for label, changed in [('missing_candidate',dict(candidate=[])),
                       ('omitted_registered_target',dict(registered_pairs=[['fixture','rate1'],['fixture','rate2']]))]:
    d=copy.deepcopy(doc);d.update(changed)
    try:
        evaluate(d)
        raise AssertionError(label)
    except ValueError: checks.append(dict(case=label,status='passed'))
result=dict(status='passed',scope='Synthetic contract logic only; no measured VMAF',checks=checks)
Path(__file__).with_name('quality-gate-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
