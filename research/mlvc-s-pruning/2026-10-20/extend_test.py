"""Freeze all seven MOT17 videos before any held-out quality evaluation.

This extends test duration without changing train/calibration/selection files.
"""
import configparser
import copy
import json
import re
import zipfile
from pathlib import Path
from experiment_runtime import ROOT, digest, save_json
from prepare_data import extract


def main():
    source = ROOT/'data/manifest.json'
    original = json.loads(source.read_text())
    output = ROOT/'data/test-full/manifest.json'
    if output.exists():
        record = json.loads(output.read_text())
        if record['training_manifest_sha256'] != digest(source):
            raise ValueError('Training/selection manifest changed')
        print(json.dumps(record['counts'], indent=2))
        return
    document = {'training_manifest_sha256': digest(source),
                'policy': 'Full duration of all seven deduplicated MOT17 training videos, external held out from Vimeo optimization; plus the original sixteen Vimeo official-test clips. No temporal subsampling.',
                'test': [], 'files': {}, 'frozen_before_any_heldout_quality_measurement': True}
    for sequence in original['test']:
        if sequence['dataset'] == 'Vimeo-90k':
            document['test'].append(copy.deepcopy(sequence))
            for path in sequence['files']:
                document['files'][path] = original['files'][path]
    archive = Path('/root/autodl-pub/mot17/MOT17.zip')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for old in [s for s in original['test'] if s['dataset'] == 'MOT17']:
            sequence = copy.deepcopy(old)
            info_path = next(n for n in names if n.endswith('/'+sequence['id']+'/seqinfo.ini'))
            info = configparser.ConfigParser()
            info.read_string(z.read(info_path).decode())
            expected = int(info['Sequence']['seqLength'])
            members = []
            for name in names:
                match = re.search(r'/'+re.escape(sequence['id'])+r'/img1/(\d+)\.jpg$', name)
                if match:
                    members.append((int(match[1]), name))
            members.sort()
            if [i for i, _ in members] != list(range(1, expected+1)):
                raise ValueError('Full source video frame coverage is incomplete')
            files = []
            for index, member in members:
                target = ROOT/'data/mot17'/(sequence['id']+f'_{index:06}.jpg')
                path = str(target.resolve())
                if path in original['files']:
                    record = original['files'][path]
                    if digest(path) != record['sha256']:
                        raise ValueError('Previously frozen frame changed')
                else:
                    record = extract(z, member, target)
                document['files'][path] = record
                files.append(path)
            sequence['files'] = files
            sequence['full_source_video'] = True
            document['test'].append(sequence)
            print(json.dumps({'full_test_video': sequence['id'], 'frames': len(files), 'fps': sequence['fps']}), flush=True)
    document['counts'] = {'sequences': len(document['test']),
                           'frames_once': sum(len(s['files']) for s in document['test']),
                           'MOT17_full_videos': 7, 'Vimeo_clips': 16}
    save_json(output, document)
    print(json.dumps({'counts': document['counts'], 'sha256': digest(output)}, indent=2))


if __name__ == '__main__':
    main()
