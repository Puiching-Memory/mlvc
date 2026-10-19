"""Select public archive members without extracting entire datasets.

Train/calibration/selection/test are separated by Vimeo source-video folder;
MOT17 FRCNN sequences are external held-out tests, never training examples.
"""
import argparse
import configparser
import hashlib
import io
import json
import random
import re
import shutil
import zipfile
from pathlib import Path
from PIL import Image
from experiment_runtime import save_json, digest


def extract(z, member, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # ZipFile.open validates the member CRC at EOF. Destination is constructed
    # independently from a selected identity, never an archive-provided path.
    with z.open(member) as source, path.open('wb') as target:
        shutil.copyfileobj(source, target)
    return {'path': str(path.resolve()), 'sha256': digest(path), 'archive_member': member}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/root/autodl-pub'))
    p.add_argument('--output', type=Path, default=Path('data'))
    p.add_argument('--train-clips', type=int, default=512)
    p.add_argument('--calibration-clips', type=int, default=32)
    p.add_argument('--selection-clips', type=int, default=8)
    p.add_argument('--test-clips', type=int, default=16)
    p.add_argument('--long-frames', type=int, default=120)
    p.add_argument('--inspect', action='store_true')
    a = p.parse_args()
    a.output = a.output.resolve()
    vimeo = a.root/'Vimeo-90k/vimeo_septuplet.zip'
    mot = a.root/'mot17/MOT17.zip'
    if a.inspect:
        for path in (vimeo, mot):
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                print(json.dumps({'path': str(path), 'bytes': path.stat().st_size,
                                  'members': len(names), 'first': names[:15],
                                  'lists': [n for n in names if n.endswith(('trainlist.txt', 'testlist.txt', 'seqinfo.ini'))][:20]}, indent=2))
        return
    document = {key: [] for key in ('train', 'calibration', 'selection', 'test')}
    document.update({'schema': 1, 'seed': 20261019, 'policy': 'Vimeo source group SHA256 bucket; official training lists for optimization; official test lists for Vimeo tests; all MOT17 external held out',
                     'source_archives': [], 'files': {}})
    rng = random.Random(document['seed'])
    limits = {'train': a.train_clips, 'calibration': a.calibration_clips,
              'selection': a.selection_clips, 'test': a.test_clips}
    with zipfile.ZipFile(vimeo) as z:
        names = z.namelist()
        list_members = {n.rsplit('/', 1)[-1]: n for n in names if n.endswith(('sep_trainlist.txt', 'sep_testlist.txt'))}
        if set(list_members) != {'sep_trainlist.txt', 'sep_testlist.txt'}:
            raise ValueError('Expected official Vimeo septuplet train/test lists')
        train_list = set(z.read(list_members['sep_trainlist.txt']).decode().split())
        test_list = set(z.read(list_members['sep_testlist.txt']).decode().split())
        if train_list & test_list:
            raise ValueError('Official clip lists overlap')
        clips = {}
        for name in names:
            match = re.search(r'(?:^|/)sequences/(\d+/\d+)/im([1-7])\.png$', name)
            if match:
                clips.setdefault(match[1], {})[int(match[2])] = name
        partitions = {key: [] for key in limits}
        for clip, frames in clips.items():
            if len(frames) != 7:
                continue
            group = clip.split('/')[0]
            bucket = int(hashlib.sha256(('mlvc-pruning-20261019:'+group).encode()).hexdigest(), 16)%10
            split = 'train' if bucket < 7 else ('calibration' if bucket == 7 else ('selection' if bucket == 8 else 'test'))
            if clip not in (test_list if split == 'test' else train_list):
                continue
            partitions[split].append(clip)
        split_groups = {}
        for split, candidates in partitions.items():
            candidates.sort()
            rng.shuffle(candidates)
            if len(candidates) < limits[split]:
                raise ValueError(f'Insufficient disjoint clips for {split}')
            split_groups[split] = set()
            for clip in candidates[:limits[split]]:
                group = clip.split('/')[0]
                split_groups[split].add(group)
                identity = 'vimeo_'+clip.replace('/', '_')
                files = []
                for index, member in sorted(clips[clip].items()):
                    record = extract(z, member, a.output/'vimeo'/split/identity/f'im{index}.png')
                    files.append(record['path'])
                    document['files'][record['path']] = record
                with Image.open(files[0]) as image:
                    width, height = image.size
                document[split].append({'id': identity, 'dataset': 'Vimeo-90k', 'source_group': group,
                                        'files': files, 'width': width, 'height': height,
                                        'fps': 30., 'fps_origin': 'fixed research playback convention; Vimeo PNGs do not retain original timestamps'})
        for first, groups in split_groups.items():
            for second, other in split_groups.items():
                if first != second and groups & other:
                    raise ValueError('Source-video group leakage')
        document['vimeo_group_counts'] = {k: len(v) for k, v in split_groups.items()}
        document['vimeo_available_clips_by_split'] = {k: len(v) for k, v in partitions.items()}
        document['source_archives'].append({'path': str(vimeo), 'bytes': vimeo.stat().st_size,
                                            'central_directory_names_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest(),
                                            'full_archive_digest': 'not_computed; selected members validated by ZIP CRC and individual SHA256'})
    with zipfile.ZipFile(mot) as z:
        names = z.namelist()
        sequences = {}
        for name in names:
            match = re.search(r'(?:^|/)(MOT17-\d{2}-FRCNN)/img1/(\d+)\.jpg$', name)
            if match and '/train/' in '/'+name:
                sequences.setdefault(match[1], []).append((int(match[2]), name))
        if len(sequences) != 7:
            raise ValueError(f'Expected seven unique MOT17 training videos, found {len(sequences)}')
        for identity, members in sorted(sequences.items()):
            selected = sorted(members)[:a.long_frames]
            if len(selected) < a.long_frames:
                raise ValueError('Insufficient long sequence frames')
            files = []
            for index, member in selected:
                record = extract(z, member, a.output/'mot17'/(identity+f'_{index:06}.jpg'))
                files.append(record['path'])
                document['files'][record['path']] = record
            info_candidates = [n for n in names if n.endswith('/'+identity+'/seqinfo.ini')]
            if len(info_candidates) != 1:
                raise ValueError('Missing MOT frame-rate metadata')
            info = configparser.ConfigParser()
            info.read_string(z.read(info_candidates[0]).decode())
            fps = float(info['Sequence']['frameRate'])
            with Image.open(files[0]) as image:
                source_w, source_h = image.size
            scale = min(640/source_w, 360/source_h)
            width, height = int(source_w*scale)//2*2, int(source_h*scale)//2*2
            document['test'].append({'id': identity, 'dataset': 'MOT17', 'source_group': identity.split('-FRCNN')[0],
                                      'files': files, 'width': width, 'height': height, 'fps': fps,
                                      'source_width': source_w, 'source_height': source_h,
                                      'resize': 'aspect-preserving fit within 640x360; Lanczos; no temporal subsampling'})
        document['source_archives'].append({'path': str(mot), 'bytes': mot.stat().st_size,
                                            'central_directory_names_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest(),
                                            'full_archive_digest': 'not_computed; selected members validated by ZIP CRC and individual SHA256'})
    hashes = {}
    for split in limits:
        for sequence in document[split]:
            # Exact selected-image duplication across splits cannot pass.
            for file in sequence['files']:
                h = document['files'][file]['sha256']
                if h in hashes and hashes[h] != split:
                    raise ValueError(f'Exact image duplicate across {hashes[h]} and {split}')
                hashes[h] = split
    document['counts'] = {k: len(document[k]) for k in limits}
    save_json(a.output/'manifest.json', document)
    print(json.dumps({'counts': document['counts'], 'groups': document['vimeo_group_counts'],
                      'manifest_sha256': digest(a.output/'manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
