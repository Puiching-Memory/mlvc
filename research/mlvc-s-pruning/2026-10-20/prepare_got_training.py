"""Freeze a small long-sequence training/selection split from GOT10k train.

No MOT17 or GOT10k official test/validation video is used for optimization.
"""
import argparse
import hashlib
import json
import random
import re
import zipfile
from pathlib import Path
from PIL import Image
from experiment_runtime import ROOT, save_json, digest
from prepare_data import extract


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--frames', type=int, default=96)
    p.add_argument('--train-videos', type=int, default=16)
    p.add_argument('--selection-videos', type=int, default=4)
    a = p.parse_args()
    source = Path('/root/autodl-pub/GOT10k/train_data')
    archives = sorted(source.glob('GOT-10k_Train_split_*.zip'))[:2]
    if len(archives) != 2:
        raise ValueError('Two known public GOT10k training archives required')
    groups = {}
    archive_records = []
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if a.inspect:
                print(json.dumps({'path': str(archive), 'members': len(names), 'first': names[:12]}, indent=2))
                continue
            for name in names:
                match = re.search(r'(GOT-10k_Train_\d+)/(\d+)\.jpg$', name)
                if match:
                    groups.setdefault(match[1], []).append((int(match[2]), str(archive), name))
            archive_records.append({'path': str(archive), 'bytes': archive.stat().st_size,
                    'member_names_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest()})
    if a.inspect:
        return
    candidates = sorted([name for name, frames in groups.items() if len(frames) >= a.frames],
                        key=lambda name: hashlib.sha256(('mlvc-drift-20261019:'+name).encode()).hexdigest())
    train, selection = [], []
    hashes = {}
    for identity in candidates:
        if len(train) == a.train_videos and len(selection) == a.selection_videos:
            break
        subset = sorted(groups[identity])[:a.frames]
        archive = Path(subset[0][1])
        if any(Path(row[1]) != archive for row in subset):
            raise ValueError('Video spans unexpected archives')
        with zipfile.ZipFile(archive) as z:
            with z.open(subset[0][2]) as f, Image.open(f) as image:
                source_w, source_h = image.size
            if min(source_w, source_h) < 256:
                continue
            split = 'selection' if len(selection) < a.selection_videos else 'train'
            files, records = [], []
            for index, _, member in subset:
                target = ROOT/'data/got-long'/split/identity/f'{index:08}.jpg'
                record = extract(z, member, target)
                if record['sha256'] in hashes and hashes[record['sha256']] != split:
                    raise ValueError('Exact frame duplicate across long-sequence splits')
                hashes[record['sha256']] = split
                files.append(record['path'])
                records.append(record)
        scale = min(1., 640/source_w, 360/source_h)
        sequence = {'id': identity, 'dataset': 'GOT10k_train_public', 'source_group': identity,
                    'files': files, 'file_records': records, 'width': int(source_w*scale)//2*2,
                    'height': int(source_h*scale)//2*2, 'source_width': source_w, 'source_height': source_h,
                    'fps': 30., 'fps_origin': 'fixed playback convention; source JPEG archive has no timing used here',
                    'training_crop': 'one fixed random128x128 crop from256x256 per video/rate bank item'}
        (selection if split == 'selection' else train).append(sequence)
        print(json.dumps({'split': split, 'video': identity, 'frames': len(files)}), flush=True)
    if len(train) != a.train_videos or len(selection) != a.selection_videos:
        raise ValueError('Insufficient valid long-sequence videos')
    document = {'train': train, 'selection': selection, 'test': [], 'archives': archive_records,
                 'policy': 'Only GOT10k official training archives; SHA256 order determines 4 selection videos and16 training videos; sequence identities and exact frame hashes disjoint; no MOT frame used',
                 'seed': 20261019, 'frames_per_video': a.frames,
                 'pruning_selection_uses_mot_metrics': False}
    output = ROOT/'data/got-long/manifest.json'
    save_json(output, document)
    print(json.dumps({'manifest_sha256': digest(output), 'training_videos': len(train), 'selection_videos': len(selection)}))


if __name__ == '__main__':
    main()
