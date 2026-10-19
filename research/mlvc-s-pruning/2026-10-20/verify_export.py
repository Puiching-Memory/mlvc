"""Verify a retrieved complete archive and every manifest file before reporting."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads(a.receipt.read_text())
    if a.archive.stat().st_size != receipt['bytes'] or sha(a.archive) != receipt['sha256']:
        raise ValueError('Full retrieved archive size/digest mismatch')
    if a.destination.exists():
        raise FileExistsError('Extraction target already exists; inspect before replacing')
    with tarfile.open(a.archive, 'r:gz') as tar:
        members = tar.getmembers()
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or not path.parts or
                path.parts[0] != 'mlvc-s-pruning-v2' or member.issym() or member.islnk() or
                not (member.isdir() or member.isfile()) or ':' in member.name or '\\' in member.name):
                raise ValueError('Unsafe/unexpected archive member')
            key = member.name.casefold()
            if key in seen:
                raise ValueError('Duplicate archive member')
            seen.add(key)
        if len(members) > 100000:
            raise ValueError('Unexpected archive member count')
        a.destination.mkdir(parents=True)
        tar.extractall(a.destination, filter='data')
    root = a.destination/'mlvc-s-pruning-v2'
    manifest = json.loads((root/'bundle-manifest.json').read_text())
    total = 0
    for record in manifest['files']:
        path = root/record['path']
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Manifest file missing or outside bundle')
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Bundle file digest mismatch: '+record['path'])
        total += record['bytes']
    result = {'status': 'passed', 'full_archive_sha256': receipt['sha256'],
              'archive_bytes': receipt['bytes'], 'all_manifest_files_verified': len(manifest['files']),
              'uncompressed_verified_bytes': total, 'local_extracted_root': str(root),
              'private_keys_included': any('id_ed25519' in r['path'] for r in manifest['files'])}
    if result['private_keys_included']:
        raise ValueError('Private key must never enter experiment bundle')
    a.verification.parent.mkdir(parents=True, exist_ok=True)
    a.verification.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
