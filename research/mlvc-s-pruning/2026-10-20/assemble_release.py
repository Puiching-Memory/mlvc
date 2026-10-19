"""Verify and join the Release volumes with Python's standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('.'))
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root/'release-parts.json').read_text(encoding='utf-8'))
    name = manifest['archive']
    if Path(name).name != name or '/' in name or '\\' in name or ':' in name:
        raise ValueError('Invalid archive name')
    target = root/name
    if target.exists():
        raise FileExistsError('Inspect the existing archive before assembling again')
    temporary = target.with_name(target.name+'.assembling')
    # Exclusive creation avoids replacing another download or assembly.
    full_hash, total = hashlib.sha256(), 0
    with temporary.open('xb') as output:
        for number, entry in enumerate(manifest['parts']):
            part_name = entry['name']
            if (Path(part_name).name != part_name or '/' in part_name or
                    '\\' in part_name or ':' in part_name or
                    part_name != name+f'.part-{number:03d}'):
                raise ValueError('Unsafe or unordered volume name')
            part_hash, size = hashlib.sha256(), 0
            with (root/part_name).open('rb') as source:
                for block in iter(lambda: source.read(8*1024*1024), b''):
                    output.write(block)
                    part_hash.update(block)
                    full_hash.update(block)
                    size += len(block)
            if size != entry['bytes'] or part_hash.hexdigest() != entry['sha256']:
                raise ValueError('Volume is incomplete or changed: '+part_name)
            total += size
    if total != manifest['bytes'] or full_hash.hexdigest() != manifest['sha256']:
        raise ValueError('Assembled archive digest mismatch')
    os.rename(temporary, target)
    print(json.dumps({'status':'verified_and_assembled','archive':str(target),
                      'bytes':total,'sha256':full_hash.hexdigest()}))


if __name__ == '__main__':
    main()
