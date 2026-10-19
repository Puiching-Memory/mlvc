"""Verify trusted inputs before unpacking the pinned source archive."""
import hashlib
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parent
expected={
    'mlvc-source-64b10d4.zip':'c2878a303711e1889637d8547519df57f3fffb004f20472141f95058ac9a3aa3',
    'mlvc-s-psnr-v1.ckpt':'1b86b757ddb115342293efb57719d6216c6ee2e459ae796ec41723b5c05ca896'
}
for name,digest in expected.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
target=root/'vendor'/'mlvc'
target.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(root/'mlvc-source-64b10d4.zip') as z:
    for member in z.infolist():
        if not (target/member.filename).resolve().is_relative_to(target.resolve()):
            raise ValueError('Unsafe archive member')
    z.extractall(target)
(root/'source-integrity.json').write_text(json.dumps(expected,indent=2),encoding='utf-8')
print('Pinned MLVC source and official checkpoint verified')
