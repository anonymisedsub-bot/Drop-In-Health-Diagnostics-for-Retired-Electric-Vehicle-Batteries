"""Create and verify the SHA-256 inventory of a local release.

No source directory, network connection or scientific Python package is needed.
The manifest detects accidental modifications; it is not a digital signature.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {'.git', '__pycache__', '.pytest_cache', '.venv', 'runs'}
MANIFEST_NAME = 'release_manifest.json'

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def create_manifest(root: Path) -> dict:
    root = root.resolve()
    files = {}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if (not path.is_file() or any(p in EXCLUDED_PARTS for p in rel.parts)
                or rel.as_posix() == MANIFEST_NAME or path.suffix in {'.pyc', '.pyo'}):
            continue
        if path.is_symlink():
            raise ValueError(f'Release files cannot be symlinks: {rel}')
        files[rel.as_posix()] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    return {'schema_version': 1, 'algorithm': 'sha256',
            'excluded_directory_names': sorted(EXCLUDED_PARTS), 'files': files}

def verify_manifest(root: Path, manifest: dict) -> list[str]:
    root = root.resolve()
    errors = []
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root:
            errors.append(f'unsafe path: {name}')
        elif not path.is_file():
            errors.append(f'missing: {name}')
        elif path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            errors.append(f'changed: {name}')
    return errors

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--create', action='store_true', help='Freeze current files (release maintenance only).')
    args = parser.parse_args(argv)
    manifest_path = args.root / MANIFEST_NAME
    if args.create:
        manifest = create_manifest(args.root)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'Inventoried {len(manifest["files"])} files in {manifest_path}')
        return 0
    if not manifest_path.is_file():
        parser.error(f'Missing release inventory: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    errors = verify_manifest(args.root, manifest)
    print(json.dumps({'verified_files': len(manifest['files']), 'errors': errors}, indent=2))
    return 1 if errors else 0

if __name__ == '__main__':
    raise SystemExit(main())
