"""Prepare only the reviewed Render runtime files; no application or DB import."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
TEXT = {'.py', '.json', '.html', '.css', '.js', '.txt', '.crt'}


def fingerprint(path):
    value = Path(path).read_bytes()
    if Path(path).suffix in TEXT:
        value = value.replace(b'\r\n', b'\n')
    return hashlib.sha256(value).hexdigest()


def prepare(destination, *, base=ROOT, strict_context=False):
    sys.path.insert(0, str(base))
    import inference_releases as releases
    base, destination = Path(base), Path(destination)
    frontend = base.parent / 'frontend'
    split_frontend = base.name == 'backend' and frontend.is_dir()
    manifest = releases.read_json(base / 'deploy/render_runtime_manifest.json')
    releases.require(manifest.get('schema') == 'ASUNCION_RENDER_SOURCE_V1'
        and isinstance(manifest.get('files'),dict) and 1<=len(manifest['files'])<=512,
        'RENDER_SOURCE_SET_MISMATCH')
    releases.require(not destination.exists(), 'RENDER_NEW_OUTPUT_REQUIRED')
    if strict_context:
        allowed=set(manifest['files'])|{'Dockerfile.render','Dockerfile.render.dockerignore',
            'deploy/render_runtime_manifest.json','deploy/render_delivery.py',
            'deploy/render_image_check.py','deploy/render_start.py'}
        actual={p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
        if split_frontend:
            actual.update(
                p.relative_to(frontend).as_posix()
                for folder in ('templates', 'static')
                for p in (frontend / folder).rglob('*')
                if p.is_file()
            )
        releases.require(actual<=allowed,'RENDER_UNEXPECTED_CONTEXT_FILE')
    sources = {}
    for name, expected in manifest['files'].items():
        if split_frontend and name.startswith(('templates/', 'static/')):
            source = releases.safe_path(base.parent, 'frontend/' + name)
        else:
            source = releases.safe_path(base, name)
        releases.require(fingerprint(source) == expected, 'RENDER_SOURCE_CHANGED')
        sources[name] = source
    releases.metadata('nlp-online-original-v2', base=base, dependencies=False,
                      profiles=('retrained_rf', 'semantic_ocr'))
    destination.mkdir(parents=True)
    for name, source in sources.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(base / 'deploy/render_image_check.py', destination / 'deploy/render_image_check.py')
    shutil.copyfile(base / 'deploy/render_start.py', destination / 'deploy/render_start.py')
    return {'source_commit': manifest['source_commit'], 'files': len(manifest['files']),
            'database_contacted': False, 'weights_in_source': False}


if __name__ == '__main__':
    print(json.dumps(prepare(sys.argv[1],strict_context=True), sort_keys=True))
