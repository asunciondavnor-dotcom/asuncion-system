"""Activate a checksum-pinned FastText PR–PO release."""

import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import sys

VERSION = 'asuncion-fasttext-calibrated-svm-v1'
NAME = 'FastText + Calibrated Linear SVM'
ROOT = Path(__file__).resolve().parent / 'inference_releases' / VERSION
ENTRY = 'release_item3_assessor'
ENTRY_DIR = 'Replacement part distinction'
MANIFEST = 'serving-manifest.json'
MAX_MANIFEST_BYTES = 200_000
_model = None


def configure(config):
    if os.environ.get('PSMS_PR_PO_FASTTEXT') != '1':
        return
    from render_model_preparation import enabled

    digest = os.environ.get('PSMS_PR_PO_FASTTEXT_MANIFEST_SHA256', '')
    if (config.get('PSMS_ENV') not in {'production', 'test'} or not enabled(config)
            or len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest)):
        raise RuntimeError('PR_PO_FASTTEXT_RELEASE_CONFIGURATION')
    config['PR_PO_FASTTEXT_MANIFEST_SHA256'] = digest


def artifact_digest(stream, size):
    if size <= 32 * 1024**2 or not hasattr(os, 'posix_fadvise'):
        return hashlib.file_digest(stream, 'sha256').hexdigest()
    digest = hashlib.sha256()
    offset = 0
    for block in iter(lambda: stream.read(4 * 1024**2), b''):
        digest.update(block)
        try:
            os.posix_fadvise(stream.fileno(), offset, len(block), os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        offset += len(block)
    return digest.hexdigest()


def verify(root, expected_sha256):
    root = Path(root).resolve(strict=True)
    with (root / MANIFEST).open('rb') as stream:
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('PR_PO_FASTTEXT_MANIFEST_INTEGRITY')
    data = json.loads(raw)
    files = data.get('files')
    entry = f'{ENTRY_DIR}/{ENTRY}.py'
    if (data.get('schema') != 'ASUNCION_FASTTEXT_SERVING_V1' or data.get('version') != VERSION
            or not isinstance(files, dict) or not 1 <= len(files) <= 300 or entry not in files):
        raise ValueError('PR_PO_FASTTEXT_MANIFEST_CONTRACT')
    actual = {path.relative_to(root).as_posix()
              for path in root.rglob('*') if path.is_symlink() or path.is_file()}
    if actual != (files.keys() | {MANIFEST}):
        raise ValueError('PR_PO_FASTTEXT_RELEASE_FILE_SET')
    for name, pin in files.items():
        path = root / name
        if (Path(name).is_absolute() or not path.is_relative_to(root) or path.resolve() != path
                or not path.is_file() or path.stat().st_size != pin['bytes']):
            raise ValueError('PR_PO_FASTTEXT_RELEASE_FILE')
        with path.open('rb') as stream:
            if artifact_digest(stream, pin['bytes']) != pin['sha256']:
                raise ValueError('PR_PO_FASTTEXT_ARTIFACT_INTEGRITY')
    return root


class Assessor:
    version = VERSION

    def __init__(self, model):
        self._model = model

    def assess(self, pr, po):
        assessment = self._model.assess(pr, po)
        assessment['model_version'] = self.version
        return assessment


def checked_model(expected_sha256, *, root=ROOT):
    root = verify(root, expected_sha256)
    module_path = root / ENTRY_DIR / (ENTRY + '.py')
    directories = [module_path.parent, root]
    directories.extend(path for path in sorted(root.iterdir()) if path.is_dir() and path != module_path.parent)
    sys.path.extend(str(path) for path in directories if str(path) not in sys.path)
    module = import_module(ENTRY)
    if Path(module.__file__).resolve() != module_path:
        raise RuntimeError('PR_PO_FASTTEXT_MODULE_IDENTITY')
    model = Assessor(module.FullAssessor())
    pr = [{'item': 'projector', 'description': 'LCD', 'qty': '7', 'unit': 'pcs'}]
    for name, quantity, expected in (
        ('video projection unit', '7', 'Acceptable'),
        ('video projection unit', '5', 'Needs Recanvass'),
        ('video projection unit', '', 'Needs Review'),
        ('television', '7', 'Reject'),
    ):
        po = [{'item': name, 'description': 'LCD', 'qty': quantity, 'unit': 'pieces'}]
        result = model.assess(pr, po)
        if result['document']['label'] != expected or result['match']['label'] != expected:
            raise RuntimeError('PR_PO_FASTTEXT_OUTPUT_CONTRACT')
    return model


def prepare(expected_sha256):
    global _model
    if _model is None:
        _model = checked_model(expected_sha256)
        from offline_ocr_mapper import _release_native_heap
        _release_native_heap()


def prepared():
    return _model is not None


def assessor():
    if _model is None:
        raise RuntimeError('PR_PO_FASTTEXT_NOT_PREPARED')
    return _model
