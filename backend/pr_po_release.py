"""The immutable PR–PO model deployed alongside the approved OCR release."""

import hashlib
import os
from pathlib import Path


VERSION = 'asuncion-minilm-linear-svm-v1'
NAME = 'MiniLM + Linear SVM'
ROOT = Path(__file__).resolve().parent / 'inference_releases'
ORIGINAL = 'nlp-online-original-v2'
ARTIFACTS = {
    'minilm-fp32.ort': '949b416b3a8e8428bf3b9742b454097042951c7e4ccf8be77a8e18d8d52a0187',
    'linear-svm.json': '1e13f83352ccd1f7fab4327a8d96aed4549bb1a14348f68529b2a6dbdc33773d',
}

_assessor = None


def configure(config):
    if os.environ.get('PSMS_RENDER_PR_PO_MODEL') != '1':
        return
    from render_model_preparation import enabled
    if config.get('PSMS_ENV') != 'production' or not enabled(config) or config.get('_PSMS_INFERENCE_RELEASE_ID') != ORIGINAL:
        raise RuntimeError('PR_PO_MODEL_RELEASE_CONFIGURATION')
    config['PR_PO_LINEAR_MODEL_ENABLED'] = True
    config['PR_PO_ENSEMBLE_ACTIVE'] = os.environ.get('PSMS_PR_PO_ENSEMBLE_ACTIVE') == '1'


def load_semantic_encoder(*, root=ROOT):
    from inference_releases import metadata, safe_path
    from pr_po_onnx_encoder import MAPPED_ORT_VERSION, MiniLMEncoder, ort

    if ort.__version__ != MAPPED_ORT_VERSION:
        raise RuntimeError('PR_PO_ENCODER_RUNTIME_VERSION')
    artifacts = safe_path(root, VERSION, exists=False)
    tokenizer = safe_path(root, f'{ORIGINAL}/semantic/tokenizer.json')
    pin = metadata(ORIGINAL, dependencies=False, profiles=('semantic_ocr',))['manifest']['artifacts']['semantic/tokenizer.json']
    with tokenizer.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != pin['sha256']:
            raise RuntimeError('PR_PO_TOKENIZER_INTEGRITY')
    return MiniLMEncoder(tokenizer.parent, safe_path(artifacts, 'minilm-fp32.ort'), ARTIFACTS['minilm-fp32.ort'])


def load(*, root=ROOT):
    from inference_releases import safe_path
    from pr_po_linear_runtime import load_parameters
    from pr_po_model_assessment import Assessor

    artifacts = safe_path(root, VERSION, exists=False)
    parameters = load_parameters(safe_path(artifacts, 'linear-svm.json'), ARTIFACTS['linear-svm.json'])
    encoder = load_semantic_encoder(root=root)
    return Assessor(**parameters, encode=lambda texts: encoder.encode(texts, normalize_embeddings=True), version=VERSION)


def prepare():
    global _assessor
    if _assessor is None:
        _assessor = checked_model()


class SemanticEncoder:
    """Reuse the checksum-verified MiniLM session for OCR field mapping."""

    def __init__(self):
        prepare()

    def encode(self, sentences, normalize_embeddings=False):
        # The packaged MiniLM graph produces unit vectors for both callers.
        return assessor().encode(sentences)


def checked_model():
    """Exercise the model with source-independent fixtures without saving records."""
    model = load()
    pr = [{'item': 'projector', 'description': 'LCD', 'qty': 7, 'unit': 'pcs'}]
    for name, qty, expected in (
        ('video projection unit', '7', 'Acceptable'),
        ('video projection unit', '5', 'Needs Recanvass'),
        ('video projection unit', '', 'Needs Review'),
        ('television', '7', 'Reject'),
    ):
        po = [{'item': name, 'description': 'LCD', 'qty': qty, 'unit': 'pieces'}]
        result = model.assess(pr, po)
        if result['document']['label'] != expected or any(row['label'] != expected for row in result['rows']):
            raise RuntimeError('PR_PO_MODEL_OUTPUT_CHECK')
    return model


def prepared():
    return _assessor is not None


def assessor():
    if _assessor is None:
        raise RuntimeError('PR_PO_MODEL_NOT_PREPARED')
    return _assessor
