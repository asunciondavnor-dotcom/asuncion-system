"""Prepare the experimental model without changing the official PR–PO predictor."""

import json

from inference_releases import safe_path
from pr_po_preview_model import VERSION, load
from pr_po_release import ROOT


PARAMETERS_SHA256 = '73554185362f6e54a79597ca75bca77ad439afa2f58ac99514d5ec34c9c10905'
NAME = 'MiniLM PR–PO ensemble'
_model = None


def checked_model(encode, *, root=ROOT):
    path = safe_path(root, f'{VERSION}/ensemble.json.gz')
    model = load(path, PARAMETERS_SHA256, encode)
    pr = [{'item': 'projector', 'description': 'LCD', 'qty': '7', 'unit': 'pcs'}]
    for name, quantity, expected in (
        ('video projection unit', '7', 'Acceptable'),
        ('video projection unit', '5', 'Needs Recanvass'),
        ('video projection unit', '', 'Needs Review'),
        ('television', '7', 'Reject'),
    ):
        po = [{'item': name, 'description': 'LCD', 'qty': quantity, 'unit': 'pieces'}]
        result = model.assess(pr, po)
        if result['match']['label'] != expected:
            raise RuntimeError('PR_PO_PREVIEW_OUTPUT_CHECK')
    return model


def prepare(encode):
    global _model
    if _model is None:
        _model = checked_model(encode)
        print(json.dumps({'event': 'pr_po_ensemble_preview_ready', 'model': VERSION, 'cases_passed': 4}), flush=True)


def prepared():
    return _model is not None


def assessor():
    if _model is None:
        raise RuntimeError('PR_PO_PREVIEW_NOT_PREPARED')
    return _model
