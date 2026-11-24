"""Prepare the experimental model without changing the official PR–PO predictor."""

import json

from inference_releases import safe_path
from pr_po_preview_model import VERSION, load
from pr_po_release import ROOT


PARAMETERS_SHA256 = 'c4fc60f2b93b0f1778883296fa20682fe056638203f4e4bc752b01183799b145'
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


def assessor():
    if _model is None:
        raise RuntimeError('PR_PO_PREVIEW_NOT_PREPARED')
    return _model
