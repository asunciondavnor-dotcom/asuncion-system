"""Read the verified learned linear parameters without importing a training runtime."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit


SCHEMA = 'PR_PO_CALIBRATED_LINEAR_V1'


def _array(data, shape):
    values = np.asarray(data, dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError('PR_PO_LINEAR_PARAMETER_CONTRACT')
    return values


class CalibratedLinearClassifier:
    def __init__(self, data, labels, width):
        self.classes_ = np.asarray(labels)
        if not isinstance(data,list) or len(data)!=3:
            raise ValueError('PR_PO_LINEAR_FOLD_CONTRACT')
        self.folds = []
        for fold in data:
            center = _array(fold['center'],(width,))
            scale = _array(fold['scale'],(width,))
            if np.any(scale<=0):
                raise ValueError('PR_PO_LINEAR_SCALE_CONTRACT')
            self.folds.append((center, scale, _array(fold['coef'],(len(labels),width)),
                _array(fold['intercept'],(len(labels),)),_array(fold['slope'],(len(labels),)),
                _array(fold['offset'],(len(labels),))))

    def predict_proba(self, rows):
        inputs = np.asarray(rows)
        if inputs.dtype not in (np.dtype('float32'),np.dtype('float64')):
            inputs = inputs.astype(np.float64)
        probabilities = np.zeros((len(inputs),len(self.classes_)))
        for center,scale,coef,intercept,slope,offset in self.folds:
            scaled = inputs.copy()
            scaled -= center
            scaled /= scale
            margins = scaled @ coef.T + intercept
            values = expit(-(slope*margins+offset))
            totals = values.sum(axis=1,keepdims=True)
            probabilities += np.divide(values,totals,out=np.full_like(values,1/len(self.classes_)),where=totals!=0)
        return probabilities / len(self.folds)

    def predict(self, rows):
        return self.classes_[np.argmax(self.predict_proba(rows),axis=1)]


class LinearLinker:
    def __init__(self, data):
        self.coef = _array(data['coef'],(4,))
        self.intercept = _array(data['intercept'],(1,))[0]

    def decision_function(self, rows):
        return np.asarray(rows,dtype=np.float64) @ self.coef + self.intercept


def load_parameters(path, expected_sha256):
    """Check the caller's trusted checksum before parsing non-executable weights."""
    from pr_po_model_assessment import LABELS

    with Path(path).open('rb') as handle:
        payload = handle.read(128*1024+1)
    if len(payload)>128*1024 or hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError('PR_PO_LINEAR_ARTIFACT_INTEGRITY')
    data = json.loads(payload)
    if data.get('schema')!=SCHEMA or data.get('labels')!=sorted(LABELS):
        raise ValueError('PR_PO_LINEAR_LABEL_CONTRACT')
    return dict(document=CalibratedLinearClassifier(data['document'],data['labels'],109),
                line=CalibratedLinearClassifier(data['line'],data['labels'],37),linker=LinearLinker(data['linker']))
