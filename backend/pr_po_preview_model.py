"""Load and serve the checksum-pinned PR–PO ensemble for read-only previews."""

import gzip
import hashlib
import json

import numpy as np
from scipy.special import expit

from pr_po_linear_runtime import CalibratedLinearClassifier
from pr_po_match_runtime import Assessor, parse_parameters
from pr_po_match_score import CLASSIFIER_LABELS, FEATURE_WIDTH
from pr_po_model_features import document_alignment, normalized, text_similarity
from pr_po_substitution import ACCEPTABLE, QualifierEvidence, SubstitutionEvidence


VERSION = 'asuncion-pr-po-ensemble-verified-v4'
PACKAGE_SCHEMA = 'PR_PO_ENSEMBLE_VERIFIED_V4'
MAX_COMPRESSED_BYTES = 2 * 1024 * 1024
MAX_JSON_BYTES = 12 * 1024 * 1024


def identity_vector(evidence, left, right):
    first, second = normalized(left.name), normalized(right.name)
    a = ''.join(c for c in first if c.isalnum())
    b = ''.join(c for c in second if c.isalnum())
    values = [text_similarity(a, b), text_similarity(' '.join(sorted(first.split())), ' '.join(sorted(second.split())))]
    for length in (2, 3):
        aa = {a[i:i + length] for i in range(len(a) - length + 1)}
        bb = {b[i:i + length] for i in range(len(b) - length + 1)}
        values.extend((len(aa & bb) / max(1, len(aa | bb)), len(aa & bb) / max(1, min(len(aa), len(bb)))))
    return np.asarray((*evidence.pair_vector(left, right), *values), dtype=np.float32)


class FrozenForest:
    def __init__(self, data, *, width, classes):
        self.roots = np.asarray(data['roots'], dtype=np.int32)
        self.features = np.asarray(data['features'], dtype=np.int16)
        self.thresholds = np.asarray(data['thresholds'], dtype=np.float64)
        self.left = np.asarray(data['left'], dtype=np.int32)
        self.right = np.asarray(data['right'], dtype=np.int32)
        self.probabilities = np.asarray(data['probabilities'], dtype=np.float64)
        count = len(self.features)
        if (not 0 < count <= 65536 or self.features.shape != (count,) or self.thresholds.shape != (count,)
                or self.left.shape != (count,) or self.right.shape != (count,)
                or self.probabilities.shape != (count, classes) or self.roots.ndim != 1 or not 1 <= len(self.roots) <= 256
                or self.roots[0] != 0 or self.roots[-1] >= count or not np.all(np.diff(self.roots) > 0)
                or not np.isfinite(self.thresholds).all() or not np.isfinite(self.probabilities).all()
                or not np.all((self.features >= -1) & (self.features < width))):
            raise ValueError('PR_PO_PREVIEW_FOREST_CONTRACT')
        nodes = np.arange(count)
        splits = self.features >= 0
        limits = np.r_[self.roots[1:], count][np.searchsorted(self.roots, nodes, side='right') - 1]
        if (np.any(self.left[splits] <= nodes[splits]) or np.any(self.right[splits] <= nodes[splits])
                or np.any(self.left[splits] >= limits[splits]) or np.any(self.right[splits] >= limits[splits])
                or not np.all(self.probabilities[~splits] >= 0)
                or not np.allclose(self.probabilities[~splits].sum(axis=1), 1)):
            raise ValueError('PR_PO_PREVIEW_FOREST_TOPOLOGY')
        self.left[~splits] = nodes[~splits]
        self.right[~splits] = nodes[~splits]

    def predict_proba(self, rows):
        inputs = np.asarray(rows, dtype=np.float32).astype(np.float64)
        nodes = np.broadcast_to(self.roots, (len(inputs), len(self.roots)))
        index = np.arange(len(inputs))[:, None]
        while True:
            features = self.features[nodes]
            if np.all(features == -1):
                return self.probabilities[nodes].mean(axis=1)
            nodes = np.where(inputs[index, features] <= self.thresholds[nodes], self.left[nodes], self.right[nodes])


class ItemIdentity:
    def __init__(self, data):
        self.linear = tuple((np.asarray(row['center']), np.asarray(row['scale']), np.asarray(row['coef']), float(row['bias']))
                            for row in data['linear'])
        self.forests = tuple(FrozenForest(row, width=24, classes=2) for row in data['forest'])
        if len(self.linear) != 4 or len(self.forests) != 4 or any(
            center.shape != (24,) or scale.shape != (24,) or coef.shape != (24,) or not np.all(scale > 0)
            for center, scale, coef, _ in self.linear
        ):
            raise ValueError('PR_PO_PREVIEW_IDENTITY_CONTRACT')

    def signals(self, rows):
        linear = []
        for center, scale, coef, bias in self.linear:
            standardized = rows.copy()
            standardized -= center
            standardized /= scale
            linear.append(expit(standardized @ coef + bias))
        ensembles = [np.mean(linear, axis=0), np.mean([forest.predict_proba(rows)[:, 1] for forest in self.forests], axis=0)]
        return np.asarray([value for scores in ensembles for value in (min(scores), np.mean(scores), max(scores))], dtype=np.float32)


class EnsemblePreview(Assessor):
    def __init__(self, *, identity, forest, alternate, shares, substitution, wording, qualifiers, **settings):
        super().__init__(**settings)
        self.identity = identity
        self.substitution = substitution
        self.qualifiers = QualifierEvidence(qualifiers)
        self.wording = FrozenForest(wording['forest'], width=160, classes=2)
        self.wording_cutoff = float(wording['cutoff'])
        self.wording_exponent = float(wording['exponent'])
        if not 0 < self.wording_cutoff < 1 or not 0 < self.wording_exponent <= 1:
            raise ValueError('PR_PO_WORDING_PARAMETERS')
        self.forest = forest
        self.alternate = alternate
        self.shares = np.asarray(shares)
        if self.shares.shape != (3,) or not np.all(self.shares >= 0) or not np.isclose(self.shares.sum(), 1):
            raise ValueError('PR_PO_PREVIEW_BLEND_CONTRACT')

    def document_inputs(self, left, right, evidence, pairs):
        base = evidence.document_input(left, right, pairs)
        positions = document_alignment(pairs)
        identity = np.asarray([identity_vector(evidence, left[i], right[j]) for i, j in positions])
        augmented = np.concatenate((base, self.identity.signals(identity)))
        return base, augmented

    def document_probabilities(self, left, right, evidence, pairs):
        base, augmented = self.document_inputs(left, right, evidence, pairs)
        return base, self.blend(base[None, :], augmented[None, :])[0]

    def assess_document(self, left, right, evidence, pairs, rows):
        base, augmented = self.document_inputs(left, right, evidence, pairs)
        probabilities = self.blend(base[None, :], augmented[None, :])[0]
        result = self.substitution.assess(self.match_score, base, augmented, probabilities,
                                          left, right, pairs, evidence, rows, identity_vector)
        chosen = probabilities
        support = float(self.wording.predict_proba(augmented[None, :])[0, 1])
        if support >= self.wording_cutoff and result['label'] != 'Acceptable':
            share = support ** self.wording_exponent
            adjusted = probabilities * (1 - share)
            adjusted[ACCEPTABLE] += share
            rescored = self.match_score.assess(base, adjusted)
            if rescored['score'] > result['score']:
                chosen, result = adjusted, rescored
        if result['label'] == 'Acceptable' and len(left) == len(right) == 1:
            adjusted = self.qualifiers.assess(probabilities, left[0], right[0], evidence.vectors)
            if adjusted is not probabilities:
                assessed = self.match_score.assess(base, adjusted)
                if assessed['score'] < result['score']:
                    return adjusted, assessed
        return chosen, result

    def blend(self, base, augmented):
        staged, forest, alternate = self.shares
        return (staged * self.document.predict_proba(base) + forest * self.forest.predict_proba(augmented)
                + alternate * self.alternate.predict_proba(base))


def load(path, expected_sha256, encode):
    with path.open('rb') as stream:
        if path.stat().st_size > MAX_COMPRESSED_BYTES or hashlib.file_digest(stream, 'sha256').hexdigest() != expected_sha256:
            raise ValueError('PR_PO_PREVIEW_INTEGRITY')
        stream.seek(0)
        with gzip.GzipFile(fileobj=stream) as compressed:
            payload = compressed.read(MAX_JSON_BYTES + 1)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError('PR_PO_PREVIEW_PACKAGE_SIZE')
    data = json.loads(payload)
    if data['schema'] != PACKAGE_SCHEMA or data['version'] != VERSION:
        raise ValueError('PR_PO_PREVIEW_PACKAGE_CONTRACT')
    return EnsemblePreview(**parse_parameters(data['base']), encode=encode, version=VERSION,
                           identity=ItemIdentity(data['identity']), forest=FrozenForest(data['forest'], width=160, classes=4),
                           alternate=CalibratedLinearClassifier(data['alternate'], CLASSIFIER_LABELS, FEATURE_WIDTH),
                           shares=data['shares'], substitution=SubstitutionEvidence(data['substitution']), wording=data['wording'], qualifiers=data['qualifiers'])
