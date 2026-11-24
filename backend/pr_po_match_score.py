"""Frozen regression model for the PR–PO match scale and its learned bands."""

from collections.abc import Mapping
import math

import numpy as np


MATCH_LABELS = ('Reject', 'Needs Recanvass', 'Needs Review', 'Acceptable')
LEGACY_MATCH_LABELS = ('Reject', 'Needs Review', 'Needs Recanvass', 'Acceptable')
CLASSIFIER_LABELS = tuple(sorted(MATCH_LABELS))
ACCEPTABLE_MINIMUM = 90.0
FEATURE_WIDTH = 154
FEATURE_PRECISION = 3


def agreement_features(features):
    """Keep tree splits stable across minor encoder rounding differences."""
    return np.round(np.asarray(features, dtype=np.float64), decimals=FEATURE_PRECISION)


def scale_scores(raw_scores, acceptable_cut):
    """Map the learned Acceptable boundary to 90 on one common score scale."""
    scores = np.clip(raw_scores, 0, 3)
    return np.round(np.interp(scores, [0, acceptable_cut, 3], [0, ACCEPTABLE_MINIMUM, 100]), 2)


def band_label(score, boundaries, labels=MATCH_LABELS):
    return labels[sum(score >= boundary for boundary in boundaries)]


def band_order(value):
    if not isinstance(value, (list, tuple)):
        return None
    labels = tuple(value)
    return labels if labels in (MATCH_LABELS, LEGACY_MATCH_LABELS) else None


def match_evidence(value):
    """Validate an optional historical match result without recomputing it."""
    if not isinstance(value, Mapping):
        return None
    score = value.get('score')
    bands = value.get('boundaries')
    labels = band_order(value.get('band_labels', LEGACY_MATCH_LABELS))
    if (type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 100
            or not isinstance(bands, list) or len(bands) != 3
            or any(type(cut) not in (float, int) or not math.isfinite(cut) for cut in bands)
            or not 0 < bands[0] < bands[1] < bands[2] == ACCEPTABLE_MINIMUM
            or labels is None or value.get('label') != band_label(score, bands, labels)):
        return None
    return dict(label=value['label'], score=float(score), boundaries=list(bands), band_labels=list(labels))


def agreement_estimate(regression, probabilities, temperature, regression_weight, labels=MATCH_LABELS):
    """Estimate agreement on a single ordinal scale using all four class probabilities."""
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.shape[-1:] != (4,) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=-1), 1)):
        raise ValueError('PR_PO_MATCH_PROBABILITY_CONTRACT')
    ordinal_columns = [CLASSIFIER_LABELS.index(label) for label in labels]
    sharpened = probabilities[..., ordinal_columns] ** temperature
    ordinal = (sharpened @ np.arange(4)) / sharpened.sum(axis=-1)
    return regression_weight * regression + (1 - regression_weight) * ordinal


class MatchScoreModel:
    def __init__(self, artifact):
        self.baseline = float(artifact['baseline'])
        self.labels = band_order(artifact.get('band_labels', LEGACY_MATCH_LABELS))
        if self.labels is None:
            raise ValueError('PR_PO_MATCH_LABEL_ORDER_CONTRACT')
        self.raw_boundaries = np.asarray(artifact['raw_boundaries'], dtype=float)
        if (not math.isfinite(self.baseline) or self.raw_boundaries.shape != (3,)
                or not 0 < self.raw_boundaries[0] < self.raw_boundaries[1] < self.raw_boundaries[2] < 3):
            raise ValueError('PR_PO_MATCH_SCALE_CONTRACT')
        self.boundaries = scale_scores(self.raw_boundaries, self.raw_boundaries[-1]).tolist()
        if not 0 < self.boundaries[0] < self.boundaries[1] < self.boundaries[2] == ACCEPTABLE_MINIMUM:
            raise ValueError('PR_PO_MATCH_BAND_CONTRACT')
        self.temperature = float(artifact.get('temperature', 1))
        self.regression_weight = float(artifact.get('regression_weight', 1))
        if not 1 <= self.temperature <= 16 or not 0 < self.regression_weight <= 1:
            raise ValueError('PR_PO_MATCH_ESTIMATION_CONTRACT')
        trees = artifact['trees']
        if not isinstance(trees, list) or not 1 <= len(trees) <= 256:
            raise ValueError('PR_PO_MATCH_TREE_CONTRACT')
        self.trees = tuple(self._tree(tree) for tree in trees)

    @staticmethod
    def _tree(tree):
        if not isinstance(tree, list) or not 1 <= len(tree) <= 255:
            raise ValueError('PR_PO_MATCH_TREE_CONTRACT')
        for index, node in enumerate(tree):
            if not isinstance(node, list) or len(node) not in (1, 4):
                raise ValueError('PR_PO_MATCH_NODE_CONTRACT')
            if len(node) == 1:
                if type(node[0]) not in (float, int) or not math.isfinite(node[0]):
                    raise ValueError('PR_PO_MATCH_NODE_CONTRACT')
                continue
            feature, threshold, left, right = node
            if (type(feature) is not int or not 0 <= feature < FEATURE_WIDTH
                    or type(threshold) not in (float, int) or not math.isfinite(threshold)
                    or type(left) is not int or type(right) is not int or not index < left < right < len(tree)):
                raise ValueError('PR_PO_MATCH_NODE_CONTRACT')
        return tuple(map(tuple, tree))

    def predict_raw(self, features):
        inputs = agreement_features(features).astype(np.float32).astype(np.float64)
        if inputs.shape != (FEATURE_WIDTH,) or not np.isfinite(inputs).all():
            raise ValueError('PR_PO_MATCH_FEATURE_CONTRACT')
        score = self.baseline
        for tree in self.trees:
            node = tree[0]
            while len(node) == 4:
                feature, threshold, left, right = node
                node = tree[left if inputs[feature] <= threshold else right]
            score += node[0]
        return score

    def assess(self, features, probabilities):
        agreement = agreement_estimate(self.predict_raw(features), probabilities, self.temperature, self.regression_weight, self.labels)
        score = float(scale_scores(agreement, self.raw_boundaries[-1]))
        return dict(label=band_label(score, self.boundaries, self.labels), score=score,
                    boundaries=self.boundaries.copy(), band_labels=list(self.labels))
