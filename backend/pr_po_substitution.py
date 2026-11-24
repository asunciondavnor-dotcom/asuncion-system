"""Combine the trained item-identity and item-line checks for document matching."""

import numpy as np
from scipy.special import expit

from pr_po_match_score import CLASSIFIER_LABELS
from pr_po_model_features import Line, distinguishing_words, document_alignment, normalized


ACCEPTABLE = CLASSIFIER_LABELS.index('Acceptable')
REJECT = CLASSIFIER_LABELS.index('Reject')


class EvidenceHead:
    def __init__(self, data, width):
        self.center = np.asarray(data['center'], dtype=np.float64)
        self.scale = np.asarray(data['scale'], dtype=np.float64)
        self.coefficients = np.asarray(data['coef'], dtype=np.float64)
        self.bias = float(data['bias'])
        self.cutoff = float(data['cutoff'])
        if (any(values.shape != (width,) or not np.isfinite(values).all()
                for values in (self.center, self.scale, self.coefficients))
                or np.any(self.scale <= 0) or not np.isfinite(self.bias)
                or not 0 < self.cutoff < 1):
            raise ValueError('PR_PO_SUBSTITUTION_PARAMETERS')

    def risks(self, inputs):
        standardized = np.asarray(inputs).copy()
        standardized -= self.center
        standardized /= self.scale
        return expit(standardized @ self.coefficients + self.bias)


class RejectionHead(EvidenceHead):
    def __init__(self, data, width):
        super().__init__(data, width)
        self.temperature = float(data['temperature'])
        if not 0 < self.temperature <= 100:
            raise ValueError('PR_PO_SUBSTITUTION_TEMPERATURE')

    def apply(self, probabilities, risk):
        updated = probabilities.copy()
        moved = updated[ACCEPTABLE] * expit((risk - self.cutoff) * self.temperature)
        updated[ACCEPTABLE] -= moved
        updated[REJECT] += moved
        return updated


class SubstitutionEvidence:
    def __init__(self, data):
        self.product = RejectionHead(data['product'], 792)
        self.lines = RejectionHead(data['lines'], 13)
        self.equivalence = EvidenceHead(data['equivalence'], 160)
        self.equivalence_exponent = float(data['equivalence']['exponent'])
        if not 0 < self.equivalence_exponent <= 1:
            raise ValueError('PR_PO_EQUIVALENCE_EXPONENT')

    def assess(self, match, base, augmented, probabilities, left, right, pairs, evidence, rows, identity_vector):
        neutral = np.zeros(384, dtype=np.float32)
        names = [sorted((normalized(left[i].name), normalized(right[j].name))) for i, j in document_alignment(pairs)]
        vectors = []
        for first, second in names:
            a, b = Line(first, 1, '', ''), Line(second, 1, '', '')
            diff_a, diff_b = distinguishing_words(a, b)
            first_vec, second_vec = (evidence.vectors.get(value, neutral) for value in (first, second))
            first_diff, second_diff = (evidence.vectors.get(value, neutral) for value in (diff_a, diff_b))
            vectors.append(np.concatenate((identity_vector(evidence, a, b), np.abs(first_vec - second_vec),
                                           np.abs(first_diff - second_diff))))
        product_risk = self.product.risks(np.asarray(vectors)).max()
        linked, unlinked = 0., 0.
        for row in rows:
            if row['label'] == 'Reject':
                if row['pr_position'] is None or row['po_position'] is None:
                    unlinked = max(unlinked, row['score'] / 100)
                else:
                    linked = max(linked, row['score'] / 100)
        line_inputs = np.r_[linked, unlinked, probabilities, augmented[[0, 4, 5, 6, 7, 154, 157]]]
        line_risk = self.lines.risks(line_inputs)
        checks = (match.assess(base, head.apply(probabilities, risk))
                  for head, risk in ((self.product, product_risk), (self.lines, line_risk)))
        result = min(checks, key=lambda check: check['score'])

        support = self.equivalence.risks(augmented)
        if support >= self.equivalence.cutoff:
            share = support ** self.equivalence_exponent
            revised = probabilities * (1 - share)
            revised[ACCEPTABLE] += share
            equivalent = match.assess(base, revised)
            if equivalent['score'] > result['score']:
                return equivalent
        return result
