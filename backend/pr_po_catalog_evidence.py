"""Serve the fitted product-type probabilities for equivalent wording."""

import re

import numpy as np
from scipy.special import softmax

from pr_po_catalog_text_features import CHAR_WIDTH, characters, product_phrase
from pr_po_model_features import effective_quantity, metric_length_text, normalized


SPECIFICATION_DELIMITER = re.compile(r'[,;\n]+')
AMBIGUOUS = re.compile(r'\b(?:or|subject to)\b')


def specification_clauses(value):
    formatted = metric_length_text(value)
    return sorted(fragment for part in SPECIFICATION_DELIMITER.split(formatted)
                  if (fragment := normalized(part).strip('. ')))


def supporting_facts(pr, po):
    first, u1, s1 = effective_quantity(pr, pack_aliases=True)
    second, u2, s2 = effective_quantity(po, pack_aliases=True)
    return (bool(first and second and u1 and u2 and s1 and s2) and first == second and u1 == u2
            and not (AMBIGUOUS.search(s1) or AMBIGUOUS.search(s2))
            and specification_clauses(s1) == specification_clauses(s2))


class CatalogEvidence:
    def __init__(self, artifact):
        self.weights = np.asarray(artifact['weights'], dtype=float)
        self.bias = np.asarray(artifact['bias'], dtype=float)
        self.text_weight = float(artifact['text_weight'])
        self.cutoff = float(artifact['cutoff'])
        self.exponent = float(artifact['exponent'])
        width = 384 + (CHAR_WIDTH if self.text_weight else 0)
        if (self.bias.ndim != 1 or not 2 <= len(self.bias) <= 64
                or self.weights.shape != (len(self.bias), width)
                or not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all()
                or not 0 <= self.text_weight <= 4 or not 0 < self.cutoff < 1 or not 0 < self.exponent <= 1):
            raise ValueError('PR_PO_CATALOG_EVIDENCE_PARAMETERS')

    def confidence(self, pairs, vectors, add_vectors):
        names = [tuple(normalized(item.name) for item in pair) for pair in pairs]
        articles = [tuple(product_phrase(name) for name in pair) for pair in names]
        terms = list(dict.fromkeys(name for group in (names, articles) for pair in group for name in pair))
        add_vectors(terms, vectors)
        if any(term not in vectors for term in terms):
            return 0.
        feature = np.asarray([vectors[term] for term in terms])
        if self.text_weight:
            feature = np.c_[feature, np.asarray([characters(term) for term in terms]) * self.text_weight]
        probabilities = softmax(feature @ self.weights.T + self.bias, axis=1)[:, :-1]
        estimated = dict(zip(terms, probabilities, strict=True))
        return min(float(estimated[a] @ estimated[b]) for group in (names, articles) for a, b in group)
