"""Feature extraction shared with the fitted verified-quantities candidate."""

import re

import numpy as np

from pr_po_model_features import (aggregate_features, cosine, document_alignment, effective_quantity,
                                  line_features, name_features, normalized, text_similarity)


def tokens(value):
    return re.findall(r'[a-z]+', normalized(value))


def heads(name):
    words = tokens(name)
    position = next((i for i, word in enumerate(words) if i>0 and word in ('for', 'with')), len(words))
    lead = words[:position]
    return ' '.join(lead[-1:]), ' '.join(lead[-2:]), ' '.join(words[-2:])


class IdentityEvidence:
    def __init__(self, component_model, vectors, *, verified_quantities=False):
        self.component_model = component_model
        self.vectors = vectors
        self.cache = {}
        self.verified_quantities = verified_quantities

    def part_score(self, name):
        name = normalized(name)
        if name not in self.cache:
            relevant = (name, *heads(name))
            inputs = [self.vectors.get(fragment, np.zeros(384, dtype=np.float32)) for fragment in relevant]
            magnitudes = self.component_model.decision_function(inputs)
            self.cache[name] = tuple(float(x) for x in (magnitudes[0], magnitudes[1:].max()))
        return self.cache[name]

    def pair_vector(self, pr, po):
        left, right = heads(pr.name), heads(po.name)
        pr_full, pr_head = self.part_score(pr.name)
        po_full, po_head = self.part_score(po.name)
        left_words, right_words = set(tokens(pr.name)), set(tokens(po.name))
        shared = left_words & right_words
        return (*name_features(pr, po, self.vectors),
                *(cosine(a, b, self.vectors) for a, b in zip(left, right, strict=True)),
                *(text_similarity(a, b) for a, b in zip(left, right, strict=True)),
                pr_full, pr_head, po_full, po_head, po_full-pr_full, po_head-pr_head,
                len(left_words-shared), len(right_words-shared))

    def line_vector(self, pr, po):
        base = line_features(pr, po, self.vectors, pack_aliases=self.verified_quantities)
        vector = (*base, *self.pair_vector(pr, po)[4:])
        if self.verified_quantities:
            requested, pr_unit, _ = effective_quantity(pr, pack_aliases=True)
            offered, po_unit, _ = effective_quantity(po, pack_aliases=True)
            change = bool(requested and offered and pr_unit and pr_unit == po_unit and requested != offered)
            vector += (float(change),)
        return vector

    def line_input(self, pr, po):
        pair = self.line_vector(pr, po) if pr is not None and po is not None else np.zeros(49+self.verified_quantities)
        return [float(pr is not None), float(po is not None), *pair]

    def paired_data(self, left, right):
        return np.array([[self.pair_vector(pr, po) for po in right] for pr in left])

    def document_input(self, left, right, paired=None):
        paired = self.paired_data(left, right) if paired is None else paired
        alignment = document_alignment(paired)
        return aggregate_features(left, right, np.array([self.line_vector(left[i], right[j]) for i, j in alignment]))
