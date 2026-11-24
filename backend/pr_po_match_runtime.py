"""Serve the learned match model without training libraries or executable weights."""

import hashlib
import json
from pathlib import Path

from pr_po_identity_features import IdentityEvidence, heads
from pr_po_linear_runtime import CalibratedLinearClassifier, LinearLinker
from pr_po_match_score import FEATURE_WIDTH, MatchScoreModel
from pr_po_model_assessment import ASSESSMENT_VERSION, LABELS, Assessor as BaseAssessor, model_alignment, predictions
from pr_po_model_features import distinguishing_words, document_alignment


SCHEMA = 'PR_PO_LEARNED_MATCH_V1'
MAX_PARAMETERS_BYTES = 2 * 1024 * 1024


def load_parameters(path, expected_sha256):
    with Path(path).open('rb') as stream:
        payload = stream.read(MAX_PARAMETERS_BYTES + 1)
    if len(payload) > MAX_PARAMETERS_BYTES or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('PR_PO_MATCH_ARTIFACT_INTEGRITY')
    return parse_parameters(json.loads(payload))


def parse_parameters(data):
    labels = data.get('labels')
    if data.get('schema') != SCHEMA or labels != list(LABELS):
        raise ValueError('PR_PO_MATCH_ARTIFACT_CONTRACT')
    return dict(document=CalibratedLinearClassifier(data['document'], labels, FEATURE_WIDTH),
                line=CalibratedLinearClassifier(data['line'], labels, 52),
                linker=LinearLinker(data['linker'], width=18), component=LinearLinker(data['component'], width=384),
                match_score=MatchScoreModel(data['match_score']))


class Assessor(BaseAssessor):
    def __init__(self, *, component, match_score, **settings):
        super().__init__(**settings)
        self.component = component
        self.match_score = match_score

    def assess_lines(self, left, right):
        evidence, pairs = self.prepared_inputs(left, right)
        return self.assess_prepared(left, right, evidence, pairs=pairs)

    def prepared_inputs(self, left, right):
        vectors = {}
        terms = (term for item in (*left, *right) for term in (item.name, item.specification, *heads(item.name)))
        self._vectors(terms, vectors)
        evidence = IdentityEvidence(self.component, vectors, verified_quantities=True)
        return evidence, evidence.paired_data(left, right)

    def document_probabilities(self, left, right, evidence, pairs):
        features = evidence.document_input(left, right, pairs)
        return features, self.document.predict_proba([features])[0]

    def assess_document(self, left, right, evidence, pairs, rows):
        features, probabilities = self.document_probabilities(left, right, evidence, pairs)
        return probabilities, self.match_score.assess(features, probabilities)

    def assess_prepared(self, left, right, evidence, *, pairs=None):
        pairs = evidence.paired_data(left, right) if pairs is None else pairs
        document_pairs = document_alignment(pairs)
        links = dict(model_alignment(self.linker, pairs))
        self._vectors((value for i, j in set(document_pairs) | set(links.items())
                       for value in distinguishing_words(left[i], right[j])), evidence.vectors)
        indices = [(i, links.get(i)) for i in range(len(left))]
        matched = set(links.values())
        indices.extend((None, j) for j in range(len(right)) if j not in matched)
        inputs = [evidence.line_input(left[i] if i is not None else None, right[j] if j is not None else None)
                  for i, j in indices]
        rows = [dict(prediction, pr_position=i + 1 if i is not None else None, po_position=j + 1 if j is not None else None)
                for (i, j), prediction in zip(indices, predictions(self.line, inputs), strict=True)]
        probabilities, match = self.assess_document(left, right, evidence, pairs, rows)
        winner = int(probabilities.argmax())
        document = dict(label=str(self.document.classes_[winner]), score=float(probabilities[winner]) * 100)
        return dict(schema=ASSESSMENT_VERSION, model_version=self.version, document=document, match=match, rows=rows)
