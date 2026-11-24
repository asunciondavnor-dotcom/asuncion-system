"""Learned PR-PO correspondence and item assessments for isolated verification."""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from pr_po_match_score import match_evidence
from pr_po_model_features import (Line, distinguishing_words, document_alignment,
                                  document_features, line_features, name_features, normalized)
from procurement_comparison_v1 import (NAME, NARRATIVE_FIELDS, norm, number, project,
                                       specification_without_order_header)


LABELS = ('Acceptable','Needs Recanvass','Needs Review','Reject')
ASSESSMENT_VERSION = 'PR_PO_MODEL_ASSESSMENT_V1'


def source_lines(items, side):
    lines = []
    for row in project(items,side,technical=True):
        specifications = []
        for key,value in row['fields'].items():
            if key not in NAME and norm(value)!=norm(row['name']):
                specification = specification_without_order_header(value, row) if norm(key) in NARRATIVE_FIELDS else value
                if specification not in specifications:
                    specifications.append(specification)
        amount = None if row['conflicting_quantity'] else number(row['quantity'])
        quantity = float(amount) if amount is not None else None
        if quantity is not None and not math.isfinite(quantity):
            raise ValueError('MODEL_QUANTITY_RANGE')
        lines.append(Line(row['name'],quantity,str(row['unit'] or '') if not row['conflicting_unit'] else '',
                          ', '.join(specifications)))
    return tuple(lines)


def model_alignment(linker, names):
    """Maximize learned link odds, allowing every item an unmatched alternative."""
    pr_size,po_size,_ = names.shape
    odds = linker.decision_function(names.reshape(-1,names.shape[-1])).reshape(pr_size,po_size)
    padded = np.zeros((pr_size+po_size,pr_size+po_size))
    padded[:pr_size,:po_size] = odds
    return tuple((int(i),int(j)) for i,j in zip(*linear_sum_assignment(padded,maximize=True))
                 if i<pr_size and j<po_size and odds[i,j]>0)


def line_input(pr, po, vectors):
    pair = line_features(pr,po,vectors) if pr is not None and po is not None else np.zeros(35)
    return [float(pr is not None),float(po is not None),*pair]


def predictions(model, inputs):
    labels = model.predict(inputs)
    probabilities = model.predict_proba(inputs)
    for index,label in enumerate(labels):
        label = str(label)
        if label not in LABELS:
            raise ValueError('MODEL_CLASS_INVALID')
        score = float(probabilities[index,list(model.classes_).index(label)])*100
        if not math.isfinite(score) or not 0<=score<=100:
            raise ValueError('MODEL_SCORE_INVALID')
        yield dict(label=label,score=score)


def document_result(assessment):
    """Keep the original confidence while the learned match band supplies the result."""
    document = assessment['document']
    label, confidence = document['label'], document['score']
    if label not in LABELS or type(confidence) not in (int, float) or not 0 <= confidence <= 100:
        raise ValueError('PR_PO_DOCUMENT_RESULT_CONTRACT')
    if 'match' in assessment:
        match = match_evidence(assessment['match'])
        if match is None:
            raise ValueError('PR_PO_MATCH_RESULT_CONTRACT')
        label = match['label']
    return label, confidence


class Assessor:
    def __init__(self, *, document, line, linker, encode, version):
        self.document = document
        self.line = line
        self.linker = linker
        self.encode = encode
        self.version = version

    def _vectors(self, texts, vectors):
        pending = sorted({normalized(value) for value in texts if value} - vectors.keys() - {''})
        if pending:
            vectors.update(zip(pending,self.encode(pending),strict=True))

    def assess(self, pr, po):
        return self.assess_lines(source_lines(pr,'PR'),source_lines(po,'PO'))

    def assess_lines(self, left, right):
        vectors = {}
        self._vectors((value for line in (*left,*right) for value in (line.name,line.specification)),vectors)
        names = np.array([[name_features(pr,po,vectors) for po in right] for pr in left])
        document_pairs = document_alignment(names)
        links = dict(model_alignment(self.linker,names))
        self._vectors((word for i,j in set(document_pairs) | set(links.items())
                       for word in distinguishing_words(left[i],right[j])),vectors)

        inputs = [(i,links.get(i)) for i in range(len(left))]
        matched = set(links.values())
        inputs.extend((None,j) for j in range(len(right)) if j not in matched)
        items = [line_input(left[i] if i is not None else None,right[j] if j is not None else None,vectors) for i,j in inputs]
        rows = [dict(prediction,pr_position=i+1 if i is not None else None,po_position=j+1 if j is not None else None)
                for (i,j),prediction in zip(inputs,predictions(self.line,items),strict=True)]
        document = next(predictions(self.document,[document_features(left,right,document_pairs,vectors)]))
        return dict(schema=ASSESSMENT_VERSION,model_version=self.version,document=document,rows=rows)


def display(assessment, pr, po):
    """Resolve the saved model rows against the same positional source documents."""
    if not isinstance(assessment,dict) or assessment.get('schema')!=ASSESSMENT_VERSION:
        return None
    if not isinstance(assessment.get('rows'),list) or not 1<=len(assessment['rows'])<=len(pr)+len(po):
        return None
    seen_pr,seen_po,rows = set(),set(),[]
    for row in assessment['rows']:
        if not isinstance(row,dict) or row.get('label') not in LABELS:
            return None
        score = row.get('score')
        i,j = row.get('pr_position'),row.get('po_position')
        if (type(score) not in (float,int) or not math.isfinite(score) or not 0<=score<=100
                or (i is None and j is None)):
            return None
        for position,items,seen in ((i,pr,seen_pr),(j,po,seen_po)):
            if position is not None:
                if type(position) is not int or not 1<=position<=len(items) or position in seen:
                    return None
                seen.add(position)
        rows.append(dict(row,pr=pr[i-1] if i is not None else {},po=po[j-1] if j is not None else {}))
    if len(seen_pr)!=len(pr) or len(seen_po)!=len(po):
        return None
    counts = {label:sum(row['label']==label for row in rows) for label in LABELS}
    return dict(rows=rows,counts=counts,model_version=assessment.get('model_version'),
                flagged_rate=None if counts['Needs Review'] else (counts['Needs Recanvass']+counts['Reject'])/len(rows)*100)
