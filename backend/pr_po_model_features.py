"""Shared PR-PO feature extraction for training and serving."""

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

import numpy as np
from scipy.optimize import linear_sum_assignment


PACK = re.compile(r'\b(\d+)\s*(?:pcs\.?|pieces|units)\s*(?:per|/)\s*(pack|box|carton)\b', re.I)


BOUND = re.compile(r'\b(minimum|at least|maximum|at most)\s+(\d+(?:\.\d+)?\s*[a-z]+)', re.I)


GENERIC_IDENTITY = {'device','equipment','instrument','item','product','apparatus','accessory','machine','unit'}


METRIC_LENGTH = re.compile(r'\b(\d+(?:\.\d+)?)\s*(mm|cm|m)\b')


@dataclass(frozen=True)
class Line:
    name: str
    quantity: int | None
    unit: str
    specification: str


def normalized(value):
    return ' '.join(str(value or '').lower().split())


def unit(value, *, pack_aliases=False):
    aliases = {'pc': 'piece', 'pcs': 'piece', 'pieces': 'piece', 'reams': 'ream', 'units': 'unit',
               'rolls': 'roll', 'sets': 'set', 'bottles': 'bottle', 'pairs': 'pair', 'packs': 'pack'}
    key = normalized(value).strip('.')
    if pack_aliases and key in ('boxes', 'cartons'):
        return {'boxes': 'box', 'cartons': 'carton'}[key]
    return aliases.get(key, key)


def text_similarity(left, right):
    return SequenceMatcher(None, normalized(left), normalized(right), autojunk=False).ratio() if left and right else 0.


def numeric_tokens(value):
    tokens = set()
    for match in re.finditer(r'\b(\d+(?:\.\d+)?)\s*(tb|gb|mb|mm|cm|m|kg|g|ml|l|dpi|w|v|va|inches|inch|in)?\b',
                             normalized(value)):
        number, measure = float(match[1]), match[2]
        scale, standard = {'tb': (1024, 'gb'), 'cm': (10, 'mm'), 'm': (1000, 'mm'),
                           'kg': (1000, 'g'), 'l': (1000, 'ml'), 'inches': (1, 'in'),
                           'inch': (1, 'in')}.get(measure, (1, measure or ''))
        tokens.add((round(number*scale, 5), standard))
    return tokens


def effective_quantity(line, *, pack_aliases=False):
    specification = normalized(line.specification)
    match = PACK.search(specification)
    resolved_unit = unit(line.unit, pack_aliases=pack_aliases)
    if match and resolved_unit == match[2] and line.quantity:
        return line.quantity * int(match[1]), 'piece', PACK.sub('', specification).strip(' ;,')
    return line.quantity, resolved_unit, specification


def specification_constraints(requested, offered):
    matches = list(BOUND.finditer(requested))
    offered_values = numeric_tokens(offered)
    satisfied, contradicted = 0, 0
    for match in matches:
        for target, tag in numeric_tokens(match[2]):
            observed = [number for number, kind in offered_values if kind == tag]
            if observed:
                satisfies = all(v >= target for v in observed) if match[1] in ('minimum','at least') else all(v <= target for v in observed)
                satisfied += int(satisfies)
                contradicted += int(not satisfies)
    return float(bool(matches)), float(satisfied), float(contradicted)


def cosine(left, right, vectors):
    first, second = vectors.get(normalized(left)), vectors.get(normalized(right))
    return max(0., float(first @ second)) if first is not None and second is not None else 0.


def distinguishing_words(pr,po):
    """Extract word differences for the model to weigh alongside the full descriptions."""
    requested = re.findall(r'[\w]+',normalized(pr.name))
    offered = re.findall(r'[\w]+',normalized(po.name))
    shared = set(requested) & set(offered)
    return ' '.join(w for w in requested if w not in shared), ' '.join(w for w in offered if w not in shared)


def embedding_inputs(cases):
    values = {normalized(value) for _, _, left, right in cases for line in (*left,*right)
              for value in (line.name,line.specification) if value}
    values.update(text for _,_,left,right in cases for pr in left for po in right
                  for text in distinguishing_words(pr,po) if text)
    return values


def line_features(pr, po, vectors, *, pack_aliases=False):
    pq, pu, pr_spec = effective_quantity(pr, pack_aliases=pack_aliases)
    oq, ou, po_spec = effective_quantity(po, pack_aliases=pack_aliases)
    left, right = numeric_tokens(pr_spec), numeric_tokens(po_spec)
    quantities = bool(pq and oq)
    second = set(normalized(po.name).split())
    differing_pr,differing_po = distinguishing_words(pr,po)
    requested_specs = set(re.findall(r'\w+',metric_length_text(pr_spec)))
    offered_specs = set(re.findall(r'\w+',metric_length_text(po_spec)))
    common_specs = len(requested_specs & offered_specs)
    return (
        *name_features(pr,po,vectors),
        cosine(pr.specification, po.specification, vectors), text_similarity(pr.specification, po.specification),
        float(bool(pr.specification)), float(bool(po.specification)),
        len(left & right)/max(1,len(left)), float(bool(left)), float(bool(right)),
        float(bool(left - right)), float(bool(right - left)),
        float(quantities), min(3., abs(pq-oq)/pq) if quantities else 0.,
        float(bool(pu and ou)), float(bool(pu and pu == ou)),
        float(pr_spec==po_spec and bool(pr_spec)),
        float(bool(re.search(r'\b(?:or|subject to)\b',po_spec))),
        *specification_constraints(pr_spec,po_spec),
        float(bool(second & GENERIC_IDENTITY)),
        float(len(second - GENERIC_IDENTITY)),
        float(bool(pr_spec) and metric_length_text(pr_spec)==metric_length_text(po_spec)),
        float(bool(differing_pr)),float(bool(differing_po)),
        cosine(differing_pr,differing_po,vectors),text_similarity(differing_pr,differing_po),
        float(len(differing_pr.split())),float(len(differing_po.split())),
        common_specs/max(1,len(requested_specs)),common_specs/max(1,len(offered_specs)),
        float(len(requested_specs-offered_specs)),float(len(offered_specs-requested_specs)))


def metric_length_text(value):
    return METRIC_LENGTH.sub(lambda m: f"{float(m[1])*{'mm':1,'cm':10,'m':1000}[m[2]]:g} mm", value)


def name_features(pr, po, vectors):
    first, second = set(normalized(pr.name).split()), set(normalized(po.name).split())
    common = len(first & second)
    return (cosine(pr.name,po.name,vectors),text_similarity(pr.name,po.name),
            common/max(1,len(first)),common/max(1,len(second)))


def document_alignment(names):
    matching = .65*names[:, :, 0] + .35*names[:, :, 1]
    return tuple(zip(*linear_sum_assignment(-matching)))


def aligned_features(left, right, vectors):
    if not len(left) or not len(right):
        return (), np.zeros((1, 35), dtype=np.float32)
    names = np.array([[name_features(pr,po,vectors) for po in right] for pr in left])
    positions = document_alignment(names)
    return positions, np.array([line_features(left[i],right[j],vectors) for i,j in positions])


def features(left, right, vectors):
    return aggregate_features(left,right,aligned_features(left,right,vectors)[1])


def document_features(left, right, positions, vectors):
    pairs = np.array([line_features(left[i],right[j],vectors) for i,j in positions]) if positions else np.zeros((1,35))
    return aggregate_features(left,right,pairs)


def aggregate_features(left, right, pairs):
    return np.concatenate(([len(left)-len(right), abs(len(left)-len(right)), len(left), len(right)],
                           pairs.min(axis=0), pairs.mean(axis=0), pairs.max(axis=0))).astype(np.float32)
