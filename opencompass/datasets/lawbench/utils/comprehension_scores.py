import re
from ..utils.rc_f1 import CJRCEvaluator


"""
given a target substring. find its all occurances in the string s
return the starting and ending index of every occurance
"""


def __find_substring_starts(s, target):
    return [(m.start(), m.end()) for m in re.finditer(target, s)]


"""
compute the reading comprehension F1 scores
hyps and refs are lists of hyposisis and reference strings
"""


def compute_rc_f1(hyps, refs):
    scores = 0
    for h, r in zip(hyps, refs):
        scores += CJRCEvaluator.compute_f1(r, h)
    return {'score': scores / len(hyps)}


"""
compute the information extraction F1 scores
hyps and refs are lists of hyposisis and reference strings
entity_types: a set of all possible entity types
"""


def compute_ie_f1(hyps, refs, entity_types):
    assert (len(hyps) == len(refs))
    scores, abstentions = 0, 0
    for h, r in zip(hyps, refs):
        h = __extract_entities_pred(h, entity_types)
        r = __extract_entities_ref(r)
        if r == {}:
            scores += 1 if h == {} else 0
            continue
        if h == {}:
            abstentions += 1
        intersected = [CJRCEvaluator.compute_f1(r[etype], einstance) for etype, einstance in h.items() if etype in r]
        prec = sum(intersected) / len(h) if len(h) > 0 else 0
        rec = sum(intersected) / len(r) if len(r) > 0 else 0
        # print(prec, rec, intersected)
        scores += 2 * prec * rec / (prec + rec + 1e-10)
    return {'score': scores / len(hyps), "anstention_rate": abstentions / len(hyps)}


def __extract_entities_ref(ref):
    outputs = {}
    if ref.strip() == '':
        return outputs
    for seg in ref.split(';'):
        seg = seg.split(':')
        outputs[seg[0]] = seg[1]
    return outputs


"""
extract entity type and instances from the model prediction
pred: string of model prediction
entity_types: a set of all possible entity types
"""


def __extract_entities_pred(pred, entity_types):
    outputs = {}
    for etype in entity_types:
        occurances = __find_substring_starts(pred, etype)
        for start, end in occurances:
            if end >= (len(pred) - 2):
                continue
            if pred[end] == ":" or pred[end] == "：":
                einstance = re.split("\n| ", pred[end + 1:].strip())[0].strip()
                if einstance != '无' and einstance != '未提及':
                    outputs[etype] = einstance
    return outputs
