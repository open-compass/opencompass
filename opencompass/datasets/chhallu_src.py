"""CHHallu-Src: Chinese historical-fact hallucination & source-attribution benchmark.

Contributed dataset for OpenCompass. Closed-book; answer keys 100% derived from a
public-domain primary-source corpus. Scoring mirrors the standalone reference impl
(https://huggingface.co/datasets/lizhuojun/chhallu-src-v1 → eval_score.py); if the two
disagree, the standalone scorer is authoritative.

Place at: opencompass/datasets/chhallu_src.py
"""
import json
import os
import re

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

HF_ID = "lizhuojun/chhallu-src-v1"

BOOK_WINDOW = {
    "史记": (-2100, -90), "汉书": (-206, 23), "后汉书": (25, 220), "三国志": (184, 280),
    "左传": (-722, -468), "资治通鉴": (-403, 959), "吕氏春秋": (-2100, -239),
    "论语": (-551, -479), "孟子": (-372, -289),
}
BOOK_ALIASES = [
    ("后汉书", ["后汉书", "後漢書"]), ("三国志", ["三国志", "三國志"]),
    ("资治通鉴", ["资治通鉴", "資治通鑑", "通鉴", "通鑑"]), ("吕氏春秋", ["吕氏春秋", "呂氏春秋"]),
    ("汉书", ["汉书", "漢書", "前汉书", "前漢書"]), ("史记", ["史记", "史記", "太史公书", "太史公書"]),
    ("左传", ["左传", "左傳", "春秋左传", "左氏春秋", "左氏传", "左氏傳"]),
    ("论语", ["论语", "論語"]), ("孟子", ["孟子"]),
]
_PUNCT = re.compile(r"[《》〈〉「」『』\s、,，。.]+")
_ORD = re.compile(r"第[一二三四五六七八九十百千0-9]+")

_TAIL = {
    "attribution": '\n\n选项：\n{opts}\n\n只依据史实判断，以 JSON 返回正确选项字母：{{"choice": "A"}}',
    "crossbook": '\n\n以 JSON 返回你判断确有记载的书名数组（只从候选9部中选，书名不带书名号）：{"books": ["史记", "汉书"]}',
    "sourcing": '\n\n以 JSON 返回篇名（不含"第几"序号亦可）：{"chapter": "淮阴侯列传"}',
}


def _coerce(row):
    """questions.jsonl ships options/answer/grounding as JSON strings (HF-viewer flat form)."""
    for k in ("options", "answer", "grounding"):
        v = row.get(k)
        if isinstance(v, str):
            try:
                row[k] = json.loads(v)
            except (ValueError, TypeError):
                pass
    return row


def _render(row):
    row = _coerce(row)
    t = row["type"]
    if t == "attribution":
        opts = "\n".join(f"{k}. {v}" for k, v in row["options"].items())
        return row["stem"] + _TAIL[t].format(opts=opts)
    return row["stem"] + _TAIL[t]


def _norm_book(s):
    s = _PUNCT.sub("", s or "")
    if not s:
        return None
    for canon, aliases in BOOK_ALIASES:
        if s in aliases:
            return canon
    for canon, aliases in BOOK_ALIASES:
        if any(a in s for a in aliases):
            return canon
    return None


def _norm_chapter(s):
    return _ORD.sub("", _PUNCT.sub("", s or "")).strip()


def _parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _score_one(row, pred_text):
    row = _coerce(row)
    parsed = _parse_json(pred_text)
    t = row["type"]
    if t == "attribution":
        ch = str((parsed or {}).get("choice", "")).strip().upper()[:1]
        return ch == row["answer"]["choice"]
    if t == "sourcing":
        pred = (parsed or {}).get("chapter") or (parsed or {}).get("chapters") or ""
        preds = pred if isinstance(pred, list) else [pred]
        pnorms = [_norm_chapter(str(p)) for p in preds]
        for g in row["answer"]["chapters"]:
            gn = _norm_chapter(g)
            if any(pn and gn and (gn in pn or pn in gn) and min(len(gn), len(pn)) >= 2 for pn in pnorms):
                return True
        return False
    # crossbook: recall==1 and no era-impossible over-prediction
    gold = {_norm_book(b) for b in row["answer"]["books"]} - {None}
    pred_raw = (parsed or {}).get("books", []) if isinstance(parsed, dict) else []
    pred = {_norm_book(b) for b in (pred_raw if isinstance(pred_raw, list) else [])} - {None}
    if gold - pred:
        return False
    ew = (row.get("grounding") or {}).get("era_window")
    if ew:
        lo, hi = ew
        for b in pred - gold:
            bw = BOOK_WINDOW.get(b)
            if bw and (bw[1] < lo or bw[0] > hi):
                return False
    return True


@LOAD_DATASET.register_module()
class CHHalluSrcDataset(BaseDataset):

    @staticmethod
    def load(path: str = HF_ID, **kwargs):
        # DATASET_SOURCE: HF (default) / ModelScope / local jsonl path
        src = os.environ.get("DATASET_SOURCE", "").lower()
        rows = []
        if src == "modelscope":
            from modelscope import MsDataset
            ds = MsDataset.load(path, split="test")
            rows = [dict(r) for r in ds]
        elif os.path.isfile(path):                      # local questions.jsonl
            rows = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
        else:                                           # HuggingFace hub
            from datasets import load_dataset
            ds = load_dataset(path, split="test")
            rows = [dict(r) for r in ds]
        return Dataset.from_list([{"prompt": _render(dict(r)), "row": _coerce(dict(r))} for r in rows])


@ICL_EVALUATORS.register_module()
class CHHalluSrcEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {"error": "preds and refs length mismatch"}
        correct = sum(_score_one(ref, pred) for pred, ref in zip(predictions, references))
        return {"accuracy": 100.0 * correct / len(predictions) if predictions else 0.0}
