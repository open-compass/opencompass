"""Helium Trades open benchmarks for OpenCompass."""

from __future__ import annotations

import json
import re
from typing import Any

from datasets import load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

MCQ_TASKS = {
    "moneyness_logic",
    "prob_itm",
    "term_structure_mcq",
    "relative_iv",
    "relative_price",
    "time_value_sign",
    "delta_bounds_mcq",
    "put_call_parity",
}

IV_TASKS = {"implied_volatility", "implied_volatility_prior", "implied_volatility_inversion"}

REGIME_IV_TOL = {"high_vol": 18.0, "moderate": 16.0, "low_vol": 14.0, "canary": 20.0}
DEFAULT_IV_TOL = 18.0
REGIME_DELTA_TOL = {"high_vol": 0.22, "moderate": 0.20, "low_vol": 0.18, "canary": 0.25}
DEFAULT_DELTA_TOL = 0.22


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def _parse_letter(text: str, valid: str = "ABCDE") -> str | None:
    line = _first_line(text).upper()
    if re.fullmatch(rf"[{valid}]", line):
        return line
    m = re.match(rf"^([{valid}])[\).\s]", line)
    if m:
        return m.group(1)
    m = re.search(rf"(?<![A-Z])([{valid}])(?![A-Z])", line)
    return m.group(1) if m else None


def _parse_number(text: str) -> float | None:
    line = _first_line(text).replace(",", "").replace("%", "").replace("$", "")
    m = re.search(r"-?\d+(?:\.\d+)?", line)
    return float(m.group()) if m else None


def _normalize_iv_percent(value: float) -> float:
    if 0 < value <= 3.0:
        return value * 100.0
    return value


def _regime_tol(regime: str, kind: str) -> float:
    if kind == "iv":
        return REGIME_IV_TOL.get(regime, DEFAULT_IV_TOL)
    return REGIME_DELTA_TOL.get(regime, DEFAULT_DELTA_TOL)


def score_market_resolution_item(item: dict, response: str) -> float:
    task = item["task"]
    gt = item["ground_truth"]

    if task in MCQ_TASKS:
        pred = _parse_letter(response, "ABC")
        return 1.0 if pred == gt.get("answer") else 0.0

    if task == "canary_watermark":
        first = _first_line(response).upper()
        return 1.0 if first == "UNKNOWN" else 0.0

    if task == "intrinsic_value":
        pred = _parse_number(response)
        true = gt.get("intrinsic_value", 0)
        if pred is None:
            return 0.0
        err = abs(pred - true)
        if true > 0:
            return max(0.0, 1.0 - err / max(true * 0.5, 0.5))
        return 1.0 if err < 0.05 else 0.0

    if task in IV_TASKS:
        pred = _parse_number(response)
        true = gt.get("iv_percent")
        tol = _regime_tol(item.get("regime", ""), "iv")
        if pred is None or true is None:
            return 0.0
        pred = _normalize_iv_percent(pred)
        return max(0.0, 1.0 - abs(pred - true) / tol)

    if task == "delta":
        pred = _parse_number(response)
        true = gt.get("delta")
        tol = _regime_tol(item.get("regime", ""), "delta")
        if pred is None or true is None:
            return 0.0
        return max(0.0, 1.0 - abs(pred - true) / tol)

    if task == "prob_itm_brier":
        pred = _parse_number(response)
        true = float(gt.get("prob_itm", 0))
        if pred is None:
            brier = 1.0
        else:
            pred = max(0.0, min(1.0, pred))
            brier = (pred - true) ** 2
        return max(0.0, 1.0 - brier)

    if task == "option_mid_price":
        pred = _parse_number(response)
        true = gt.get("mid_price", 0)
        if pred is None or true <= 0:
            return 0.0
        return max(0.0, 1.0 - abs(pred - true) / true)

    return 0.0


def _parse_reference(ref: Any) -> dict:
    if isinstance(ref, dict):
        return ref
    return json.loads(ref)


@LOAD_DATASET.register_module()
class HeliumMarketResolutionDataset(BaseDataset):
    """Helium Market Resolution — frozen option-chain prompts from Hugging Face.

    Dataset: HeliumTrades/helium-market-resolution-benchmark
    Landing: https://heliumtrades.com/benchmarks/
    """

    @staticmethod
    def load(path: str, mini: bool = False):
        ds = load_dataset(path, split="test")

        def prep(example):
            gt = example["ground_truth"]
            if isinstance(gt, str):
                gt = json.loads(gt)
            example["reference"] = json.dumps(
                {
                    "task": example["task"],
                    "ground_truth": gt,
                    "regime": example.get("regime", ""),
                    "scoring_tier": example.get("scoring_tier", ""),
                }
            )
            return example

        ds = ds.map(prep)
        if mini:
            ds = ds.select(range(min(20, len(ds))))
        return {"train": ds, "test": ds}


@ICL_EVALUATORS.register_module()
class HeliumMarketResolutionEvaluator(BaseEvaluator):
    """Partial-credit scoring for option-chain tasks (IV, delta, MCQ)."""

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {"error": "predictions and references have different length"}

        scores: list[float] = []
        core_scores: list[float] = []
        details = []

        for pred, ref in zip(predictions, references):
            item = _parse_reference(ref)
            s = score_market_resolution_item(item, str(pred))
            scores.append(s)
            if item.get("scoring_tier") == "core":
                core_scores.append(s)
            details.append({"score": s, "task": item.get("task")})

        overall = sum(scores) / len(scores) * 100 if scores else 0.0
        core = sum(core_scores) / len(core_scores) * 100 if core_scores else overall
        return {
            "score": overall,
            "core_score": core,
            "details": details,
        }
