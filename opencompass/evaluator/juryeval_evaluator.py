"""JuryEval evaluator — wraps juryeval judges as an OpenCompass evaluator."""

from typing import Dict, List, Optional

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils.logging import get_logger

logger = get_logger(__name__)


@ICL_EVALUATORS.register_module()
class JuryEvalEvaluator(BaseEvaluator):
    """Evaluator that uses juryeval LLM-as-Judge for scoring.

    Requires ``pip install juryeval[judge]``.

    Args:
        judge_type: ``"pairwise"`` or ``"pointwise"`` (default: ``"pairwise"``).
        judge_model: Model name passed to the juryeval judge
            (default: ``"gpt-4"``).
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        temperature: Sampling temperature (default: ``0.0``).
        max_retries: Retry attempts on API failure (default: ``3``).

    Example config::

        evaluator = dict(
            type=JuryEvalEvaluator,
            judge_type='pairwise',
            judge_model='gpt-4',
        )
    """

    def __init__(
        self,
        judge_type: str = 'pairwise',
        judge_model: str = 'gpt-4',
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        pred_postprocessor: Optional[dict] = None,
    ):
        super().__init__(pred_postprocessor=pred_postprocessor)
        self.judge_type = judge_type
        self.judge_model = judge_model
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries

    def score(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        test_set: Optional[Dataset] = None,
    ) -> Dict:
        try:
            if self.judge_type == 'pairwise':
                from juryeval import PairwiseJudge
                judge = PairwiseJudge(
                    model=self.judge_model,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    max_retries=self.max_retries,
                )
                scores = []
                questions = (
                    [str(s.get('question', '')) for s in test_set]
                    if test_set else ['' for _ in predictions]
                )
                for pred, ref, q in zip(predictions, references or [], questions):
                    result = judge.compare(answer_a=pred, answer_b=ref, question=q)
                    scores.append(result['score'])
                return {'score': sum(scores) / len(scores) if scores else 0.0}
            elif self.judge_type == 'pointwise':
                from juryeval import PointwiseJudge
                judge = PointwiseJudge(
                    model=self.judge_model,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    max_retries=self.max_retries,
                )
                scores = []
                questions = (
                    [str(s.get('question', '')) for s in test_set]
                    if test_set else ['' for _ in predictions]
                )
                refs = references or [None for _ in predictions]
                for pred, ref, q in zip(predictions, refs, questions):
                    result = judge.score(output=pred, question=q, reference=ref)
                    scores.append(result['score'])
                return {'score': sum(scores) / len(scores) if scores else 0.0}
            else:
                raise ValueError(f"Unknown judge_type: {self.judge_type}")
        except ImportError:
            logger.error(
                'juryeval is not installed. Run: pip install juryeval[judge]'
            )
            raise
