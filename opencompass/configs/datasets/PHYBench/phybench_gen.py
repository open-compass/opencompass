import sys
import os


sys.path.append(os.path.dirname(__file__))

from EED import EED
from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATOR


@ICL_EVALUATOR.register_module()
class MathEEDEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        scores = []
        for pred, ref in zip(predictions, references):
            score, _, _, _ = EED(ref, pred)
            scores.append(score)
        return {'accuracy': sum(scores) / len(scores)}



from opencompass.datasets import PhyBenchDataset

phybench_datasets = [
    dict(
        abbr='phybench-eed',
        type=PhyBenchDataset,
        path='opencompass/PHYBench', 
        reader_cfg=dict(
            input_columns=['input'],
            output_column='target',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='plain',
                template='Solve the following physics problem and return only the final result as a clean LaTeX expression. No explanation. No text.\n\nQuestion: {{input}}\nAnswer: '
            ),
            retriever=dict(type='zero_shot'),
            inferencer=dict(type='gen', max_out_len=512)
        ),
        eval_cfg=dict(
            evaluator=dict(type=MathEEDEvaluator)
        )
    )
]
