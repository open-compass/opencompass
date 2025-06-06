from opencompass.datasets import PhyBenchDataset, MathEEDEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

phybench_reader_cfg = dict(
            input_columns=['input'],
            output_column='target',
        )

phybench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=f'Solve the following physics problem and return only the final result as a clean LaTeX expression. No explanation. No text.\n\nQuestion: {{input}}\nAnswer: ',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

phybench_eval_cfg = dict(
            evaluator=dict(type=MathEEDEvaluator)
        )

phybench_datasets = [
    dict(
        abbr='phybench-eed',
        type=PhyBenchDataset,
        path='opencompass/PHYBench', 
        reader_cfg=phybench_reader_cfg,
        infer_cfg=phybench_infer_cfg,
        eval_cfg=phybench_eval_cfg,
    )
]
