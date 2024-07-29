from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WinograndeDataset

# WARNING: This config cannot reproduce results in the paper.
# e.g. LLAMA2-7B Winogrande 69.2 (paper) -> 62.27 (this config)
# Please try winogrande_ll_c5cf57

winogrande_reader_cfg = dict(
    input_columns=['opt1', 'opt2'],
    output_column='answer',
)

winogrande_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            i: dict(round=[
                dict(role='HUMAN', prompt=f'Good sentence: {{opt{i}}}'),
            ])
            for i in range(1, 3)
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

winogrande_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

winogrande_datasets = [
    dict(
        abbr='winogrande',
        type=WinograndeDataset,
        path='opencompass/winogrande',
        reader_cfg=winogrande_reader_cfg,
        infer_cfg=winogrande_infer_cfg,
        eval_cfg=winogrande_eval_cfg)
]
