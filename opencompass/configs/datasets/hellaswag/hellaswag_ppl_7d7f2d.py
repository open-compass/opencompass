from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HellaswagDataset_V2

hellaswag_reader_cfg = dict(
    input_columns=['query', 'A', 'B', 'C', 'D'],
    output_column='label')

hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            ans: dict(round=[
                dict(role='HUMAN', prompt='{ctx}\nQuestion: Which ending makes the most sense?\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: '),
                dict(role='BOT', prompt=f'{ans}'),
            ]) for ans in ['A', 'B', 'C', 'D']
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

hellaswag_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

hellaswag_datasets = [
    dict(
        abbr='hellaswag',
        type=HellaswagDataset_V2,
        path='opencompass/hellaswag',
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
