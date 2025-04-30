from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import WritingBenchDataset, writingbench_postprocess
from mmengine.config import read_base

subjective_reader_cfg = dict(
    input_columns=['question'],
    output_column='judge',
    )

subjective_all_sets = [
    'writingbench'
]

writingbench_datasets = []

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{question}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer,),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            multi_eval=True,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt='You are an expert evaluator with extensive experience in evaluating response of given query.')
                ],
                    round=[
                    dict(
                        role='HUMAN',
                        prompt = '{prediction}'
                    ),
                ]),
            ),
            dict_postprocessor=dict(type=writingbench_postprocess),
        ),
        pred_role='BOT',
    )

    writingbench_datasets.append(
        dict(
            abbr=f'{_name}',
            type=WritingBenchDataset,
            path='./data/subjective/writingbench',
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='singlescore',
        ))
