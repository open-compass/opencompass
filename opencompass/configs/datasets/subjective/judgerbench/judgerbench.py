from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.subjective import JudgerBenchDataset, JudgerBenchEvaluator
from mmengine.config import read_base

subjective_reader_cfg = dict(
    input_columns=['judge_prompt'],
    output_column='judge',
    )

subjective_all_sets = [
    'judgerbench_A_cn', 'judgerbench_A_en', 'judgerbench_B'
]

judgerbench_datasets = []

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{judge_prompt}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=4096),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=JudgerBenchEvaluator,
        ),
        pred_role='BOT',
    )

    judgerbench_datasets.append(
        dict(
            abbr=f'{_name}',
            type=JudgerBenchDataset,
            path='./data/subjective/judgerbench',
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
        ))
