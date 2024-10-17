from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer, GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import MTBench101Dataset
from opencompass.summarizers import MTBench101Summarizer

subjective_reader_cfg = dict(
    input_columns=['dialogue','task','multi_id','turn_id','system_prompt','prompt_template'],
    output_column='judge',
    )

subjective_all_sets = [
    'mtbench101',
]
data_path ='data/subjective/'

mtbench101_datasets = []

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template="""{dialogue}""",
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=ChatInferencer, max_seq_len=4096, max_out_len=4096, infer_mode='last'),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt='{system_prompt}')
                ],
                    round=[
                    dict(
                        role='HUMAN',
                        prompt = '{prompt_template}'
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    mtbench101_datasets.append(
        dict(
            abbr=f'{_name}',
            type=MTBench101Dataset,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='singlescore',
            summarizer = dict(type=MTBench101Summarizer, judge_type='single')
        ))
