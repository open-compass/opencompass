from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import Judgerbenchv2Evaluator
from opencompass.datasets import Judgerbenchv2Dataset

judgerbenchv2_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='judge',
    )

data_path = './data/judgeeval/judgerbenchv2'
judgerbenchv2_all_sets = ['Knowledge', 'Longtext', 'Reason_and_analysis', 'safe', 'Hallucination', 'chatQA', 'IF', 'LanTask', 'Creation', 'Code_and_AI']
get_judgerbenchv2_dataset = []


for _name in judgerbenchv2_all_sets:
    judgerbenchv2_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{prompt}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=4096),
        )

    judgerbenchv2_eval_cfg = dict(
        evaluator=dict(
            type=Judgerbenchv2Evaluator,
        ),
    )

    get_judgerbenchv2_dataset.append(
        dict(
            abbr=f'{_name}',
            type=Judgerbenchv2Dataset,
            path=data_path,
            name=_name,
            reader_cfg=judgerbenchv2_reader_cfg,
            infer_cfg=judgerbenchv2_infer_cfg,
            eval_cfg=judgerbenchv2_eval_cfg,
        ))
