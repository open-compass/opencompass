from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AiderEvaluator
from opencompass.datasets import AiderDataset

aider_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='judge',
    )

data_path = './data/aider/'
aider_all_sets = ['Aider.json']
get_aider_dataset = []


for _name in aider_all_sets:
    aider_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[dict(role='SYSTEM', fallback_role='HUMAN', prompt='{system_prompt}')],round=[dict(
                        role='HUMAN',
                        prompt='{prompt}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )

    aider_eval_cfg = dict(
        evaluator=dict(
            type=AiderEvaluator,
            ip_address='https://sd17oge6kiaj519k4ofj0.apigateway-cn-beijing.volceapi.com'
        ),
    )

    get_aider_dataset.append(
        dict(
            abbr=f'{_name.split(".")[0]}',
            type=AiderDataset,
            path=data_path,
            name=_name,
            reader_cfg=aider_reader_cfg,
            infer_cfg=aider_infer_cfg,
            eval_cfg=aider_eval_cfg,
        ))
