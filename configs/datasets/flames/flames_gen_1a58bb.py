from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import FlamesDataset

subjective_reader_cfg = dict(
    input_columns=['prompt','instruction'],
    output_column='judge',
    )

subjective_all_sets = [
    'data_protection', 'legality', 'morality_non_environmental_friendly', 'morality_disobey_social_norm', 'morality_chinese_values', 'safety_non_anthropomorphism', 'safety_physical_harm', 'safety_mental_harm', 'safety_property_safety', 'fairness'
]


#this is the path to flames dataset
data_path ='./data/flames'

flames_datasets = []

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
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
            inferencer=dict(type=GenInferencer, max_out_len=2048),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = '{instruction}{prediction}',
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    flames_datasets.append(
        dict(
            abbr=f'{_name}',
            type=FlamesDataset,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg
        ))
