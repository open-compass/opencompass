from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.smolinstruct import RMSEEvaluator
from opencompass.datasets import SmolInstructDataset

pp_rmse_reader_cfg = dict(
    input_columns=['input'],
    output_column='output',
    train_split='validation')

pp_rmse_hint_dict = {
    'ESOL': """You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the log solubility of the compound.
    The input contains the SMILES representation of the compound. Your reply should contain the log solubility of the compound wrapped in <NUMBER> and </NUMBER> tags. Your reply must be valid and chemically reasonable.""",
    'Lipo': """You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the octanol/water partition coefficient of the compound.
    The input contains the SMILES representation of the compound. Your reply should contain the octanol/water partition coefficient of the compound wrapped in <NUMBER> and </NUMBER> tags. Your reply must be valid and chemically reasonable."""
}

name_dict = {
    'ESOL': 'property_prediction-esol',
    'Lipo': 'property_prediction-lipo'
}

pp_rmse_datasets = []
for _name in pp_rmse_hint_dict:
    _hint = pp_rmse_hint_dict[_name]
    pp_rmse_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=f'{_hint}\nQuestion: {{input}}\nAnswer: '
                ),
                dict(role='BOT', prompt='{output}\n')
            ]),
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=f'{_hint}\nQuestion: {{input}}\nAnswer: '
                    ),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0]),
        inferencer=dict(type=GenInferencer),
    )

    pp_rmse_eval_cfg = dict(
        evaluator=dict(type=RMSEEvaluator),
    )

    pp_rmse_datasets.append(
        dict(
            abbr=f'PP-{_name}',
            type=SmolInstructDataset,
            path='osunlp/SMolInstruct',
            name=name_dict[_name],
            reader_cfg=pp_rmse_reader_cfg,
            infer_cfg=pp_rmse_infer_cfg,
            eval_cfg=pp_rmse_eval_cfg,
        ))

del _name, _hint