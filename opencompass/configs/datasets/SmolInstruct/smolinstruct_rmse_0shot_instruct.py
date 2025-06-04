from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.smolinstruct import RMSEEvaluator
from opencompass.datasets import SmolInstructDataset

pp_rmse_0shot_reader_cfg = dict(
    input_columns=['input'],
    output_column='output',
    train_split='validation')

pp_rmse_hint_dict = {
    'ESOL': """You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the log solubility of the compound.
    The input contains the SMILES representation of the compound. Your reply should contain the log solubility of the compound wrapped in \\boxed{}. Your reply must be valid and chemically reasonable.""",
    'Lipo': """You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the octanol/water partition coefficient of the compound.
    The input contains the SMILES representation of the compound. Your reply should contain the octanol/water partition coefficient of the compound wrapped in \\boxed{}. Your reply must be valid and chemically reasonable."""
}

name_dict = {
    'ESOL': 'property_prediction-esol',
    'Lipo': 'property_prediction-lipo'
}

pp_rmse_0shot_instruct_datasets = []
for _name in name_dict:
    _hint = pp_rmse_hint_dict[_name]
    pp_rmse_0shot_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=f'{_hint}\nQuestion: {{input}}\nAnswer: ',
            # template=f'<s>[INST] {{input}} [/INST]',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    pp_rmse_0shot_eval_cfg = dict(
        evaluator=dict(type=RMSEEvaluator),
    )

    pp_rmse_0shot_instruct_datasets.append(
        dict(
            abbr=f'PP-{_name}-0shot-instruct',
            type=SmolInstructDataset,
            path='osunlp/SMolInstruct',
            name=name_dict[_name],
            reader_cfg=pp_rmse_0shot_reader_cfg,
            infer_cfg=pp_rmse_0shot_infer_cfg,
            eval_cfg=pp_rmse_0shot_eval_cfg,
        ))

del _name