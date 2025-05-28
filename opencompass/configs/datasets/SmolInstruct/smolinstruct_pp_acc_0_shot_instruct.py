from opencompass.openicl import AccEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SmolInstructDataset
from opencompass.datasets.smolinstruct import smolinstruct_acc_0shot_postprocess

pp_acc_reader_cfg = dict(
    input_columns=['input'],
    output_column='output',
    train_split='validation')

pp_acc_hint_dict = {
    'BBBP': """You are an expert chemist. Given the smiles representation of the compound, your task is to predict whether blood-brain barrier permeability (BBBP) is a property of the compound.
    The input contains the compound. Your reply should only contain Yes or No. Your reply must be valid and chemically reasonable.""",
    'ClinTox': """You are an expert chemist. Given the smiles representation of the compound, your task is to predict whether the compound is toxic.
    The input contains the compound. Your reply should contain only Yes or No. Your reply must be valid and chemically reasonable.""",
    'HIV': """You are an expert chemist. Given the smiles representation of the compound, your task is to predict whether the compound serve as an inhibitor of HIV replication.
    The input contains the compound. Your reply should contain only Yes or No. Your reply must be valid and chemically reasonable.""",
    'SIDER': """You are an expert chemist. Given the smiles representation of the compound, your task is to predict whether the compound has any side effects.
    The input contains the compound. Your reply should contain only Yes or No. Your reply must be valid and chemically reasonable.""",
}

name_dict = {
    'BBBP': 'property_prediction-bbbp',
    'ClinTox': 'property_prediction-clintox',
    'HIV': 'property_prediction-hiv',
    'SIDER': 'property_prediction-sider',
}

pp_acc_datasets_0shot_instruct = []
for _name in pp_acc_hint_dict:
    _hint = pp_acc_hint_dict[_name]

    pp_acc_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=f'{_hint}\nQuestion: {{input}}\nAnswer: ',
            # template=f'<s>[INST] {{input}} [/INST]',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    pp_acc_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=smolinstruct_acc_0shot_postprocess)
    )

    pp_acc_datasets_0shot_instruct.append(
        dict(
            abbr=f'PP-{_name}-0shot-instruct',
            type=SmolInstructDataset,
            path='osunlp/SMolInstruct',
            name=name_dict[_name],
            reader_cfg=pp_acc_reader_cfg,
            infer_cfg=pp_acc_infer_cfg,
            eval_cfg=pp_acc_eval_cfg,
        ))

del _name, _hint
