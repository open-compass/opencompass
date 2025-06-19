from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.smolinstruct import NCExactMatchEvaluator, NCElementMatchEvaluator
from opencompass.datasets import SmolInstructDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

nc_0shot_reader_cfg = dict(
    input_columns=['input'],
    output_column='output',
    train_split='validation')

nc_hint_dict = {
    'I2F': """You are an expert chemist. Given the IUPAC representation of compounds, your task is to predict the molecular formula of the compound.
    The input contains the IUPAC representation of the compound. Your reply should contain only the molecular formula of the compound wrapped in <MOLFORMULA> and </MOLFORMULA> tags and no other text. Your reply must be valid and chemically reasonable.""",
    'I2S': """You are an expert chemist. Given the IUPAC representation of compounds, your task is to predict the SMILES representation of the compound.
    The input contains the IUPAC representation of the compound. Your reply should contain only the SMILES representation of the compound wrapped in <SMILES> and </SMILES> tags and no other text. Your reply must be valid and chemically reasonable.""",
    'S2F': """You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the molecular formula of the compound.
    The input contains the SMILES representation of the compound. Your reply should contain only the molecular formula of the compound wrapped in <MOLFORMULA> and </MOLFORMULA> tags and no other text. Your reply must be valid and chemically reasonable.""",
    'S2I': """You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the IUPAC representation of the compound.
    The input contains the SMILES representation of the compound. Your reply should contain only the IUPAC representation of the compound wrapped in <IUPAC> and </IUPAC> tags and no other text. Your reply must be valid and chemically reasonable.""",
}

name_dict = {
    'I2F': 'name_conversion-i2f',
    'I2S': 'name_conversion-i2s',
    'S2F': 'name_conversion-s2f',
    'S2I': 'name_conversion-s2i',
}

nc_0shot_instruct_datasets = []
for _name in name_dict:
    _hint = nc_hint_dict[_name]
    nc_0shot_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt=f'{_hint}\nQuestion: {{input}}\nAnswer: '),
                dict(role='BOT', prompt='{output}\n')
            ]),
            # template=f'<s>[INST] {{input}} [/INST]',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    if _name in ['I2F', 'S2F']:
        nc_0shot_eval_cfg = dict(
            evaluator=dict(type=NCElementMatchEvaluator),
        )
    else:
        nc_0shot_eval_cfg = dict(
            evaluator=dict(type=NCExactMatchEvaluator),
        )

    nc_0shot_instruct_datasets.append(
        dict(
            abbr=f'NC-{_name}-0shot-instruct',
            type=SmolInstructDataset,
            path='osunlp/SMolInstruct',
            name=name_dict[_name],
            reader_cfg=nc_0shot_reader_cfg,
            infer_cfg=nc_0shot_infer_cfg,
            eval_cfg=nc_0shot_eval_cfg,
        ))

del _name
