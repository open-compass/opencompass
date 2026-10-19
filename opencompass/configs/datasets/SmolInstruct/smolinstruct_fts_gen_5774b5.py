from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.smolinstruct import FTSEvaluator
from opencompass.datasets import SmolInstructDataset

fts_reader_cfg = dict(
    input_columns=['input'],
    output_column='output',
    train_split='validation')

fts_hint_dict = {
    'MG': """You are an expert chemist. Given the description of a molecule, your task is to generate the potential SMILES representation of the molecule.
    The input contains the description of the molecule. Your reply should contain the potential SMILES representation of the molecule wrapped in <SMILES> and </SMILES> tags. Your reply must be valid and chemically reasonable.""",
    'FS': """You are an expert chemist. Given the SMILES representation of reactants and reagents, your task is to predict the potential product using your chemical reaction knowledge.
    The input contains both reactants and reagents, and different reactants and reagents are separated by ".". Your reply should contain only the SMILES representation of the predicted product and no other text. Your reply must be valid and chemically reasonable.""",
    'RS': """You are an expert chemist. Given the SMILES representation of the product, your task is to predict the potential reactants and reagents using your chemical reaction knowledge.
    The input contains the SMILES representation of the product. Your reply should contain the SMILES representation of both reactants and reagents, and different reactants and reagents should be separated by ".". Your reply must be valid and chemically reasonable.""",
}

name_dict = {
    'MG': 'molecule_generation',
    'FS': 'forward_synthesis',
    'RS': 'retrosynthesis'
}

fts_datasets = []
for _name in fts_hint_dict:
    _hint = fts_hint_dict[_name]
    fts_infer_cfg = dict(
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

    fts_eval_cfg = dict(
        evaluator=dict(type=FTSEvaluator),
    )

    fts_datasets.append(
        dict(
            abbr=f'{_name}',
            type=SmolInstructDataset,
            path='osunlp/SMolInstruct',
            name=name_dict[_name],
            reader_cfg=fts_reader_cfg,
            infer_cfg=fts_infer_cfg,
            eval_cfg=fts_eval_cfg,
        ))

del _name, _hint