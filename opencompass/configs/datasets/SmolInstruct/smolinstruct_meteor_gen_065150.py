from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.smolinstruct import MeteorEvaluator
from opencompass.datasets import SmolInstructDataset

meteor_reader_cfg = dict(
    input_columns=['input'],
    output_column='output',
    train_split='validation')

meteor_hint_dict = {
    'MC': """You are an expert chemist. Given the SMILES representation of a molecule, your task is to describe the molecule in natural language.
    The input contains the SMILES representation of the molecule. Your reply should contain a natural language description of the molecule. Your reply must be valid and chemically reasonable.""",
}

name_dict = {
    'MC': 'molecule_captioning',
}

meteor_datasets = []
for _name in meteor_hint_dict:
    _hint = meteor_hint_dict[_name]
    meteor_infer_cfg = dict(
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

    meteor_eval_cfg = dict(
        evaluator=dict(type=MeteorEvaluator),
    )

    meteor_datasets.append(
        dict(
            abbr=f'{_name}',
            type=SmolInstructDataset,
            path='osunlp/SMolInstruct',
            name=name_dict[_name],
            reader_cfg=meteor_reader_cfg,
            infer_cfg=meteor_infer_cfg,
            eval_cfg=meteor_eval_cfg,
        ))

del _name, _hint
