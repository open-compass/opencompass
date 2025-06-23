from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.smolinstruct import MeteorEvaluator
from opencompass.datasets import SmolInstructDataset

meteor_0shot_reader_cfg = dict(
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

meteor_0shot_instruct_datasets = []
for _name in name_dict:
    _hint = meteor_hint_dict[_name]
    meteor_0shot_infer_cfg = dict(
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

    meteor_0shot_eval_cfg = dict(
        evaluator=dict(type=MeteorEvaluator),
    )

    meteor_0shot_instruct_datasets.append(
        dict(
            abbr=f'{_name}-0shot-instruct',
            type=SmolInstructDataset,
            path='osunlp/SMolInstruct',
            name=name_dict[_name],
            reader_cfg=meteor_0shot_reader_cfg,
            infer_cfg=meteor_0shot_infer_cfg,
            eval_cfg=meteor_0shot_eval_cfg,
        ))

del _name
