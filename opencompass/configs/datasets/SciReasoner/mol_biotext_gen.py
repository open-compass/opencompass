# base config for LLM4Chem
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import Mol_Instructions_postprocess_BioText, Mol_Instructions_Evaluator_BioText, \
    Mol_Instructions_Dataset_BioText

TASKS = [
    'chemical_disease_interaction_extraction',
    'chemical_entity_recognition',
    'chemical_protein_interaction_extraction',
    'multi_choice_question',
    'open_question',
    'true_or_false_question'
]

reader_cfg = dict(input_columns=['input'], output_column='output')

mol_biotext_datasets = []
mini_mol_biotext_datasets = []

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{input}\n{output}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

infer_cfg_true_or_false = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{input}Your answer should start with 'Yes' or 'Maybe' or 'No'.\n{output}",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

infer_cfg_CER = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                # Optional but recommended: A system prompt for better instructions.
                dict(role='SYSTEM', fallback_role='HUMAN',
                     prompt='There is a single choice question about chemistry. Answer the question directly.'),
                # The placeholder is the ice_token string itself, used as a direct list element.
                '</E>',
            ],
            round=[
                dict(role='HUMAN', prompt='Query: {input}'),
                dict(role='BOT', prompt=''),
            ]
        ),
        ice_token='</E>',
    ),
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Query: {input}'),
                dict(role='BOT', prompt='{output}'),
            ]
        )
    ),
    retriever=dict(
        type=FixKRetriever,
        fix_id_list=[0, ],  # 使用前1个示例
    ),
    inferencer=dict(type=GenInferencer)
)

for task in TASKS:
    eval_cfg = dict(
        evaluator=dict(type=Mol_Instructions_Evaluator_BioText, task=task),
        pred_postprocessor=dict(type=Mol_Instructions_postprocess_BioText, task=task),
        dataset_postprocessor=dict(type=Mol_Instructions_postprocess_BioText, task=task),
    )

    if task == 'true_or_false_question':
        apply_infer_cfg = infer_cfg_true_or_false
    elif task == 'chemical_entity_recognition':
        apply_infer_cfg = infer_cfg_CER
    else:
        apply_infer_cfg = infer_cfg

    mol_biotext_datasets.append(
        dict(
            abbr=f'SciReasoner-mol_instruction_{task}',
            type=Mol_Instructions_Dataset_BioText,
            path='opencompass/SciReasoner-Mol_Instructions',
            task=task,
            reader_cfg=reader_cfg,
            infer_cfg=apply_infer_cfg,
            eval_cfg=eval_cfg)
    )
    mini_mol_biotext_datasets.append(
        dict(
            abbr=f'SciReasoner-mol_instruction_{task}-mini',
            type=Mol_Instructions_Dataset_BioText,
            path='opencompass/SciReasoner-Mol_Instructions',
            task=task,
            mini_set=True,
            reader_cfg=reader_cfg,
            infer_cfg=apply_infer_cfg,
            eval_cfg=eval_cfg)
    )
