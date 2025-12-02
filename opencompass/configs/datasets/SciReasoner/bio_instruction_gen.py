from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import Bioinstruction_Dataset, bio_instruction_Evaluator

reader_cfg = dict(
    input_columns=['input'],
    output_column='output'
)

MODEL_NAME = r'model'

bio_instruction_datasets = []
mini_bio_instruction_datasets = []

path = ['antibody_antigen', 'rna_protein_interaction', 'emp', 'enhancer_activity', 'tf_m', 'Isoform', 'Modification',
        'MeanRibosomeLoading', 'ProgrammableRNASwitches',
        'CRISPROnTarget', 'promoter_enhancer_interaction', 'sirnaEfficiency', 'cpd', 'pd', 'tf_h']
extra_path = ['Fluorescence', 'FunctionEC', 'Stability', 'Solubility', 'Thermostability']  # protein的这几个

for task in path:
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='{input}'),
                ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    eval_cfg = dict(
        evaluator=dict(type=bio_instruction_Evaluator,
                       path='opencompass/SciReasoner-bio_instruction',
                       task=task,
                       model_name=MODEL_NAME),
        pred_role='BOT',
        num_gpus=1
    )
    eval_mini_cfg = dict(
        evaluator=dict(type=bio_instruction_Evaluator,
                       path='opencompass/SciReasoner-bio_instruction',
                       task=task,
                       mini_set=True,
                       model_name=MODEL_NAME),
        pred_role='BOT',
        num_gpus=1
    )

    bio_instruction_datasets.append(
        dict(
            type=Bioinstruction_Dataset,
            abbr=f'SciReasoner-bio_instruction-{task}',
            path='opencompass/SciReasoner-bio_instruction',
            task=task,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg,
        )
    )
    mini_bio_instruction_datasets.append(
        dict(
            type=Bioinstruction_Dataset,
            abbr=f'SciReasoner-bio_instruction-{task}-mini',
            path='opencompass/SciReasoner-bio_instruction',
            task=task,
            mini_set=True,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_mini_cfg,
        )
    )
