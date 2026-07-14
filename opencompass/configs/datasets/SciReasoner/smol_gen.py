# base config for LLM4Chem
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LLM4Chem_postprocess, LLM4Chem_Evaluator, LLM4ChemDataset

TASKS = (
    'forward_synthesis',
    'retrosynthesis',
    'molecule_captioning',
    'molecule_generation',
    'name_conversion-i2f',
    'name_conversion-i2s',
    'name_conversion-s2f',
    'name_conversion-s2i',
    'property_prediction-esol',
    'property_prediction-lipo',
    'property_prediction-bbbp',
    'property_prediction-clintox',
    'property_prediction-hiv',
    'property_prediction-sider',
)

TASKS_single = (
    'property_prediction-esol',
    'property_prediction-lipo',
    'property_prediction-bbbp',
    'property_prediction-clintox',
    'property_prediction-hiv',
    'property_prediction-sider',
)

TASK_TAGS = {
    'forward_synthesis': ('<SMILES>', '</SMILES>'),
    'retrosynthesis': ('<SMILES>', '</SMILES>'),
    'molecule_generation': ('<SMILES>', '</SMILES>'),
    'molecule_captioning': (None, None),
    'name_conversion-i2f': ('<MOLFORMULA>', '</MOLFORMULA>'),
    'name_conversion-i2s': ('<SMILES>', '</SMILES>'),
    'name_conversion-s2f': ('<MOLFORMULA>', '</MOLFORMULA>'),
    'name_conversion-s2i': ('<IUPAC>', '</IUPAC>'),
    'property_prediction-esol': ('<NUMBER>', '</NUMBER>'),
    'property_prediction-lipo': ('<NUMBER>', '</NUMBER>'),
    'property_prediction-bbbp': ('<BOOLEAN>', '</BOOLEAN>'),
    'property_prediction-clintox': ('<BOOLEAN>', '</BOOLEAN>'),
    'property_prediction-hiv': ('<BOOLEAN>', '</BOOLEAN>'),
    'property_prediction-sider': ('<BOOLEAN>', '</BOOLEAN>'),
}

all_datasets = []
mini_all_datasets = []

for task in TASKS:

    reader_cfg = dict(input_columns=['input'], output_column='output')

    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    # Optional but recommended: A system prompt for better instructions.
                    dict(role='SYSTEM', fallback_role='HUMAN', prompt=''),
                    # The placeholder is the ice_token string itself, used as a direct list element.
                    '</E>',
                ],
                round=[
                    dict(role='HUMAN', prompt=f'{{input}}'),
                ]
            ),
            ice_token='</E>',
        ),
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='{input}'),
                    dict(role='BOT', prompt='{output}'),
                ]
            )
        ),
        # retriever is responsible for retrieving examples and using ice_template to format them
        retriever=dict(
            # type=FixKRetriever,
            # fix_id_list=[0, 1, 2, 3, 4],  # Use the first 5 examples
            type=ZeroRetriever,  # For our trained model, use zero-shot
        ),
        inferencer=dict(
            type=GenInferencer,
        ))

    eval_cfg = dict(
        evaluator=dict(type=LLM4Chem_Evaluator, task=task),
        pred_postprocessor=dict(type=LLM4Chem_postprocess, task=task),
        dataset_postprocessor=dict(type=LLM4Chem_postprocess, task=task),
    )

    all_datasets.append(
        dict(
            abbr='SciReasoner-smol_' + task,
            type=LLM4ChemDataset,
            path='opencompass/SciReasoner-smol',
            task=task,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg
        )
    )
    mini_all_datasets.append(
        dict(
            abbr='SciReasoner-smol_' + task + '-mini',
            type=LLM4ChemDataset,
            path='opencompass/SciReasoner-smol',
            task=task,
            mini_set=True,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg
        )
    )
