# base config for LLM4Chem
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import Mol_Instructions_postprocess_Mol, Mol_Instructions_Evaluator_Mol, Mol_Instructions_Dataset

TASKS = [
         'property_prediction_str',
         'description_guided_molecule_design',
         'forward_reaction_prediction',
         'retrosynthesis',
         'reagent_prediction',
         'molecular_description_generation'
         ]

reader_cfg = dict(input_columns=['input'], output_column='output')

mol_mol_datasets = []
mini_mol_mol_datasets = []

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{input}\n{output}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=32000))

for task in TASKS:
    eval_cfg = dict(
        evaluator=dict(type=Mol_Instructions_Evaluator_Mol, task=task),
        pred_postprocessor=dict(type=Mol_Instructions_postprocess_Mol, task=task),
        dataset_postprocessor=dict(type=Mol_Instructions_postprocess_Mol, task=task),
    )

    mol_mol_datasets.append(
        dict(
            abbr=f'mol_instruction_{task}',
            type=Mol_Instructions_Dataset,
            train_path=f'/path/Mol-Instructions-test/{task}/dev/data.json',
            test_path=f'/path/Mol-Instructions-test/{task}/test/data.json',
            hf_hub=False,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg
        )
    )
    mini_mol_mol_datasets.append(
        dict(
            abbr=f'mol_instruction_{task}-mini',
            type=Mol_Instructions_Dataset,
            train_path=f'/path/Mol-Instructions-test/{task}/dev/data.json',
            test_path=f'/path/Mol-Instructions-test/{task}/test/data.json',
            mini_set=True,
            hf_hub=False,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg
        )
    )



