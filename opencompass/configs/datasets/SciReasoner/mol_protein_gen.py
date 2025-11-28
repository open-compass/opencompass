# base config for LLM4Chem
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (Mol_Instructions_postprocess_Protein, Mol_Instructions_Evaluator_Protein,
                                  Mol_Instructions_Dataset, Mol_Instructions_postprocess_Protein_Design,
                                  Mol_Instructions_Evaluator_Protein_Design, Mol_Instructions_Dataset_Protein_Design)

TASKS = [
    'catalytic_activity',
    'domain_motif',
    'general_function',
    'protein_function',
]

reader_cfg = dict(input_columns=['input'], output_column='output')

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{input}\n{output}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=32000))

eval_cfg = dict(
    evaluator=dict(type=Mol_Instructions_Evaluator_Protein),
    pred_postprocessor=dict(type=Mol_Instructions_postprocess_Protein),
    dataset_postprocessor=dict(type=Mol_Instructions_postprocess_Protein),
)

eval_cfg_protein_design = dict(
    evaluator=dict(type=Mol_Instructions_Evaluator_Protein_Design),
    pred_postprocessor=dict(type=Mol_Instructions_postprocess_Protein_Design),
    dataset_postprocessor=dict(type=Mol_Instructions_postprocess_Protein_Design),
)

mol_protein_datasets = []
mini_mol_protein_datasets = []

for task in TASKS:
    mol_protein_datasets.append(
        dict(
            abbr=f'mol_instruction_{task}',
            type=Mol_Instructions_Dataset,
            train_path=f'/path/Mol-Instructions-test/{task}/dev/data.json',
            test_path=f'/path/Mol-Instructions-test/{task}/test/data.json',
            hf_hub=False,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg))
    mini_mol_protein_datasets.append(
        dict(
            abbr=f'mol_instruction_{task}-mini',
            type=Mol_Instructions_Dataset,
            train_path=f'/path/Mol-Instructions-test/{task}/dev/data.json',
            test_path=f'/path/Mol-Instructions-test/{task}/test/data.json',
            mini_set=True,
            hf_hub=False,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg))

task = 'protein_design'
mol_protein_datasets.append(
    dict(
        abbr='mol_instruction_protein_design',
        type=Mol_Instructions_Dataset_Protein_Design,
        train_path=f'/path/Mol-Instructions-test/{task}/dev/data.json',
        test_path=f'/path/Mol-Instructions-test/{task}/test/data.json',
        hf_hub=False,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg_protein_design))
mini_mol_protein_datasets.append(
    dict(
        abbr='mol_instruction_protein_design-mini',
        type=Mol_Instructions_Dataset_Protein_Design,
        train_path=f'/path/Mol-Instructions-test/{task}/dev/data.json',
        test_path=f'/path/Mol-Instructions-test/{task}/test/data.json',
        mini_set=True,
        hf_hub=False,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg_protein_design))
