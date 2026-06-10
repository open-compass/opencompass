from opencompass.datasets.s2_tomg_bench import (
    S2TOMGBenchDataset,
    S2TOMGBenchEvaluator,
)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


s2_tomg_bench_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='reference',
)

s2_tomg_bench_datasets = []
mini_s2_tomg_bench_datasets = []

SYSTEM_PROMPT = (
    'You are working as an assistant of a chemist user. Please follow the '
    'instruction of the chemist and generate a molecule that satisfies the '
    'requirements of the chemist user. Your final response should be a JSON '
    'object with the key \'molecule\' and the value as a SMILES string. '
    'For example, {"molecule": "[SMILES_STRING]"}.'
)

for _name, _family, _subtask in [
        ('MolCustom_AtomNum', 'MolCustom', 'AtomNum'),
        ('MolCustom_BondNum', 'MolCustom', 'BondNum'),
        ('MolCustom_FunctionalGroup', 'MolCustom', 'FunctionalGroup'),
        ('MolEdit_AddComponent', 'MolEdit', 'AddComponent'),
        ('MolEdit_DelComponent', 'MolEdit', 'DelComponent'),
        ('MolEdit_SubComponent', 'MolEdit', 'SubComponent'),
        ('MolOpt_LogP', 'MolOpt', 'LogP'),
        ('MolOpt_MR', 'MolOpt', 'MR'),
        ('MolOpt_QED', 'MolOpt', 'QED'),
]:
    s2_tomg_bench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='SYSTEM', prompt=SYSTEM_PROMPT),
                dict(role='HUMAN', prompt='{prompt}'),
            ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    s2_tomg_bench_eval_cfg = dict(
        evaluator=dict(type=S2TOMGBenchEvaluator),
    )
    s2_tomg_bench_datasets.append(
        dict(
            abbr=f'S2-TOMG-Bench-{_name}',
            type=S2TOMGBenchDataset,
            path='phenixace/S2-TOMG-Bench',
            name=_name,
            task_family=_family,
            subtask=_subtask,
            reader_cfg=s2_tomg_bench_reader_cfg,
            infer_cfg=s2_tomg_bench_infer_cfg,
            eval_cfg=s2_tomg_bench_eval_cfg,
        ))
    mini_s2_tomg_bench_datasets.append(
        dict(
            abbr=f'S2-TOMG-Bench-{_name}-mini',
            type=S2TOMGBenchDataset,
            path='phenixace/S2-TOMG-Bench-mini',
            name=_name,
            task_family=_family,
            subtask=_subtask,
            reader_cfg=s2_tomg_bench_reader_cfg,
            infer_cfg=s2_tomg_bench_infer_cfg,
            eval_cfg=s2_tomg_bench_eval_cfg,
        ))

del _name, _family, _subtask
