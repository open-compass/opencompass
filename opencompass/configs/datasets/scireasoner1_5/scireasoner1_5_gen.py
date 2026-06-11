import os

from opencompass.datasets.SciReasoner1_5 import (
    SciReasoner15Dataset,
    SciReasoner15DudeEvaluator,
    SciReasoner15GOEvaluator,
    SciReasoner15MaterialEvaluator,
    SciReasoner15TMScoreEvaluator,
)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


SCIREASONER15_DATA_ROOT = 'SciReasoner/test-1.5'
SCIREASONER15_MINI_SAMPLE_SIZE = int(
    os.environ.get('SCIREASONER15_MINI_SAMPLE_SIZE', '150'))

scireasoner1_5_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answer',
)

scireasoner1_5_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{prompt}'),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

MATERIAL_TASKS = [
    ('OQMD-bandgap', 'oqmd', 'bandgap'),
    ('OQMD-e_form', 'oqmd', 'e_form'),
    ('JARVISDFT-formation_energy_peratom', 'jarvis_dft',
     'formation_energy_peratom'),
    ('JARVISDFT-optb88vdw_bandgap', 'jarvis_dft', 'optb88vdw_bandgap'),
    ('JARVISDFT-optb88vdw_total_energy', 'jarvis_dft',
     'optb88vdw_total_energy'),
    ('JARVISDFT-ehull', 'jarvis_dft', 'ehull'),
    ('JARVISDFT-n-Seebeck', 'jarvis_dft', 'n-Seebeck'),
    ('JARVISDFT-n-powerfact', 'jarvis_dft', 'n-powerfact'),
    ('JARVISDFT-p-Seebeck', 'jarvis_dft', 'p-Seebeck'),
    ('JARVISDFT-p-powerfact', 'jarvis_dft', 'p-powerfact'),
    ('JARVISDFT-bulk_modulus_kv', 'jarvis_dft', 'bulk_modulus_kv'),
    ('JARVISDFT-shear_modulus_gv', 'jarvis_dft', 'shear_modulus_gv'),
    ('JARVISDFT-mbj_bandgap', 'jarvis_dft', 'mbj_bandgap'),
    ('JARVISDFT-mepsx', 'jarvis_dft', 'mepsx'),
    ('JARVISDFT-avg_elec_mass', 'jarvis_dft', 'avg_elec_mass'),
    ('JARVISDFT-max_efg', 'jarvis_dft', 'max_efg'),
    ('JARVISDFT-spillage', 'jarvis_dft', 'spillage'),
    ('JARVISDFT-slme', 'jarvis_dft', 'slme'),
    ('JARVISDFT-dfpt_piezo_max_eij', 'jarvis_dft',
     'dfpt_piezo_max_eij'),
    ('JARVISDFT-dfpt_piezo_max_dielectric', 'jarvis_dft',
     'dfpt_piezo_max_dielectric'),
    ('JARVISDFT-dfpt_piezo_max_dij', 'jarvis_dft',
     'dfpt_piezo_max_dij'),
    ('JARVISDFT-exfoliation_energy', 'jarvis_dft',
     'exfoliation_energy'),
]

OTHER_TASKS = [
    ('GO-BP', 'go_bp', SciReasoner15GOEvaluator),
    ('TMScore', 'tmscore', SciReasoner15TMScoreEvaluator),
    ('DUDE-count', 'dude_count', SciReasoner15DudeEvaluator),
]


def _make_material_dataset(display_name, source, property_name, mini_set=False):
    sample_size = SCIREASONER15_MINI_SAMPLE_SIZE if mini_set else None
    suffix = '-mini' if mini_set else ''
    return dict(
        abbr=f'SciReasoner1_5-{display_name}{suffix}',
        type=SciReasoner15Dataset,
        path=SCIREASONER15_DATA_ROOT,
        name=f'{source}:{property_name}',
        source=source,
        task_type='material',
        property_name=property_name,
        mini_set=mini_set,
        sample_size=sample_size,
        reader_cfg=scireasoner1_5_reader_cfg,
        infer_cfg=scireasoner1_5_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=SciReasoner15MaterialEvaluator,
                property_name=property_name,
            ),
        ),
    )


def _make_simple_dataset(display_name, task_type, evaluator, mini_set=False):
    sample_size = SCIREASONER15_MINI_SAMPLE_SIZE if mini_set else None
    suffix = '-mini' if mini_set else ''
    return dict(
        abbr=f'SciReasoner1_5-{display_name}{suffix}',
        type=SciReasoner15Dataset,
        path=SCIREASONER15_DATA_ROOT,
        name=task_type,
        task_type=task_type,
        mini_set=mini_set,
        sample_size=sample_size,
        reader_cfg=scireasoner1_5_reader_cfg,
        infer_cfg=scireasoner1_5_infer_cfg,
        eval_cfg=dict(evaluator=dict(type=evaluator)),
    )


scireasoner1_5_datasets = []
mini_scireasoner1_5_datasets = []

for _display_name, _source, _property_name in MATERIAL_TASKS:
    scireasoner1_5_datasets.append(
        _make_material_dataset(
            _display_name,
            _source,
            _property_name,
            mini_set=False,
        ))
    mini_scireasoner1_5_datasets.append(
        _make_material_dataset(
            _display_name,
            _source,
            _property_name,
            mini_set=True,
        ))

for _display_name, _task_type, _evaluator in OTHER_TASKS:
    scireasoner1_5_datasets.append(
        _make_simple_dataset(
            _display_name,
            _task_type,
            _evaluator,
            mini_set=False,
        ))
    mini_scireasoner1_5_datasets.append(
        _make_simple_dataset(
            _display_name,
            _task_type,
            _evaluator,
            mini_set=True,
        ))
