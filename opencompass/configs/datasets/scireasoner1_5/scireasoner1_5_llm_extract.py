from opencompass.datasets.SciReasoner1_5 import (
    SciReasoner15Dataset,
    SciReasoner15DudeEvaluator,
    SciReasoner15GOEvaluator,
    scireasoner15_material_llm_postprocess,
    scireasoner15_tmscore_llm_postprocess,
)
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


SCIREASONER15_DATA_ROOT = 'opencompass/SciReasoner1.5'
SCIREASONER15_MINI_SAMPLE_SIZE = 150
SCIREASONER15_OUTPUT_INSTRUCTIONS = {
    'go_bp':
    'Return only Gene Ontology biological process terms separated by '
    'semicolons. Do not include explanations, numbering, bullets, or extra '
    'text.',
    'tmscore':
    'Return only one float between 0 and 1. Do not include units or '
    'explanation.',
    'dude_count':
    'Return only one similarity/probability score between 0 and 1. Do not '
    'include units or explanation.',
}

scireasoner1_5_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answer',
)

scireasoner1_5_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{prompt}'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


def _make_infer_cfg(output_instruction=None):
    if not output_instruction:
        return scireasoner1_5_infer_cfg
    return dict(
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[{
                'role': 'user',
                'content': '{prompt}\n\n' + output_instruction,
            }],
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )


# The judge LLM only extracts the final numeric value stated by the model; it
# never sees the gold answer, so extraction cannot be biased towards it. The
# extracted values are compared against gold by the dict postprocessors
# registered in opencompass/datasets/SciReasoner1_5.py.
SCIREASONER15_JUDGE_SYSTEM = (
    "You are a helpful assistant who extracts the final numeric answer from "
    "a candidate model's output.")

SCIREASONER15_MATERIAL_JUDGE_TEMPLATE = """
Please as a grading expert, extract the final numeric answer from the candidate's response below.
Here are some criteria:
1. You don't need to solve or verify the question. You only need to find the single final numeric value the candidate stated for the target material property ({property}).
2. COMPLETELY IGNORE any reasoning processes, intermediate steps, or explanations; take the FINAL value the candidate gives for the property.
3. Copy the number exactly as stated by the candidate, in plain decimal or scientific notation (e.g. 3.14, -0.05, 5.2e21), without units and without any other text. Do NOT compute, convert, or correct it yourself.
4. If the candidate states multiple values for the property, take the last one.
5. If the candidate gives no numeric value for the property (e.g. the response is truncated, irrelevant, or it refuses to answer), reply with exactly: INVALID
Reply with only the extracted number or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to extract the answer.
<Original Question Begin>:
{prompt}
<Original Question End>
<Candidate's Answer Begin>:
{prediction}
<Candidate's Answer End>
Extracting the final numeric value of {property} from the candidate's answer:
""".strip()


def _material_judge_content(property_name):
    # Replace the {property} placeholder instead of str.format so that the
    # {prompt}/{prediction} column placeholders stay intact for the template.
    return SCIREASONER15_MATERIAL_JUDGE_TEMPLATE.replace(
        '{property}', property_name)


SCIREASONER15_TMSCORE_JUDGE_TEMPLATE = """
Please as a grading expert, extract the final numeric answer from the candidate's response below.
Here are some criteria:
1. You don't need to solve or verify the question. You only need to find the single final TM-score value (a structural similarity score between 0 and 1) the candidate stated.
2. COMPLETELY IGNORE any reasoning processes, intermediate steps, or explanations; take the FINAL value the candidate gives.
3. Copy the number exactly as stated by the candidate, in plain decimal or scientific notation (e.g. 0.87), without units and without any other text. Do NOT compute, convert, or correct it yourself.
4. If the candidate states multiple values, take the last one.
5. If the candidate gives no numeric value (e.g. the response is truncated, irrelevant, or it refuses to answer), reply with exactly: INVALID
Reply with only the extracted number or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to extract the answer.
<Original Question Begin>:
{prompt}
<Original Question End>
<Candidate's Answer Begin>:
{prediction}
<Candidate's Answer End>
Extracting the final TM-score from the candidate's answer:
""".strip()

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

# The TMScore entry is handled by the GenericLLMEvaluator branch in
# _make_simple_dataset, so its evaluator slot is unused.
OTHER_TASKS = [
    ('GO-BP', 'go_bp', SciReasoner15GOEvaluator),
    ('TMScore', 'tmscore', None),
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
                type=GenericLLMEvaluator,
                prompt_template=dict(
                    type=RawPromptTemplate,
                    messages=[
                        {
                            'role': 'system',
                            'content': SCIREASONER15_JUDGE_SYSTEM,
                        },
                        {
                            'role': 'user',
                            'content':
                            _material_judge_content(property_name),
                        },
                    ],
                ),
                dataset_cfg=dict(
                    type=SciReasoner15Dataset,
                    path=SCIREASONER15_DATA_ROOT,
                    name=f'{source}:{property_name}',
                    source=source,
                    task_type='material',
                    property_name=property_name,
                    mini_set=mini_set,
                    sample_size=sample_size,
                    reader_cfg=scireasoner1_5_reader_cfg,
                ),
                judge_cfg=dict(),
                dict_postprocessor=dict(
                    type=scireasoner15_material_llm_postprocess),
            ),
        ),
    )


def _make_simple_dataset(display_name, task_type, evaluator, mini_set=False):
    sample_size = SCIREASONER15_MINI_SAMPLE_SIZE if mini_set else None
    suffix = '-mini' if mini_set else ''
    if task_type == 'tmscore':
        eval_cfg = dict(
            evaluator=dict(
                type=GenericLLMEvaluator,
                prompt_template=dict(
                    type=RawPromptTemplate,
                    messages=[
                        {
                            'role': 'system',
                            'content': SCIREASONER15_JUDGE_SYSTEM,
                        },
                        {
                            'role': 'user',
                            'content': SCIREASONER15_TMSCORE_JUDGE_TEMPLATE,
                        },
                    ],
                ),
                dataset_cfg=dict(
                    type=SciReasoner15Dataset,
                    path=SCIREASONER15_DATA_ROOT,
                    name=task_type,
                    task_type=task_type,
                    mini_set=mini_set,
                    sample_size=sample_size,
                    reader_cfg=scireasoner1_5_reader_cfg,
                ),
                judge_cfg=dict(),
                dict_postprocessor=dict(
                    type=scireasoner15_tmscore_llm_postprocess),
            ),
        )
    else:
        eval_cfg = dict(evaluator=dict(type=evaluator))
    return dict(
        abbr=f'SciReasoner1_5-{display_name}{suffix}',
        type=SciReasoner15Dataset,
        path=SCIREASONER15_DATA_ROOT,
        name=task_type,
        task_type=task_type,
        mini_set=mini_set,
        sample_size=sample_size,
        reader_cfg=scireasoner1_5_reader_cfg,
        infer_cfg=_make_infer_cfg(
            SCIREASONER15_OUTPUT_INSTRUCTIONS.get(task_type)),
        eval_cfg=eval_cfg,
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
