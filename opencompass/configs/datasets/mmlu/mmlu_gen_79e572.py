from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar
mmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev')

mmlu_all_sets = [
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_physics',
    'electrical_engineering',
    'astronomy',
    'anatomy',
    'abstract_algebra',
    'machine_learning',
    'clinical_knowledge',
    'global_facts',
    'management',
    'nutrition',
    'marketing',
    'professional_accounting',
    'high_school_geography',
    'international_law',
    'moral_scenarios',
    'computer_security',
    'high_school_microeconomics',
    'professional_law',
    'medical_genetics',
    'professional_psychology',
    'jurisprudence',
    'world_religions',
    'philosophy',
    'virology',
    'high_school_chemistry',
    'public_relations',
    'high_school_macroeconomics',
    'human_sexuality',
    'elementary_mathematics',
    'high_school_physics',
    'high_school_computer_science',
    'high_school_european_history',
    'business_ethics',
    'moral_disputes',
    'high_school_statistics',
    'miscellaneous',
    'formal_logic',
    'high_school_government_and_politics',
    'prehistory',
    'security_studies',
    'high_school_biology',
    'logical_fallacies',
    'high_school_world_history',
    'professional_medicine',
    'high_school_mathematics',
    'college_medicine',
    'high_school_us_history',
    'sociology',
    'econometrics',
    'high_school_psychology',
    'human_aging',
    'us_foreign_policy',
    'conceptual_physics',
]

mmlu_datasets = []
for _name in mmlu_all_sets:
    _hint = f'The following are multiple choice questions (with answers) about  {_name.replace("_", " ")}.\n\n'
    mmlu_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=
            '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: {target}\n',
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template=
            f'{_hint}</E>{{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer:',
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer),
    )

    mmlu_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=first_capital_postprocess),
    )

    mmlu_datasets.append(
        dict(
            abbr=f'lukaemon_mmlu_{_name}',
            type=MMLUDataset,
            path='opencompass/mmlu',
            name=_name,
            reader_cfg=mmlu_reader_cfg,
            infer_cfg=mmlu_infer_cfg,
            eval_cfg=mmlu_eval_cfg,
        ))

del _name, _hint
