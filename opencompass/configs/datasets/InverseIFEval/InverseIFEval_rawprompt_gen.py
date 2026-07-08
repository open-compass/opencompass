from opencompass.datasets import (InverseIFEvalDataset,
                                  InverseIFEvalJudgePromptTemplate)
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

inverse_ifeval_instruction_types = [
    dict(
        abbr='QC',
        name='Question Correction',
        count=90,
    ),
    dict(
        abbr='ITF',
        name='Intentional Textual Flaws',
        count=86,
    ),
    dict(
        abbr='CC',
        name='Code without Comments',
        count=198,
    ),
    dict(
        abbr='CCF',
        name='Counter-Conventional Formatting',
        count=82,
    ),
    dict(
        abbr='DIA',
        name='Deliberately Incorrect Answers',
        count=186,
    ),
    dict(
        abbr='II',
        name='Instructional Induction',
        count=154,
    ),
    dict(
        abbr='MIM',
        name='Mid-turn Instruction Modification',
        count=108,
    ),
    dict(
        abbr='CA',
        name='Counterfactual Answering',
        count=108,
    ),
]

inverse_ifeval_languages = [
    dict(abbr='zh', name='chinese'),
    dict(abbr='en', name='english'),
]


inverse_ifeval_reader_cfg = dict(
    input_columns=[
        'prompt', 'instruction_types', 'language', 'response_reference',
        'judge_prompt_template', 'judge_system_prompt'
    ],
    output_column='response_reference',
)

inverse_ifeval_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {
                'role': 'user',
                'content': '{prompt}',
            },
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

inverse_ifeval_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(type=InverseIFEvalJudgePromptTemplate),
        dataset_cfg=dict(
            type=InverseIFEvalDataset,
            path='m-a-p/Inverse_IFEval',
            reader_cfg=inverse_ifeval_reader_cfg,
        ),
        dict_postprocessor=dict(type='inverse_ifeval_judge_postprocess'),
    ),
    pred_role='BOT',
)


def _build_inverse_ifeval_eval_cfg(instruction_type, language):
    eval_cfg = inverse_ifeval_eval_cfg.copy()
    evaluator = eval_cfg['evaluator'].copy()
    dataset_cfg = evaluator['dataset_cfg'].copy()
    dataset_cfg['instruction_type'] = instruction_type['name']
    dataset_cfg['language'] = language['name']
    evaluator['dataset_cfg'] = dataset_cfg
    eval_cfg['evaluator'] = evaluator
    return eval_cfg


def _build_inverse_ifeval_dataset(instruction_type, language):
    return dict(
        type=InverseIFEvalDataset,
        abbr=f"InverseIFEval_{language['abbr']}_{instruction_type['abbr']}",
        path='m-a-p/Inverse_IFEval',
        instruction_type=instruction_type['name'],
        language=language['name'],
        reader_cfg=inverse_ifeval_reader_cfg,
        infer_cfg=inverse_ifeval_infer_cfg,
        eval_cfg=_build_inverse_ifeval_eval_cfg(instruction_type, language),
    )


inverse_ifeval_datasets = [
    _build_inverse_ifeval_dataset(instruction_type, language)
    for language in inverse_ifeval_languages
    for instruction_type in inverse_ifeval_instruction_types
]
