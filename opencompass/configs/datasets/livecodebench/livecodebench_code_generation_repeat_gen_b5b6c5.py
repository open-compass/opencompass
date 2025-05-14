from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    LCBCodeGenerationDataset,
    LCBCodeExecutionDataset,
    LCBTestOutputPredictionDataset,
    LCBCodeGenerationEvaluator,
    LCBCodeExecutionEvaluator,
    LCBTestOutputEvaluator
)
from opencompass.datasets.livecodebench import TestOutputPromptConstants


lcb_code_generation_reader_cfg = dict(
    input_columns=[
        'question_content',
        'format_prompt',
    ],
    # output_column='evaluation_sample',
    output_column='question_id',
)

SYSTEM_MESSAGE_GENERIC = f'You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.'

prompt_template = '### Question:\n{question_content}\n\n{format_prompt}' + \
                    '### Answer: (use the provided format with backticks)\n\n'


# Code Generation Tasks
lcb_code_generation_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=prompt_template
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

lcb_code_generation_eval_cfg = dict(
    evaluator=dict(
        type=LCBCodeGenerationEvaluator,
        num_process_evaluate=4,
        timeout=6,
    ),
    pred_role='BOT',
)

LCBCodeGeneration_dataset = dict(
    type=LCBCodeGenerationDataset,
    abbr='lcb_code_generation',
    path='opencompass/code_generation_lite',
    reader_cfg=lcb_code_generation_reader_cfg,
    infer_cfg=lcb_code_generation_infer_cfg,
    eval_cfg=lcb_code_generation_eval_cfg,
    n=5,
    k=3
)

# Code Execution Dataset
lcb_code_execution_reader_cfg = dict(
    input_columns=[
        'prompt',
    ],
    output_column='evaluation_sample',
)

lcb_code_execution_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt='You are an expert at Python programming, code execution, test case generation, and fuzzing.'
                ),
            ],
            round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}'
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

lcb_code_execution_eval_cfg = dict(
    evaluator=dict(
        type=LCBCodeExecutionEvaluator,
    ),
    pred_role='BOT',
)

LCBCodeExecution_dataset = dict(
    type=LCBCodeExecutionDataset,
    abbr='lcb_code_execution',
    path='opencompass/execution-v2',
    reader_cfg=lcb_code_execution_reader_cfg,
    infer_cfg=lcb_code_execution_infer_cfg,
    eval_cfg=lcb_code_execution_eval_cfg,
)

# TestOuputput Dataset
lcb_test_output_reader_cfg = dict(
    input_columns=[
        'prompt',
    ],
    output_column='evaluation_sample',
)

system_prompt = 'You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.'

lcb_test_output_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            # begin=[
            #     dict(
            #         role='SYSTEM',
            #         prompt=system_prompt
            #     ),
            # ],
            round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}'
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

lcb_test_output_eval_cfg = dict(
    evaluator=dict(
        type=LCBTestOutputEvaluator,
    ),
    pred_role='BOT',
)

LCBTestOutput_dataset = dict(
    type=LCBTestOutputPredictionDataset,
    abbr='lcb_test_output',
    path='opencompass/test_generation',
    reader_cfg=lcb_test_output_reader_cfg,
    infer_cfg=lcb_test_output_infer_cfg,
    eval_cfg=lcb_test_output_eval_cfg,
)

LCB_datasets = [
    LCBCodeGeneration_dataset,
    # LCBCodeExecution_dataset,
    # LCBTestOutput_dataset,
]
