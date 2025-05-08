from opencompass.datasets import MedCalc_BenchDataset, MedCalcOfficial_Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

ZERO_SHOT_PROMPT = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}. \n Here is the patient note:\n{patient_note}\n\nHere is the task:\n{question}\n\nPlease directly output the JSON dict formatted as {"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}:'
# Reader configuration
reader_cfg = dict(
        input_columns=[
        'row_number',
        'calculator_id',
        'calculator_name',
        'category',
        'note_id',
        'output_type',
        'note_type',
        'patient_note',
        'question',
        'relevant_entities',
        'ground_truth_answer',
        'lower_limit',
        'upper_limit',
        'ground_truth_explanation'
    ],
    output_column='ground_truth_answer',
)


# Inference configuration
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
            dict(role='HUMAN',prompt=ZERO_SHOT_PROMPT),
        ])
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# Evaluation configuration
eval_cfg = dict(
    evaluator=dict(type=MedCalcOfficial_Evaluator),
    pred_role='BOT',
)
medcal_bench_dataset = dict(
    type=MedCalc_BenchDataset,
    abbr='medcal_bench_official_zero_shot_eval',
    path='ncbi/MedCalc-Bench-v1.0',
    prompt_mode='zero-shot',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)

medcal_bench_datasets = [medcal_bench_dataset]
