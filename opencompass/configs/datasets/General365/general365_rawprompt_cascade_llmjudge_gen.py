from opencompass.datasets import (General365Dataset,
                                  general365_llmjudge_postprocess)
from opencompass.evaluator import (CascadeEvaluator, GenericLLMEvaluator,
                                   MATHVerifyEvaluator)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

DATASET_PATH = 'meituan-longcat/General365_Public'

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for question answering systems.
Your task is to determine if a prediction correctly answers a question based on the ground truth.

Rules:
1. The prediction is correct if it captures all the key information from the ground truth.
2. The prediction is correct even if phrased differently as long as the meaning is the same.
3. The prediction is incorrect if it contains incorrect information or is missing essential details.
4. Do not challenge the correctness of the standard answer.
5. If the standard answer includes multiple possibilities, the prediction must include all and only those possibilities to be considered correct.
Output a JSON object with a single field 'accuracy' whose value is true or false."""

JUDGE_PROMPT = """Question: {question}
Ground truth: {answer}
Prediction: {prediction}"""

reader_cfg = dict(
    input_columns=['question', 'answer_type', 'float_round', 'id'],
    output_column='answer',
)

infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[dict(role='user', content='{question}')],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


def _llm_evaluator(dataset_cfg):
    return dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                dict(role='system', content=JUDGE_SYSTEM_PROMPT),
                dict(role='user', content=JUDGE_PROMPT),
            ],
        ),
        dataset_cfg=dataset_cfg,
        judge_cfg=dict(),
        dict_postprocessor=dict(type=general365_llmjudge_postprocess),
    )


math_dataset_cfg = dict(
    type=General365Dataset,
    path=DATASET_PATH,
    answer_types=['number', 'interval'],
    reader_cfg=reader_cfg,
)

text_dataset_cfg = dict(
    type=General365Dataset,
    path=DATASET_PATH,
    answer_types=['text', 'single_choice', 'multiple_choice'],
    reader_cfg=reader_cfg,
)

general365_datasets = [
    dict(
        abbr='General365-mathverify',
        **math_dataset_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=CascadeEvaluator,
                rule_evaluator=dict(type=MATHVerifyEvaluator),
                llm_evaluator=_llm_evaluator(math_dataset_cfg),
                parallel=False,
            )),
    ),
    dict(
        abbr='General365-text',
        **text_dataset_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=dict(evaluator=_llm_evaluator(text_dataset_cfg)),
    ),
]
