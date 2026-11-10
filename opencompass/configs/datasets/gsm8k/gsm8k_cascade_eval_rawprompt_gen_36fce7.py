from opencompass.datasets import (GSM8KDataset, generic_llmjudge_postprocess,
                                  gsm8k_dataset_postprocess,
                                  gsm8k_postprocess)
from opencompass.evaluator import (CascadeEvaluator, GenericLLMEvaluator,
                                   MATHVerifyEvaluator)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(
                role='user',
                content='{question}\nPlease reason step by step, and put '
                'your final answer within \\boxed{}.',
            )
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

GRADER_TEMPLATE = """
Please as a grading expert, judge whether the final answer given by the
candidate below is consistent with the standard answer, that is, whether the
candidate answered correctly.

Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the
answer to the question because the standard answer has been given. You only need
to judge whether the candidate's answer is consistent with the standard answer
according to the form of the question.
2. Because the candidate's answer may be different from the standard answer in
the form of expression, before making a judgment, please understand the question
and the standard answer first, and then judge whether the candidate's answer is
correct.
3. Some answers may be expressed in different ways, such as with commas,
currency symbols, units, or explanatory text, as long as the meaning expressed is
the same.
4. If the prediction contains the correct final answer but the rule evaluator
extracts another intermediate number, the prediction should still be judged
correct.

Please judge whether the following answer is consistent with the standard
answer. Just return "A" or "B", with no text around it.

A: CORRECT
B: INCORRECT

<Original Question Begin>:
{question}
<Original Question End>

<Gold Target Begin>:
{answer}
<Gold Target End>

<Predicted Answer Begin>:
{prediction}
<Predicted Answer End>
""".strip()

cascade_evaluator = dict(
    type=CascadeEvaluator,
    rule_evaluator=dict(
        type=MATHVerifyEvaluator,
        pred_postprocessor=dict(type=gsm8k_postprocess),
    ),
    llm_evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                dict(
                    role='system',
                    content='You are a helpful assistant who evaluates the '
                    "correctness of models' outputs.",
                ),
                dict(role='user', content=GRADER_TEMPLATE),
            ],
        ),
        dataset_cfg=dict(
            type=GSM8KDataset,
            path='opencompass/gsm8k',
            reader_cfg=gsm8k_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    parallel=False,
)

gsm8k_eval_cfg = dict(
    evaluator=cascade_evaluator,
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)

gsm8k_datasets = [
    dict(
        abbr='gsm8k-cascade',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
