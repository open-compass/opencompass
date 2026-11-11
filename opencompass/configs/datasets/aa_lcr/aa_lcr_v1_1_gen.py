from opencompass.datasets import AALCRDataset, aa_lcr_llmjudge_postprocess
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

DATASET_REVISION = '9a77ef56b717057ade24ceab4d273712a0b4f19e'

AA_LCR_V1_1_JUDGE_SYSTEM_PROMPT = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given.
"""

AA_LCR_V1_1_JUDGE_TEMPLATE = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{prediction}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT.
"""

aa_lcr_v1_1_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

aa_lcr_v1_1_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[dict(role='user', content='{prompt}')],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

aa_lcr_v1_1_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                dict(role='system', content=AA_LCR_V1_1_JUDGE_SYSTEM_PROMPT),
                dict(role='user', content=AA_LCR_V1_1_JUDGE_TEMPLATE),
            ],
        ),
        dataset_cfg=dict(
            type=AALCRDataset,
            path='opencompass/AA-LCR-v1.1',
            revision=DATASET_REVISION,
            reader_cfg=aa_lcr_v1_1_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=aa_lcr_llmjudge_postprocess),
    ),
    pred_role='BOT',
)

aa_lcr_v1_1_datasets = [
    dict(
        abbr='AA-LCR-v1.1',
        type=AALCRDataset,
        path='opencompass/AA-LCR-v1.1',
        revision=DATASET_REVISION,
        reader_cfg=aa_lcr_v1_1_reader_cfg,
        infer_cfg=aa_lcr_v1_1_infer_cfg,
        eval_cfg=aa_lcr_v1_1_eval_cfg,
        n=3,
    )
]
