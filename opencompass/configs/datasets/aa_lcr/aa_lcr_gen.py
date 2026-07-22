from opencompass.datasets import AALCRDataset, aa_lcr_llmjudge_postprocess
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


AA_LCR_JUDGE_TEMPLATE = """Assess whether the following CANDIDATE ANSWER is \
CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the \
OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {answer}
CANDIDATE ANSWER TO ASSESS: {prediction}

Reply only with CORRECT or INCORRECT."""


aa_lcr_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

aa_lcr_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[dict(role='user', content='{prompt}')],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

aa_lcr_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[dict(role='user', content=AA_LCR_JUDGE_TEMPLATE)],
        ),
        dataset_cfg=dict(
            type=AALCRDataset,
            path='ArtificialAnalysis/AA-LCR',
            reader_cfg=aa_lcr_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=aa_lcr_llmjudge_postprocess),
    ),
    pred_role='BOT',
)

aa_lcr_datasets = [
    dict(
        abbr='AA-LCR',
        type=AALCRDataset,
        path='ArtificialAnalysis/AA-LCR',
        reader_cfg=aa_lcr_reader_cfg,
        infer_cfg=aa_lcr_infer_cfg,
        eval_cfg=aa_lcr_eval_cfg,
    )
]
