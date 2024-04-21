from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TheoremQADatasetV3, TheoremQA_postprocess_v3, TheoremQAEvaluatorV3

with read_base():
    from .TheoremQA_few_shot_examples import examples

num_shot = 5
prompt = ""
for index, (query, response) in enumerate(examples[:num_shot]):
    prompt += f'[INST] {query} [/INST]{response}\n\n'
prompt += '[INST] {Question} [/INST]'

TheoremQA_reader_cfg = dict(input_columns=["Question"], output_column="Answer")

TheoremQA_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=prompt),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048, stopping_criteria=["Problem:", "\n\nProblem"]),
)

TheoremQA_eval_cfg = dict(
    evaluator=dict(type=TheoremQAEvaluatorV3),
    pred_postprocessor=dict(type=TheoremQA_postprocess_v3)
)

TheoremQA_datasets = [
    dict(
        abbr="TheoremQA",
        type=TheoremQADatasetV3,
        path="data/TheoremQA/theoremqa_test.json",
        reader_cfg=TheoremQA_reader_cfg,
        infer_cfg=TheoremQA_infer_cfg,
        eval_cfg=TheoremQA_eval_cfg,
    )
]
