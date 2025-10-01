# THIS SHALL ALSO BE DEPRECATED
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import xHumanevalDataset, HumanEvalEvaluator, humaneval_postprocess_v2, HumanEvalPlusEvaluator

humaneval_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test')

# TODO: allow empty output-column
humaneval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='Complete the following python code:\n{prompt}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

humaneval_eval_cfg = dict(
    evaluator=dict(type=HumanEvalEvaluator),
    pred_role='BOT',
    k=[1, 10, 100],  # the parameter only for humaneval
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)

# humaneval_datasets = [
#     dict(
#         abbr='openai_humaneval',
#         type=xHumanevalDataset,
#         path='opencompass/humaneval',
#         reader_cfg=humaneval_reader_cfg,
#         infer_cfg=humaneval_infer_cfg,
#         eval_cfg=humaneval_eval_cfg)
# ]
LANGS = ['ar']
humaneval_datasets = []
for lang in LANGS:
    humaneval_datasets.append(
        dict(
            abbr=f'humaneval_{lang}',
            type=xHumanevalDataset,
            path=f'data/xhumaneval_plus/humaneval_plus_gpt4o_{lang}.jsonl',
            reader_cfg=humaneval_reader_cfg,
            infer_cfg=humaneval_infer_cfg,
            eval_cfg=humaneval_eval_cfg)
    )
