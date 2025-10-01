# THIS SHALL ALSO BE DEPRECATED
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import xHumanevalDataset, HumanEvalPlusEvaluator, humaneval_postprocess_v2

humaneval_plus_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test')

# TODO: allow empty output-column
humaneval_plus_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='Complete the following python code:\n{prompt}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048))




# print('humaneval_plus_datasets ru')
# humaneval_plus_datasets = [
#     dict(
#         abbr='xhumaneval_plus_ru',
#         type=xHumanevalDataset,
#         path='xhumaneval_plus',
#         name='humaneval_plus_gpt4o_ru.jsonl',
#         local_mode=True,
#         reader_cfg=humaneval_plus_reader_cfg,
#         infer_cfg=humaneval_plus_infer_cfg,
#         eval_cfg=humaneval_plus_eval_cfg)
# ]

LANGS = ['de']

humaneval_plus_datasets = []
for lang in LANGS:
    humaneval_plus_eval_cfg = dict(
    evaluator=dict(type=HumanEvalPlusEvaluator),
        pred_role='BOT',
        k=[1, 10],  # the parameter only for humaneval
        lang=lang,
        pred_postprocessor=dict(type=humaneval_postprocess_v2),
    )
    humaneval_plus_datasets.append(
        dict(
            abbr=f'xhumaneval_plus_{lang}',
            type=xHumanevalDataset,
            path=f'data/xhumaneval_plus/humaneval_{lang}.jsonl',
            reader_cfg=humaneval_plus_reader_cfg,
            infer_cfg=humaneval_plus_infer_cfg,
            eval_cfg=humaneval_plus_eval_cfg)
    )
