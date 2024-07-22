from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer, GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CRBDataset
from opencompass.summarizers import CRBBenchPairSummarizer


subjective_reader_cfg = dict(
    input_columns=['question', 'prompt'],
    output_column='judge',
    )


data_paths = ['data/crbbench/crbbench.jsonl', 'data/crbbench/cn_crbbench.jsonl']
abbrs = ['crbbench', 'cn_crbbench']
crb_datasets = []
gpt4 = [dict(
    abbr='gpt-4o',
)]

for data_path, abbr in zip(data_paths, abbrs):
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                        dict(
                            role='HUMAN',
                            prompt='{question}'
                        ),
                    ]),
                ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_seq_len=4096, max_out_len=4096),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template="""{prompt}"""
            ),
        ),
        pred_role='BOT',
    )

    crb_datasets.append(
        dict(
            abbr=abbr,
            type=CRBDataset,
            path=data_path,
            eval_mode='pair',
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='m2n',
            infer_order='random',
            base_models=gpt4,
            given_pred = [{'abbr': 'gpt-4o', 'path':'data/crbbench/gpt_4o'}]
        ))
