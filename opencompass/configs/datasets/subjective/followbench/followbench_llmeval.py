from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import FollowBenchDataset
from opencompass.summarizers import FollowBenchSummarizer

subjective_reader_cfg = dict(
    input_columns=['instruction', 'judge_prompt',],
    output_column='judge',
    )

subjective_all_sets = [
    'followbench_llmeval_cn', 'followbench_llmeval_en',
]
data_path ='data/subjective/followbench/converted_data'

followbench_llmeval_datasets = []

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{instruction}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = '{judge_prompt}'
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    followbench_llmeval_datasets.append(
        dict(
            abbr=f'{_name}',
            type=FollowBenchDataset,
            path=data_path,
            name=_name,
            mode='singlescore',
            cate='llm',
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            summarizer = dict(type=FollowBenchSummarizer,)
        ))
