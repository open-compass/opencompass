from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import AlignmentBenchDataset
from opencompass.summarizers import AlignmentBenchSummarizer

subjective_reader_cfg = dict(
    input_columns=['question', 'capability', 'critiquellm_prefix'],
    output_column='judge',
    )

subjective_all_sets = [
    'alignment_bench_v1_1', # Changed to Alignbench_v1_1 since 06/15/2024, refer to https://github.com/THUDM/AlignBench
]
data_path ='data/subjective/alignment_bench'

alignment_bench_config_path = 'data/subjective/alignment_bench/config'
alignment_bench_config_name = 'multi-dimension'

alignbench_datasets = []

for _name in subjective_all_sets:
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
            inferencer=dict(type=GenInferencer, max_out_len=2048),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = '{critiquellm_prefix}[助手的答案开始]\n{prediction}\n[助手的答案结束]\n'
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    alignbench_datasets.append(
        dict(
            abbr=f'{_name}',
            type=AlignmentBenchDataset,
            path=data_path,
            name=_name,
            alignment_bench_config_path=alignment_bench_config_path,
            alignment_bench_config_name=alignment_bench_config_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='singlescore',
            summarizer = dict(type=AlignmentBenchSummarizer, judge_type='general')
        ))
