from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import CompassBenchDataset
from opencompass.summarizers import CompassBenchSummarizer

subjective_reader_cfg = dict(
    input_columns=['question', 'judge_prompt'],
    output_column='judge',
    )

data_path ='data/subjective/compassbench'

compassbench_datasets = []

versions = ['CompassBenchV1.1']

gpt4 = [dict(
    abbr='gpt4-turbo',
)]

for version_abbr in versions:
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
            inferencer=dict(type=GenInferencer, max_seq_len=4096, max_out_len=2048),
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

    compassbench_datasets.append(
        dict(
            abbr=version_abbr,
            type=CompassBenchDataset,
            path=data_path,
            name=version_abbr,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='m2n',
            infer_order='double',
            base_models=gpt4,
            summarizer=dict(type=CompassBenchSummarizer, summary_type='half_add'),
            given_pred = [{'abbr':'gpt4-turbo', 'path':'./data/subjective/alpaca_eval/gpt4-turbo'}]
        ))
