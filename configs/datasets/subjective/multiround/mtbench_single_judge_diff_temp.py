from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer, GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import MTBenchDataset
from opencompass.summarizers import MTBenchSummarizer

subjective_reader_cfg = dict(
    input_columns=['dialogue', 'capability', 'system_prompt', 'prompt_template'],
    output_column='judge',
    )

subjective_all_sets = [
    'mtbench_0.0','mtbench_0.1','mtbench_0.7'
]
data_path ='data/subjective/mtbench'

mtbench_datasets = []

for _name in subjective_all_sets:
    temperature = float(_name.split('_')[1])
    do_sample = False if temperature == 0.0 else True
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template="""{dialogue}""",
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=ChatInferencer, max_seq_len=4096, max_out_len=1024, temperature=temperature, do_sample=do_sample,infer_mode='every'),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt='{system_prompt}')
                ],
                    round=[
                    dict(
                        role='HUMAN',
                        prompt = '{prompt_template}'
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    mtbench_datasets.append(
        dict(
            abbr=f'{_name}',
            type=MTBenchDataset,
            path=data_path,
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='singlescore',
            summarizer = dict(type=MTBenchSummarizer, judge_type='single')
        ))
