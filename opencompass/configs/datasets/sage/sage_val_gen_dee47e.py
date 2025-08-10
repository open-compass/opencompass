from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CustomDataset
from opencompass.models import OpenAISDK

from opencompass.datasets.sage.prompt import SAGE_INFER_TEMPLATE, SAGE_EVAL_TEMPLATE
from opencompass.datasets.sage.dataset_loader import SAGEDataset
from opencompass.datasets.sage.evaluation import SAGELLMEvaluator, sage_judge_postprocess, sage_pred_postprocess

compass_agi4s_reader_cfg = dict(
    input_columns=['problem'], 
    output_column='answer'
)

compass_agi4s_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=SAGE_INFER_TEMPLATE,
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

sage_datasets = [
    dict(
        type=SAGEDataset,
        n=4,
        abbr='sage-val',
        split='val',
        reader_cfg=compass_agi4s_reader_cfg,
        infer_cfg=compass_agi4s_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=SAGELLMEvaluator,
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        round=[
                            dict(role='HUMAN', prompt=SAGE_EVAL_TEMPLATE),
                        ],
                    ),
                ),
                dataset_cfg=dict(
                    type=SAGEDataset,
                    n=4,
                    abbr='sage-test',
                    split='test',
                    reader_cfg=compass_agi4s_reader_cfg,
                ),
                judge_cfg=dict(
                    judgers=[
                        dict(
                            type=OpenAISDK,
                            abbr='xxx',
                            openai_api_base='xxx',
                            path='xxx',
                            key='YOUR_API_KEY',
                            meta_template=dict(
                                reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM'),],
                                round=[
                                    dict(role='HUMAN', api_role='HUMAN'),
                                    dict(role='BOT', api_role='BOT', generate=True),
                                ]
                            ),
                            query_per_second=128,
                            max_seq_len=40960,
                            max_out_len=32768,
                            tokenizer_path='o3',
                            temperature=0.6,
                            batch_size=128,
                            retry=16,
                            run_cfg=dict(num_gpus=0)
                        ),
                    ],
                    num_gpus=0,
                ),
                pred_postprocessor=dict(
                    type=sage_pred_postprocess,
                    think_tags=('<think>', '</think>'),
                ),
                dict_postprocessor=dict(
                    type=sage_judge_postprocess,
                    think_tags=('<think>', '</think>'),
                )
            ),
        ),
    )
]

datasets = sage_datasets