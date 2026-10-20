from opencompass.datasets.korbench.korbench import korbenchDataset, korbenchEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
korbench_mixed_datasets = []

categories = ['Multi-Q', 'Multi-R', 'Multi-RQ']  # Define available modes for mixed mode

for category in categories:
    # Prompt template
    prompt_template = dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='HUMAN',
                    prompt=''
                )
            ],
            round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}' # f-string
                )
            ]
        )
    )

    # Reader configuration
    reader_cfg = dict(
        input_columns=['prompt'],
        output_column='answer',
    )

    # Inference configuration
    infer_cfg = dict(
        prompt_template=prompt_template,
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=1024),
    )

    # Evaluation configuration
    eval_cfg = dict(
        evaluator=dict(type=korbenchEvaluator),
        pred_role='BOT',
    )

    korbench_dataset = dict(
        type=korbenchDataset,
        abbr=f'korbench_mixed_{category}',
        path='opencompass/korbench',
        category=category,
        prompt_mode='mixed',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )

    korbench_mixed_datasets.append(korbench_dataset)