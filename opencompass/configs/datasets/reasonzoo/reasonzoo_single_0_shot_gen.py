from opencompass.datasets.reasonzoo.reasonzoo import reasonzooDataset, reasonzooEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

categories = ['communication_code', 'number_calculation', 'gradeschoolmath', 'formal_language', 'operation_research', 'puzzle_and_code','physics','dailylogic','boolean_logic'] 

reasonzoo_0shot_single_datasets = []

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
        inferencer=dict(type=GenInferencer),
    )

    # Evaluation configuration
    eval_cfg = dict(
        evaluator=dict(type=reasonzooEvaluator),
        pred_role='BOT',
    )

    reasonzoo_dataset = dict(
        type=reasonzooDataset,
        abbr=f'reasonzoo_{category}',
        path='opencompass/reasonzoo',
        prompt_mode='0_shot',
        category=category,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )

    reasonzoo_0shot_single_datasets.append(reasonzoo_dataset)
