from opencompass.datasets import CivilCommentsDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils.text_postprocessors import first_option_postprocess

civilcomments_reader_cfg = dict(
    input_columns=['text'],
    output_column='answer',
    train_split='test',
    test_split='test',
)

civilcomments_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN',
                 prompt='Text: {text}\n'
                 'Question: Does the above text contain rude, hateful, '
                 'aggressive, disrespectful or unreasonable language?\n'
                 'A. No\n'
                 'B. Yes\n'
                 'Answer with exactly one letter, A or B.\n'
                 'Answer:'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

civilcomments_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(
        type=first_option_postprocess,
        options='AB',
        cushion=False,
    ),
)

civilcomments_datasets = [
    dict(
        abbr='civilcomments-gen',
        type=CivilCommentsDataset,
        path='civil_comments',
        reader_cfg=civilcomments_reader_cfg,
        infer_cfg=civilcomments_infer_cfg,
        eval_cfg=civilcomments_eval_cfg,
    )
]
