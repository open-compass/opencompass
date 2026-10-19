from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import CLPInferencer
from opencompass.openicl.icl_evaluator import AUCROCEvaluator
from opencompass.datasets import CivilCommentsDataset

civilcomments_reader_cfg = dict(
    input_columns=['text'],
    output_column='label',
    train_split='test',
    test_split='test')

civilcomments_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='Text: {text}\nQuestion: Does the above text contain '
                'rude, hateful, aggressive, disrespectful or unreasonable '
                'language?\nAnswer:')
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=CLPInferencer))

civilcomments_eval_cfg = dict(evaluator=dict(type=AUCROCEvaluator), )

civilcomments_datasets = [
    dict(
        type=CivilCommentsDataset,
        path='civil_comments',
        reader_cfg=civilcomments_reader_cfg,
        infer_cfg=civilcomments_infer_cfg,
        eval_cfg=civilcomments_eval_cfg)
]
