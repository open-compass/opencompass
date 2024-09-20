from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import StoryClozeDatasetV2
from opencompass.utils.text_postprocessors import first_option_postprocess

storycloze_reader_cfg = dict(
    input_columns=['context', 'sentence_quiz1', 'sentence_quiz2'],
    output_column='answer_right_ending',
)

storycloze_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                "{context}\nQuestion: Which ending makes the most sense?\nA. {sentence_quiz1}\nB. {sentence_quiz2}\nYou may choose between 'A' and 'B'.\nAnswer:",
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

storycloze_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

# The original story cloze dataset and repo are not long maintaining.
# Using multilingual version of this dataset.
storycloze_datasets = [
    dict(
        abbr='story_cloze',
        type=StoryClozeDatasetV2,
        path='opencompass/xstory_cloze',
        lang='en',
        reader_cfg=storycloze_reader_cfg,
        infer_cfg=storycloze_infer_cfg,
        eval_cfg=storycloze_eval_cfg,
    )
]
