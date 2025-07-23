from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import GAIADataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

gaia_reader_cfg = dict(
    input_columns=['question', 'file_path'],
    output_column='answerKey',
    test_split='test')

gaia_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                '''You are a general AI assistant. I will ask you a question. Report your thoughts, and
finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated
list of numbers and/or strings.
If you are asked for a number, don’t use comma to write your number neither use units such as $ or
percent sign unless specified otherwise.
If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the
digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element
to be put in the list is a number or a string.\nGAIA Question: {question}\nFile Path: {file_path}\n'''
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
gaia_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

gaia_datasets = [
    dict(
        abbr='gaia-validation',
        type=GAIADataset,
        path='opencompass/gaia',
        local_mode=False,
        reader_cfg=gaia_reader_cfg,
        infer_cfg=gaia_infer_cfg,
        eval_cfg=gaia_eval_cfg,
    )
]
