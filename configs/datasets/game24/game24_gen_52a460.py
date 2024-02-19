from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ToTInferencer
from opencompass.datasets import (Game24Dataset, game24_postprocess,
                                  Game24Evaluator, Game24PromptWrapper)

generation_kwargs = dict(do_sample=False, temperature=0.7)

game24_reader_cfg = dict(
    input_columns=['input'],
    output_column='output')

game24_infer_cfg = dict(
        prompt_template=dict(
        type=PromptTemplate,
        template='{input}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ToTInferencer, generation_kwargs=generation_kwargs, method_generate='propose',
                    method_evaluate='value', method_select='greedy', n_evaluate_sample=3, n_select_sample=5, prompt_wrapper=dict(type=Game24PromptWrapper)))

game24_eval_cfg = dict(
    evaluator=dict(type=Game24Evaluator),
    pred_postprocessor=dict(type=game24_postprocess),
)

game24_datasets = [
    dict(
        abbr='game24',
        type=Game24Dataset,
        path='./data/game24/game24.csv',
        reader_cfg=game24_reader_cfg,
        infer_cfg=game24_infer_cfg,
        eval_cfg=game24_eval_cfg)
]
