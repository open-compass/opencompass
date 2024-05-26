from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TruthfulQADataset, TruthfulQAEvaluator

truthfulqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='reference',
    train_split='validation',
    test_split='validation')

# TODO: allow empty output-column
truthfulqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='{question}')])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

# Metrics such as 'truth' and 'info' needs
# OPENAI_API_KEY with finetuned models in it.
# Please use your own finetuned openai model with keys and refers to
# the source code of `TruthfulQAEvaluator` for more details.
#
# If you cannot provide available models for 'truth' and 'info',
# and want to perform basic metric eval, please set
# `metrics=('bleurt', 'rouge', 'bleu')`

# When key is set to "ENV", the key will be fetched from the environment
# variable $OPENAI_API_KEY. Otherwise, set key in here directly.
truthfulqa_eval_cfg = dict(
    evaluator=dict(
        type=TruthfulQAEvaluator, metrics=('truth', 'info'), key='ENV'), )

truthfulqa_datasets = [
    dict(
        abbr='truthful_qa',
        type=TruthfulQADataset,
        path='truthful_qa',
        name='generation',
        reader_cfg=truthfulqa_reader_cfg,
        infer_cfg=truthfulqa_infer_cfg,
        eval_cfg=truthfulqa_eval_cfg)
]
