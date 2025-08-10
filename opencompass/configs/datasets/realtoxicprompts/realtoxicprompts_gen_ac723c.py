from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import ToxicEvaluator
from opencompass.datasets import RealToxicPromptsDataset

realtoxicprompts_reader_cfg = dict(
    input_columns=['prompt_text'],
    output_column='filename',
    train_split='train',
    test_split='train')

# TODO: allow empty output-column
realtoxicprompts_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt_text}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

# When key is set to "ENV", the key will be fetched from the environment
# variable $PerspectiveAPIkey. Otherwise, set key in here directly.
realtoxicprompts_eval_cfg = dict(
    evaluator=dict(type=ToxicEvaluator, key='ENV'), )

realtoxicprompts_datasets = [
    dict(
        abbr='real-toxicity-prompts',
        type=RealToxicPromptsDataset,
        path='data/realtoxicprompts/realtoxicprompts_train.arrow',
        challenging_subset=True,
        reader_cfg=realtoxicprompts_reader_cfg,
        infer_cfg=realtoxicprompts_infer_cfg,
        eval_cfg=realtoxicprompts_eval_cfg)
]
