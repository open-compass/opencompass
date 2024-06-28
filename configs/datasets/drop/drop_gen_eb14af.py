from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import DropOpenAIDataset, DropOpenAIEvaluator

with read_base():
    from .drop_examples import drop_examples  # noqa: F401, F403

drop_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answers',
    train_split='validation',
    test_split='validation',
)

template = f'You will be asked to read a passage and answer a question. Think step by step, then write a line of the form "Answer: $ANSWER" at the end of your response. Some examples of passages and Q&A are provided below.\n\n{drop_examples}\n\n# Your Task\n\n---\n{{prompt}}'

drop_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=dict(round=[dict(role='HUMAN', prompt=template)])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

drop_eval_cfg = dict(evaluator=dict(type=DropOpenAIEvaluator))

drop_datasets = [
    dict(
        abbr='drop',
        type=DropOpenAIDataset,
        path='data/drop_simple_eval/dev.jsonl',
        reader_cfg=drop_reader_cfg,
        infer_cfg=drop_infer_cfg,
        eval_cfg=drop_eval_cfg)
]
