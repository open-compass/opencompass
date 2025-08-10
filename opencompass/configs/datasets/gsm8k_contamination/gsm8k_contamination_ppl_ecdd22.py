from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLOnlyInferencer
from opencompass.openicl.icl_evaluator import AveragePPLEvaluator
from opencompass.datasets import GSM8KDataset, GSM8KReferenceSkywork

gsm8k_datasets = []

gsm8k_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{question} {answer}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLOnlyInferencer),
)

gsm8k_eval_cfg = dict(evaluator=dict(type=AveragePPLEvaluator))

for split in ['train', 'test']:
    gsm8k_reader_cfg = dict(
        input_columns=['question', 'answer'],
        output_column=None,
        train_split=split,
        test_split=split,
    )
    gsm8k_datasets.append(
        dict(
            abbr=f'gsm8k-{split}-ppl',
            type=GSM8KDataset,
            path='./data/gsm8k',
            reader_cfg=gsm8k_reader_cfg,
            infer_cfg=gsm8k_infer_cfg,
            eval_cfg=gsm8k_eval_cfg)
    )


gsm8k_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{text}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLOnlyInferencer),
)

gsm8k_eval_cfg = dict(evaluator=dict(type=AveragePPLEvaluator))

gsm8k_reader_cfg = dict(
    input_columns=['text'],
    output_column=None,
)

gsm8k_datasets.append(
    dict(
        abbr=f'gsm8k-ref-ppl',
        type=GSM8KReferenceSkywork,
        path='./data/gsm8k-extra/mock_gsm8k_test.jsonl',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg
    )
)
