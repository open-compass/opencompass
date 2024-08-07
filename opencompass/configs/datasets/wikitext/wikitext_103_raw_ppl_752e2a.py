from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset


wikitext_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={0: '{text}'}
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer), # need a new ppl inferencer
)

wikitext_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

wikitext_103_raw_datasets = []
for _split in ['validation', 'test']:

    wikitext_reader_cfg = dict(
        input_columns=['text'],
        output_column=None,
        train_split='train',
        test_split=_split,
    )

    wikitext_103_raw_datasets.append(
        dict(
            abbr=f'wikitext-103-raw-{_split}',
            type=HFDataset,
            path='wikitext',
            name='wikitext-103-raw-v1',
            reader_cfg=wikitext_reader_cfg,
            infer_cfg=wikitext_infer_cfg,
            eval_cfg=wikitext_eval_cfg,
        )
    )
