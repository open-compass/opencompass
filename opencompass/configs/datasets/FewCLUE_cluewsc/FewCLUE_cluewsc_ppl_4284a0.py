from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CluewscDataset

cluewsc_reader_cfg = dict(
    input_columns=['span1', 'span2', 'text', 'new_text'],
    output_column='answer')

cluewsc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0:
            "{text}\nHere, is the pronoun \"{span2}\" used to mean \"{span1}\"? No.",
            1:
            "{text}\nHere, is the pronoun \"{span2}\" used to mean \"{span1}\"? Yes.",
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

cluewsc_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

cluewsc_datasets = [
    dict(
        type=CluewscDataset,
        path='json',
        abbr='cluewsc-dev',
        data_files='./data/FewCLUE/cluewsc/dev_few_all.json',
        split='train',
        reader_cfg=cluewsc_reader_cfg,
        infer_cfg=cluewsc_infer_cfg,
        eval_cfg=cluewsc_eval_cfg),
    dict(
        type=CluewscDataset,
        path='json',
        abbr='cluewsc-test',
        data_files='./data/FewCLUE/cluewsc/test_public.json',
        split='train',
        reader_cfg=cluewsc_reader_cfg,
        infer_cfg=cluewsc_infer_cfg,
        eval_cfg=cluewsc_eval_cfg),
]
