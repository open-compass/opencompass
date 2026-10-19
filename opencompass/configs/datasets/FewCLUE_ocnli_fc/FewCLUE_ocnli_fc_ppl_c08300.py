from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

ocnli_fc_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',
    test_split='train')

ocnli_fc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'contradiction':
            '阅读文章：{sentence1}\n根据上文，回答如下问题： {sentence2}？\n答：错',
            'entailment': '阅读文章：{sentence1}\n根据上文，回答如下问题： {sentence2}？\n答：对',
            'neutral': '如果{sentence1}为真，那么{sentence2}也为真吗?可能'
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))
ocnli_fc_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

ocnli_fc_datasets = [
    dict(
        type=HFDataset,
        abbr='ocnli_fc-dev',
        path='json',
        split='train',
        data_files='./data/FewCLUE/ocnli/dev_few_all.json',
        reader_cfg=ocnli_fc_reader_cfg,
        infer_cfg=ocnli_fc_infer_cfg,
        eval_cfg=ocnli_fc_eval_cfg),
    dict(
        type=HFDataset,
        abbr='ocnli_fc-test',
        path='json',
        split='train',
        data_files='./data/FewCLUE/ocnli/test_public.json',
        reader_cfg=ocnli_fc_reader_cfg,
        infer_cfg=ocnli_fc_infer_cfg,
        eval_cfg=ocnli_fc_eval_cfg)
]
