from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

bustm_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',
    test_split='train')

bustm_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0:
            dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt='请判断以下两句话说的是否是一个意思：')
                ],
                round=[
                    dict(role='HUMAN', prompt='{sentence1}，{sentence2}'),
                    dict(role='BOT', prompt='两句话说的毫不相关。')
                ]),
            1:
            dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt='请判断以下两句话说的是否是一个意思：')
                ],
                round=[
                    dict(role='HUMAN', prompt='{sentence1}，{sentence2}'),
                    dict(role='BOT', prompt='两句话说是的一个意思。')
                ]),
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

bustm_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

bustm_datasets = [
    dict(
        type=HFDataset,
        abbr='bustm-dev',
        path='json',
        data_files='./data/FewCLUE/bustm/dev_few_all.json',
        split='train',
        reader_cfg=bustm_reader_cfg,
        infer_cfg=bustm_infer_cfg,
        eval_cfg=bustm_eval_cfg),
    dict(
        type=HFDataset,
        abbr='bustm-test',
        path='json',
        data_files='./data/FewCLUE/bustm/test_public.json',
        split='train',
        reader_cfg=bustm_reader_cfg,
        infer_cfg=bustm_infer_cfg,
        eval_cfg=bustm_eval_cfg)
]
