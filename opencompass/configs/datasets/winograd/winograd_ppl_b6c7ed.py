from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WinogradDataset

winograd_reader_cfg = dict(
    input_columns=['prompt', 'pronoun', 'opt1', 'opt2'],
    output_column='label',
    train_split='test',
    test_split='test')

winograd_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            i: dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f"{{prompt}} Q: In the previous text, what does '{{pronoun}}' refer to? A: {{opt{i+1}}}"
                ),  # noqa
            ])
            for i in range(2)
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

winograd_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

winograd_datasets = [
    dict(
        abbr='winograd',
        type=WinogradDataset,
        path='winograd_wsc',
        trust_remote_code=True,
        name='wsc285',
        reader_cfg=winograd_reader_cfg,
        infer_cfg=winograd_infer_cfg,
        eval_cfg=winograd_eval_cfg)
]
