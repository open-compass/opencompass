from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator, AccEvaluator
from opencompass.datasets import WikiBenchDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


single_choice_prompts = {
    'single_choice_cn': '以下是一道单项选择题，请你根据你了解的知识给出正确的答案选项。\n下面是你要回答的题目：\n{question}\n答案选项：',
}

wikibench_sets = {
    'wiki': ['single_choice_cn'],
}

do_circular = True

wikibench_datasets = []

for _split in list(wikibench_sets.keys()):
    for _name in wikibench_sets[_split]:
        wikibench_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin='</E>',
                    round=[
                        dict(role='HUMAN', prompt=single_choice_prompts[_name]),
                        dict(role='BOT', prompt='{answer}'),
                    ],
                ),
                ice_token='</E>',
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )
        wikibench_eval_cfg = dict(
            evaluator=dict(type=CircularEvaluator if do_circular else AccEvaluator),
            pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
        )

        wikibench_datasets.append(
            dict(
                type=WikiBenchDataset,
                path=f'./data/WikiBench/{_name}.jsonl',
                name='circular_' + _name if do_circular else _name,
                abbr='wikibench-' + _split + '-' + _name + 'circular' if do_circular else '',
                reader_cfg=dict(
                    input_columns=['question'],
                    output_column='answer',
                ),
                infer_cfg=wikibench_infer_cfg,
                eval_cfg=wikibench_eval_cfg,
            )
        )
