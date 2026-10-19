from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MathBenchDataset, mathbench_postprocess

cloze_prompts ={
    'cloze_arith_en': [
        dict(role='HUMAN', prompt='Q: Calculate (341/11)/(9/(-6)*(-2)/3).'),
        dict(role='BOT', prompt='A: First, (9/(-6)*(-2)/3) can be simplified by : 9/(-6) = -1.5, -1.5 * (-2) = 3, 3 / 3 = 1. So, (9/(-6)*(-2)/3) is equal to 1. Now, we have `(341/11)/1` equals `341/11`. Finally, calculate `341/11 = 31`. The answer is 31.\n'),
        dict(role='HUMAN', prompt='Q: In base 14, what is 5 - 638d8d?'),
        dict(role='BOT', prompt='A: 5 - 638d8d = -638d88. The answer is -638d88.\n'),
        dict(role='HUMAN', prompt='Q: What is -491354 times -0.34?'),
        dict(role='BOT', prompt='A: The product of -491354 and -0.34 is 167060.36. The answer is 167060.36.\n'),
        dict(role='HUMAN', prompt='Q: What is the value of (-55)/(6930/(-382)) + (0 - 3)?.'),
        dict(role='BOT', prompt='A: First, (-55)/(6930/(-382)) = (-55)/(-(6930/382)) = 55*382/6930 = 21010/6930 = 2101/693. Then, 2101/693 + (0 - 3) = 2101/693 - 3 = 2101/693 - 3*693/693 = (2101-2079)/693 = 22/693 = 2/63. The answer is 2/63.\n'),
        dict(role='HUMAN', prompt='Q: {question}'),
        dict(role='BOT', prompt='A: {answer}\n'),
    ]
}

mathbench_sets = {
    'arithmetic': ['cloze_arith_en'],
}

mathbench_datasets = []

for _split in list(mathbench_sets.keys()):
    for _name in mathbench_sets[_split]:
        mathbench_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=cloze_prompts[_name],
                    ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=512),
        )

        mathbench_eval_cfg = dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=mathbench_postprocess, name=_name))

        mathbench_datasets.append(
            dict(
                type=MathBenchDataset,
                path=f'./data/mathbench/{_split}',
                name=_name,
                with_circular=False,
                abbr='mathbench-arithmetic' + _split + '-' + _name,
                reader_cfg=dict(
                    input_columns=['question'],
                    output_column='answer'
                    ),
                infer_cfg=mathbench_infer_cfg,
                eval_cfg=mathbench_eval_cfg,
            ))
