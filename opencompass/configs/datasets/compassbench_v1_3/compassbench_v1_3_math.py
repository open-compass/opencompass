from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator, AccEvaluator
from opencompass.datasets.compassbench_obj import CompassBenchObjectiveV1_3
from opencompass.datasets import MATHEvaluator, math_postprocess_v2
from opencompass.utils.text_postprocessors import first_option_postprocess


prompt_cn = {
    'single_choice_cn': '以下是一道单项选择题，请你根据你了解的知识给出正确的答案选项。请你一步步推理并在最后用“答案选项为X”来回答，其中X是ABCD中你认为正确的选项序号\n下面是你要回答的题目：\n{question}\n让我们一步步解决这个问题：',
    'cloze_cn': '以下是一道数学计算题，请你一步一步计算，并在最后用\\boxed{}包裹并返回你计算的最终答案。\n下面是你要回答的题目：\n{question}\n让我们一步步解决这个问题：',
}

prompt_en = {
    'single_choice_en': "Here is a single-choice question. Please give the correct answer based on your knowledge. Please reason step by step and answer with 'The answer is X' at the end, where X is the option number you think is correct.\nHere is the question you need to answer:\n{question}\nLet's solve this problem step by step:",
    'cloze_en': 'Here is a arithematic problem. Please reason step by step, and put your final answer within \\boxed{}. Here is the question you need to answer:\n{question}\nLet\'s solve this problem step by step:',
}


douknow_sets = {
    'arithmetic_cloze_en': ['cloze_en'],
    'college_single_choice_en': ['single_choice_en'],
    'college_single_choice_cn': ['single_choice_cn'],
}

data_path = './data/compassbench_v1_3/math'

# Set up the prompts
CircularEval = True

compassbench_math_datasets = []

for _split in list(douknow_sets.keys()):
    for _name in douknow_sets[_split]:
        if 'cn' in _name:
            single_choice_prompts = prompt_cn
            cloze_prompts = prompt_cn
        else:
            single_choice_prompts = prompt_en
            cloze_prompts = prompt_en

        if 'single_choice' in _name:
            template_round = [dict(role='HUMAN', prompt=single_choice_prompts[_name])]
            pred_postprocessor = dict(type=first_option_postprocess, options='ABCD')
            evaluator = dict(type=CircularEvaluator if CircularEval else AccEvaluator)
            dataset_name = _name + '_circular' if CircularEval else _name
            dataset_abbr = (
                'compassbench-' + _split + '_circular'
                if CircularEval
                else 'compassbench-' + _split
            )
        else:
            template_round = [dict(role='HUMAN', prompt=cloze_prompts[_name])]
            pred_postprocessor = dict(
                type=math_postprocess_v2,
            )
            evaluator = dict(type=MATHEvaluator)
            dataset_name = _name
            dataset_abbr = 'compassbench-' + _split

        douknow_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate, template=dict(round=template_round)
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=2048),
        )

        douknow_eval_cfg = dict(
            evaluator=evaluator,
            pred_postprocessor=pred_postprocessor,
        )

        compassbench_math_datasets.append(
            dict(
                type=CompassBenchObjectiveV1_3,
                path=f'{data_path}/{_split}.jsonl',
                name=dataset_name,
                abbr=dataset_abbr,
                reader_cfg=dict(input_columns=['question'], output_column='answer'),
                infer_cfg=douknow_infer_cfg,
                eval_cfg=douknow_eval_cfg,
            )
        )
del _split, _name
