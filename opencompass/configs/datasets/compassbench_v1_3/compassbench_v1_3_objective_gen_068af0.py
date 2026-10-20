from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator, AccEvaluator
from opencompass.datasets.compassbench_obj import CompassBenchObjectiveV1_3, compassbench_objective_v1_3_postprocess
from opencompass.utils.text_postprocessors import first_option_postprocess


prompt_cn = {
    'single_choice_cn': '以下是一道单项选择题，请你根据你了解的知识给出正确的答案选项。请你一步步推理并在最后用“答案选项为X”来回答，其中X是ABCD中你认为正确的选项序号\n下面是你要回答的题目：\n{question}\n让我们一步步解决这个问题：',
    'cloze_cn': '以下是一道填空题，请你根据你了解的知识一步步思考后把你的最终答案放到\\boxed{}中。\n下面是你要回答的题目：\n{question}\n让我们一步步解决这个问题：',
}

prompt_en = {
    'single_choice_en': "Here is a single-choice question. Please give the correct answer based on your knowledge. Please reason step by step and answer with 'The answer is X' at the end, where X is the option number you think is correct.\nHere is the question you need to answer:\n{question}\nLet's solve this problem step by step:",
    'cloze_en': "Here is a fill-in-the-blank question. Please think step by step based on your knowledge and put your final answer in \\boxed{}. Here is the question you need to answer:\n{question}\nLet's solve this problem step by step:",
}


douknow_sets = {
    'knowledge': ['single_choice_cn'],
    'math': ['single_choice_cn'],
}

# Set up the prompts
CircularEval = True


compassbench_aug_datasets = []

for _split in list(douknow_sets.keys()):
    for _name in douknow_sets[_split]:
        if 'cn' in _name:
            single_choice_prompts = prompt_cn
            cloze_prompts = prompt_cn
        else:
            single_choice_prompts = prompt_en
            cloze_prompts = prompt_en
        douknow_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin='</E>',
                    round=[
                        dict(
                            role='HUMAN',
                            prompt= single_choice_prompts[_name],
                        ),
                        dict(role='BOT', prompt='{answer}'),] if 'choice' in _name else cloze_prompts[_name],
                    ),
                ice_token='</E>',
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )
        douknow_eval_cfg = dict(
            evaluator=dict(type=CircularEvaluator if CircularEval else AccEvaluator) if 'single_choice' in _name else dict(type=AccEvaluator),
            pred_postprocessor=dict(type=first_option_postprocess, options='ABCD' ) if 'single_choice' in _name else dict(type=compassbench_objective_v1_3_postprocess, name=_name))

        compassbench_aug_datasets.append(
            dict(
                type=CompassBenchObjectiveV1_3,
                path=f'./data/compassbench_v1_3/{_split}/{_name}.jsonl',
                name='circular_' + _name if CircularEval else _name,
                abbr='compassbench-' + _split + '-' + _name + 'circular'if CircularEval else '',
                reader_cfg=dict(
                    input_columns=['question'],
                    output_column='answer'
                    ),
                infer_cfg=douknow_infer_cfg,
                eval_cfg=douknow_eval_cfg,
            ))

del _split, _name
