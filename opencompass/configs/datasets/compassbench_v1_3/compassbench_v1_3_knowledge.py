from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator, AccEvaluator
from opencompass.datasets.compassbench_obj import (
    CompassBenchObjectiveV1_3,
    compassbench_objective_v1_3_postprocess,
)
from opencompass.utils.text_postprocessors import first_option_postprocess


prompt_cn = {
    'single_choice_cn': '以下是一道单项选择题，请你根据你了解的知识给出正确的答案选项。请你一步步推理并在最后用“答案选项为X”来回答，其中X是ABCD中你认为正确的选项序号\n下面是你要回答的题目：\n{question}\n让我们一步步解决这个问题：',
    'cloze_cn': '以下是一道填空题，请你根据你了解的知识一步步思考后把你的最终答案放到\\boxed{}中。\n下面是你要回答的题目：\n{question}\n让我们一步步解决这个问题：',
}

prompt_en = {
    'single_choice_en': "Here is a single-choice question. Please give the correct answer based on your knowledge. Please reason step by step and answer with 'The answer is X' at the end, where X is the option letter you think is correct.\nHere is the question you need to answer:\n{question}\nLet's solve this problem step by step:",
    'cloze_en': "Here is a fill-in-the-blank question. Please think step by step based on your knowledge and put your final answer in \\boxed{}. Here is the question you need to answer:\n{question}\nLet's solve this problem step by step:",
}

douknow_sets = {
    'wiki_en_sub_500_人文科学':['single_choice_en'],
    'wiki_en_sub_500_社会科学':['single_choice_en'],
    'wiki_en_sub_500_生活常识':['single_choice_en'],
    'wiki_en_sub_500_自然科学-工科':['single_choice_en'],
    'wiki_en_sub_500_自然科学-理科':['single_choice_en'],
    'wiki_zh_sub_500_人文科学': ['single_choice_cn'],
    'wiki_zh_sub_500_社会科学': ['single_choice_cn'],
    'wiki_zh_sub_500_生活常识': ['single_choice_cn'],
    'wiki_zh_sub_500_自然科学-工科':['single_choice_cn'],
    'wiki_zh_sub_500_自然科学-理科':['single_choice_cn'],
}

data_path = './data/compassbench_v1_3/knowledge'

# Set up the prompts
CircularEval = True

compassbench_knowledge_datasets = []

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
                type=compassbench_objective_v1_3_postprocess, name=_name
            )
            evaluator = dict(type=AccEvaluator)
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

        compassbench_knowledge_datasets.append(
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
