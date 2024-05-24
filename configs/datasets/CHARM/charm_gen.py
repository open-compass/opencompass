import os
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CharmDataset, charm_rea_postprocess, CharmReaEvaluator

charm_reader_cfg = dict(input_columns=["input"], output_column="target")

charm_tasks = [
    'Chinese_Anachronisms_Judgment',
    'Chinese_Movie_and_Music_Recommendation',
    'Chinese_Natural_Language_Inference',
    'Chinese_Reading_Comprehension',
    'Chinese_Sequence_Understanding',
    'Chinese_Sport_Understanding',
    'Chinese_Time_Understanding',
    'Global_Anachronisms_Judgment',
    'Global_Movie_and_Music_Recommendation',
    'Global_Natural_Language_Inference',
    'Global_Reading_Comprehension',
    'Global_Sequence_Understanding',
    'Global_Sport_Understanding',
    'Global_Time_Understanding',
]

cot_prefix_dict_ZH = {
    'Direct': "",
    'ZH-CoT': "让我们一步一步来思考。",
    'EN-CoT': "Let's think step by step.",
    'XLT':
    """You should retell the request in English.
You should do the answer step by step to choose the right answer.
You should step-by-step answer the request.
You should tell me the answer in this format 'So the answer is'."""
} # yapf: disable
cot_prefix_dict_TransEn = {'Translate-EN': "Let's think step by step."}

data_dir = "./data/CHARM"

dataset_path_ZH = f"{data_dir}/reasoning"
fewshot_example_path_ZH = f"{data_dir}/few-shot-examples"

dataset_path_TransEn = f"{data_dir}/reasoning_Translate-EN"
fewshot_example_path_TransEn = f"{data_dir}/few-shot-examples_Translate-EN"

task_groups = {
    'ZH': (
        cot_prefix_dict_ZH,
        dataset_path_ZH,
        fewshot_example_path_ZH,
    ),
    'TransEn': (
        cot_prefix_dict_TransEn,
        dataset_path_TransEn,
        fewshot_example_path_TransEn,
    )
}

charm_rea_datasets = []

for group in task_groups.values():
    cot_prefix_dict, dataset_path, fewshot_example_path = group
    for _task in charm_tasks:
        for _cot, _cot_prefix in cot_prefix_dict.items():
            if _cot == 'XLT':
                prompt_template = "Follow the given examples and answer the question.\n{_hint}\n\n I want you to act as an commonsense reasoning expert for Chinese. \n Request: {{input}}\n"
            elif _cot == 'Translate-EN':
                prompt_template = "Follow the given examples and answer the question.\n{_hint}\n\nQ: {{input}}\nA: "
            else:
                prompt_template = "请按照给定的例子回答问题。\n{_hint}\n\nQ：{{input}}\nA："

            _fewshot_example_file = os.path.join(fewshot_example_path,
                                                 f'{_task}_{_cot}.txt')
            if not os.path.isfile(_fewshot_example_file):
                print(f"Hint file {_fewshot_example_file} not found, skip.")
                continue
            with open(_fewshot_example_file, 'r') as f:
                _hint = f.read()

            rea_infer_cfg = dict(prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(role="HUMAN",
                         prompt=prompt_template.format(_hint=_hint) +
                         _cot_prefix)
                ])),
                                 retriever=dict(type=ZeroRetriever),
                                 inferencer=dict(type=GenInferencer,
                                                 max_out_len=512))

            rea_eval_cfg = dict(
                evaluator=dict(type=CharmReaEvaluator),
                pred_role="BOT",
                pred_postprocessor=dict(type=charm_rea_postprocess),
                dataset_postprocessor=dict(type=charm_rea_postprocess))

            charm_rea_datasets.append(
                dict(type=CharmDataset,
                     path=dataset_path,
                     name=_task,
                     abbr='rea-' + _task + '_' + _cot,
                     reader_cfg=charm_reader_cfg,
                     infer_cfg=rea_infer_cfg.copy(),
                     eval_cfg=rea_eval_cfg.copy()))