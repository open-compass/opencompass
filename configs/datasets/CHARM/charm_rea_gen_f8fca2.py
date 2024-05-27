import os
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CharmDataset, charm_rea_postprocess, CharmReaEvaluator

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

data_dir = 'data/CHARM'
dataset_path_ZH = f'{data_dir}/reasoning'
dataset_path_TransEn = f'{data_dir}/reasoning_Translate-EN'
fewshot_example_path_ZH = os.path.join(os.path.dirname(__file__), 'few-shot-examples')
fewshot_example_path_TransEn = os.path.join(os.path.dirname(__file__), 'few-shot-examples_Translate-EN')

XLT_template = 'Follow the given examples and answer the question.\n{_hint}\n\n I want you to act as an commonsense reasoning expert for Chinese. \n Request: {{input}}\n'
Translate_EN_template = 'Follow the given examples and answer the question.\n{_hint}\n\nQ: {{input}}\nA: '
Other_template = '请按照给定的例子回答问题。\n{_hint}\n\nQ：{{input}}\nA：'

settings = [
    (
        'Direct',
        '',
        dataset_path_ZH,
        fewshot_example_path_ZH,
        Other_template,
    ),
    (
        'ZH-CoT',
        '让我们一步一步来思考。',
        dataset_path_ZH,
        fewshot_example_path_ZH,
        Other_template,
    ),
    (
        'EN-CoT',
        "Let's think step by step.",
        dataset_path_ZH,
        fewshot_example_path_ZH,
        Other_template,
    ),
    (
        'XLT',
        """You should retell the request in English.\nYou should do the answer step by step to choose the right answer.\nYou should step-by-step answer the request.\nYou should tell me the answer in this format 'So the answer is'.""",
        dataset_path_ZH,
        fewshot_example_path_ZH,
        XLT_template,
    ),
    (
        'Translate-EN',
        "Let's think step by step.",
        dataset_path_TransEn,
        fewshot_example_path_TransEn,
        Translate_EN_template,
    ),
]

charm_rea_datasets = []

for _cot, _cot_prefix, dataset_path, fewshot_example_path, prompt_template in settings:
    for _task in charm_tasks:
        _fewshot_example_file = os.path.join(fewshot_example_path, f'{_task}_{_cot}.txt')
        with open(_fewshot_example_file, 'r') as f:
            _hint = f.read()

        charm_rea_reader_cfg = dict(input_columns=['input'], output_column='target')

        charm_rea_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[dict(role='HUMAN', prompt=prompt_template.format(_hint=_hint) + _cot_prefix)]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=512),
        )

        charm_rea_eval_cfg = dict(
            evaluator=dict(type=CharmReaEvaluator),
            pred_role='BOT',
            pred_postprocessor=dict(type=charm_rea_postprocess),
            dataset_postprocessor=dict(type=charm_rea_postprocess),
        )

        charm_rea_datasets.append(
            dict(
                type=CharmDataset,
                path=dataset_path,
                name=_task,
                abbr='charm-rea-' + _task + '_' + _cot,
                reader_cfg=charm_rea_reader_cfg,
                infer_cfg=charm_rea_infer_cfg.copy(),
                eval_cfg=charm_rea_eval_cfg.copy(),
            )
        )
