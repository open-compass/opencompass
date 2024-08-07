from mmengine.config import read_base
with read_base():
    from .atc_choice_20 import *

needle_num_list = list(range(2, 80, 1))
needlebench_datasets = []

for _name in list(single_choice_prompts.keys()):

    needlebench_atc_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=(single_choice_prompts[_name])),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer,),
    )

    needlebench_atc_eval_cfg = dict(
        evaluator=dict(type=CircularEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

    for num_needles in needle_num_list:
        abbr = (f'NeedleBenchATCDataset-'
                f'{num_needles}Needle-{"EN" if "en" in _name else "ZH"}')
        language = 'English' if 'en' in _name else 'Chinese'
        if 'reasoning' in _name:
            abbr += '-Reasoning'
        dataset_dict = {
            'abbr': abbr,
            'type': NeedleBenchATCDataset,
            'path': names_path,
            'num_needles': num_needles,
            'language': language,
            'repeats': repeats,
            'with_circular': with_circular_eval,
            'reader_cfg': needlebench_atc_reader_cfg,
            'infer_cfg': needlebench_atc_infer_cfg,
            'eval_cfg': needlebench_atc_eval_cfg
        }
        needlebench_datasets.append(dataset_dict)
