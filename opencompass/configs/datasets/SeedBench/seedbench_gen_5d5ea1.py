from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, JiebaRougeEvaluator, RougeEvaluator
from opencompass.datasets import SeedBenchDataset, F1ScoreEvaluator, my_multiple_select_postprocess, AverageRougeScoreEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess


agri_reader_cfg = dict(
    input_columns=['instruction', 'question'],
    output_column='answer'
    )

agri_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{instruction}\n{question}\n'
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)


default_dataset_cfg = {
    'type': SeedBenchDataset,
    'path': 'json',
    'reader_cfg': agri_reader_cfg,
    'infer_cfg': agri_infer_cfg,
}

dataset_configs = [
    # 1-n
    {'abbr': 'seedbench_1-1', 'data_file': '1-1.json', 'evaluator': 'AccEvaluator', 
     'pred_postprocessor': dict(type=first_option_postprocess, options='ABCD')},
    {'abbr': 'seedbench_1-2', 'data_file': '1-2.json', 'evaluator': 'F1ScoreEvaluator',
     'pred_postprocessor': dict(type=my_multiple_select_postprocess)},
    # {'abbr': 'seedbench_1-3_em', 'data_file': '1-3.json', 'evaluator': 'ExactMatchScoreEvaluator'},
    {'abbr': 'seedbench_1-3', 'data_file': '1-3.json', 'evaluator': 'AverageRougeScoreEvaluator'},
    {'abbr': 'seedbench_1-4', 'data_file': '1-4.json', 'evaluator': 'RougeEvaluator'},
    # # 2-n
    {'abbr': 'seedbench_2-1', 'data_file': '2-1.json', 'evaluator': 'RougeEvaluator'},
    {'abbr': 'seedbench_2-2', 'data_file': '2-2.json', 'evaluator': 'RougeEvaluator'},
    # 3-n
    {'abbr': 'seedbench_3-1', 'data_file': '3-1.json', 'evaluator': 'AccEvaluator',
    'pred_postprocessor': dict(type=first_option_postprocess, options='ABCD')},
    {'abbr': 'seedbench_3-2', 'data_file': '3-2.json', 'evaluator': 'F1ScoreEvaluator',
     'pred_postprocessor': dict(type=my_multiple_select_postprocess)},
    # {'abbr': 'seedbench_3-3_em', 'data_file': '3-3.json', 'evaluator': 'ExactMatchScoreEvaluator'},
    {'abbr': 'seedbench_3-3', 'data_file': '3-3.json', 'evaluator': 'AverageRougeScoreEvaluator'},
    {'abbr': 'seedbench_3-4', 'data_file': '3-4.json', 'evaluator': 'RougeEvaluator'},
    {'abbr': 'seedbench_3-5', 'data_file': '3-5.json', 'evaluator': 'AccScoreStr_Evaluator'},
]

seedbench_datasets = []
for stage in ['zero-shot','one-shot']:
    for config in dataset_configs:
        eval_cfg = dict(
            evaluator=dict(type=config['evaluator'])
        )
        if 'pred_postprocessor' in config:
            eval_cfg['pred_postprocessor'] = config['pred_postprocessor']
        data_file = f"{stage}/{config['data_file']}"
        abbr_name = f"{config['abbr']}_{stage}"
        seedbench_datasets.append(
            dict(
                type=SeedBenchDataset,
                abbr=abbr_name,
                data_files=data_file,
                path='opencompass/seedbench',
                reader_cfg=agri_reader_cfg,
                infer_cfg=agri_infer_cfg,
                eval_cfg=eval_cfg
            )
        )
