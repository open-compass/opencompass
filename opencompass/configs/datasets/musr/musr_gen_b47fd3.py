from opencompass.datasets import MusrDataset, MusrEvaluator
from opencompass.openicl import PromptTemplate, ZeroRetriever, GenInferencer


DATASET_CONFIGS = {
    'murder_mysteries': {
        'abbr': 'musr_murder_mysteries',
        'name': 'murder_mysteries',
        'path': 'opencompass/musr',  
        'reader_cfg': dict(
            input_columns=['context', 'question_text', 'question', 'answer', 'choices', 'choices_str', 'intermediate_trees', 'intermediate_data', 'prompt', 'system_prompt', 'gold_answer', 'scidx', 'self_consistency_n', 'ablation_name'],
            output_column='gold_answer',
        ),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt='{system_prompt}'
                        )
                    ],
                    round=[
                        dict(
                            role='HUMAN',
                            prompt='{prompt}'
                        ),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        'eval_cfg': dict(
            evaluator=dict(
                type=MusrEvaluator,
                answer_index_modifier=1,
                self_consistency_n=1
            ),
        ),
    },
    'object_placements': {
        'abbr': 'musr_object_placements',
        'name': 'object_placements',
        'path': 'opencompass/musr',
        'reader_cfg': dict(
            input_columns=['context', 'question_text', 'question', 'answer', 'choices', 'choices_str', 'intermediate_trees', 'intermediate_data', 'prompt', 'system_prompt', 'gold_answer', 'scidx', 'self_consistency_n', 'ablation_name'],
            output_column='gold_answer',
        ),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt='{system_prompt}'
                        )
                    ],
                    round=[
                        dict(
                            role='HUMAN',
                            prompt='{prompt}'
                        ),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        'eval_cfg': dict(
            evaluator=dict(
                type=MusrEvaluator,
                answer_index_modifier=1,
                self_consistency_n=1
            ),
        ),
    },
    'team_allocation': {
        'abbr': 'musr_team_allocation',
        'name': 'team_allocation',
        'path': 'opencompass/musr',
        'reader_cfg': dict(
            input_columns=['context', 'question_text', 'question', 'answer', 'choices', 'choices_str', 'intermediate_trees', 'intermediate_data', 'prompt', 'system_prompt', 'gold_answer', 'scidx', 'self_consistency_n', 'ablation_name'],
            output_column='gold_answer',
        ),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt='{system_prompt}'
                        )
                    ],
                    round=[
                        dict(
                            role='HUMAN',
                            prompt='{prompt}'
                        ),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        'eval_cfg': dict(
            evaluator=dict(
                type=MusrEvaluator,
                answer_index_modifier=1,
                self_consistency_n=1
            ),
        ),
    },
}


musr_datasets = []

for config in DATASET_CONFIGS.values():
    dataset = dict(
        abbr=config['abbr'],
        type=MusrDataset,
        path=config['path'],
        name=config['name'],
        reader_cfg=config['reader_cfg'],
        infer_cfg=config['infer_cfg'],
        eval_cfg=config['eval_cfg'],
    )
    musr_datasets.append(dataset)
