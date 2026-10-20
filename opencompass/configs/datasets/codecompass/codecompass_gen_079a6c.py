

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CodeCompassCodeGenerationDataset


# Reader Config
codecompass_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='metadata',
    train_split='test'
)

# Inference Config
codecompass_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='{prompt}')])
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048)
)

codecompass_eval_cfg = dict(
    evaluator=dict(
        type='CodeCompassEvaluator',
        num_process_evaluate=16,
        timeout=15,
        k_list=[1],
        dataset_path='opencompass/CodeCompass'
    ),
    pred_role='BOT',
)

codecompass_datasets = [ 
    dict(
        type=CodeCompassCodeGenerationDataset,
        abbr='codecompass_gen_cpp',
        path='opencompass/CodeCompass', 
        reader_cfg=codecompass_reader_cfg,
        infer_cfg=codecompass_infer_cfg,
        eval_cfg=codecompass_eval_cfg
    )
]