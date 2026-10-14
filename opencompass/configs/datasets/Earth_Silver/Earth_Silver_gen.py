from opencompass.datasets import Earth_Silver_MCQDataset
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_evaluator import AccEvaluator


SYSTEM_PROMPT = 'You are a helpful assistant for answering earth science multiple-choice questions.\n\n'


ZERO_SHOT_PROMPT = 'Q: {question}\nPlease select the correct answer from the options above and output only the corresponding letter (A, B, C, or D) without any explanation or additional text.\n'


reader_cfg = dict(
    input_columns=['question'],  
    output_column='answer', 
)


infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[  
                dict(role='SYSTEM', fallback_role='HUMAN', prompt=SYSTEM_PROMPT),
            ],
            round=[  
                dict(
                    role='HUMAN',
                    prompt=ZERO_SHOT_PROMPT,  
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),  
    inferencer=dict(type=GenInferencer),  
)


eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',  
)


earth_silver_mcq_dataset = dict(
    type=Earth_Silver_MCQDataset,  
    abbr='earth_silver_mcq',  
    path='ai-earth/Earth-Silver',  
    prompt_mode='zero-shot', 
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)


earth_silver_mcq_datasets = [earth_silver_mcq_dataset]
