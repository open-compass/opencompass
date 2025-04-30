from opencompass.datasets.PHYBench.PHYBench import PHYBenchDataset, PHYBenchEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

SYSTEM_PROMPT = 'You are a physics expert. Please read the following question and provide a step-by-step solution. Put your final answer, which must be a readable LaTeX formula, in a \\boxed{} environment.'
ZERO_SHOT_PROMPT = 'Solve the following physics problem:\n\n{content}\n\nFinal Answer: The final answer is $'

# Reader configuration
reader_cfg = dict(
    input_columns=['content'],  # Updated to match the field in dataset.map
    output_column='answer',      # Using answer as the reference answer
)

# Inference configuration
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

# Evaluation configuration
eval_cfg = dict(
    evaluator=dict(type=PHYBenchEvaluator),
    pred_role='BOT',
)

# Dataset configuration
PHYBench_dataset = dict(
    type=PHYBenchDataset,
    path='Eureka-Lab/PHYBench',
    abbr='PHYBench',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)
PHYBench_datasets = [PHYBench_dataset]

