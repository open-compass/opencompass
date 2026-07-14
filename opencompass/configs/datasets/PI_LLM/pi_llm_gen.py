from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

from opencompass.datasets import PILLMDataset
from opencompass.openicl.icl_evaluator import PILLMEvaluator

# PI-LLM dataset configurations
pi_llm_reader_cfg = dict(
    input_columns=['prompt'],  # Chat messages from dataset
    output_column='answer_formatted'  # JSON ground truth answers
)

pi_llm_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='{prompt}')])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        max_out_len=2048)  # Allow longer outputs for key-value lists
)

# Dataset configurations for all 4 PI-LLM experiments
dataset_configs = [
    # Main experiments from core dataset
    {
        'abbr': 'pi_llm_exp_updates',
        'subset': 'core',
        'experiment_filter': 'exp_updates',
        'max_samples': 100,  # Limit for manageable evaluation time
        'description': 'Updates experiment: n_updates (2-400) randomized mode'
    },
    {
        'abbr': 'pi_llm_exp_keys',
        'subset': 'core',
        'experiment_filter': 'exp_keys',
        'max_samples': 100,
        'description': 'Keys experiment: n_keys with fixed n_updates (125/350)'
    },
    {
        'abbr': 'pi_llm_exp_valuelength',
        'subset': 'core',
        'experiment_filter': 'exp_valuelength',
        'max_samples': 100,
        'description': 'Value length experiment: fixed n_updates (4/20)'
    },
    # Sequential experiment from sequential_additional dataset
    {
        'abbr': 'pi_llm_exp_sequential',
        'subset': 'sequential_additional',
        'experiment_filter': 'exp_sequential',
        'max_samples': 50,
        'description': 'Sequential mode: non-randomized updates'
    }
]

# Generate dataset configurations
pi_llm_datasets = []
for config in dataset_configs:
    eval_cfg = dict(evaluator=dict(
        type=PILLMEvaluator,
        log_base=1.5  # AUC weighting base
    ))

    pi_llm_datasets.append(
        dict(type=PILLMDataset,
             abbr=config['abbr'],
             subset=config['subset'],
             path='giantfish-fly/pi-llm',
             experiment_filter=config.get('experiment_filter'),
             max_samples=config.get('max_samples'),
             reader_cfg=pi_llm_reader_cfg,
             infer_cfg=pi_llm_infer_cfg,
             eval_cfg=eval_cfg))
