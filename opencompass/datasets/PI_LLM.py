import json

from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class PILLMDataset(BaseDataset):
    """
    PI-LLM: Context (Proactive) Interference in Large Language Models

    A benchmark dataset for evaluating context (proactive) interference
    effects in LLMs through simple key-value tracking tasks. Tests and
    shows LLM working memory limitations beyond context length.

    ICML 2025 Long-Context Foundation Models Workshop - Accepted

    Dataset Structure:
    - exp_updates: Single mode (n_updates: 2-400)
    - exp_keys: Two modes (easy: n_updates=125, hard: n_updates=350)
    - exp_valuelength: Two modes (easy: n_updates=4, hard: n_updates=20)
    - exp_sequential: Single mode (n_updates: 2-400, non-randomized)

    Homepage: https://sites.google.com/view/cog4llm
    Paper: https://arxiv.org/abs/2506.08184
    HuggingFace: https://huggingface.co/datasets/giantfish-fly/pi-llm
    """

    @staticmethod
    def load(**kwargs) -> Dataset:
        """
        Load PI-LLM dataset from HuggingFace Hub.

        Args:
            subset: Dataset subset ('core' or 'sequential_additional')
            experiment_filter: Filter by experiment type
                - 'exp_updates': Main updates experiment (2-400 updates)
                - 'exp_keys': Keys experiment (varying n_keys,
                              fixed n_updates=125/350)
                - 'exp_valuelength': Value length experiment
                                     (varying lengths, fixed n_updates=4/20)
                - 'exp_sequential': Sequential mode (randomize_mode=off)
            max_samples: Maximum samples to load (for testing)

        Returns:
            Dataset: HuggingFace dataset with samples
        """
        subset = kwargs.get('subset', 'core')
        experiment_filter = kwargs.get('experiment_filter', None)
        max_samples = kwargs.get('max_samples', None)

        # Load from HuggingFace (parquet format supported natively)
        dataset = load_dataset(kwargs.get('path', None), subset)

        # Extract test split
        data = dataset['test'].to_list()

        # Filter by experiment type if specified
        if experiment_filter:
            if experiment_filter == 'exp_updates':
                # Main updates experiment from core dataset
                data = [
                    item for item in data
                    if 'exp_updates' in item.get('experiment', '')
                    and 'randomoff' not in item.get('experiment', '')
                ]
            elif experiment_filter == 'exp_keys':
                # Keys experiment from core dataset
                data = [
                    item for item in data
                    if 'exp_keys' in item.get('experiment', '')
                ]
            elif experiment_filter == 'exp_valuelength':
                # Value length experiment from core dataset
                data = [
                    item for item in data
                    if 'exp_valuelength' in item.get('experiment', '')
                ]
            elif experiment_filter == 'exp_sequential':
                # Sequential mode from sequential_additional dataset
                data = [
                    item for item in data
                    if 'extra_exp_updates_randomoff' in item.get(
                        'experiment', '')
                ]

        # Limit samples for testing if specified
        if max_samples:
            data = data[:max_samples]

        # Parse prompt strings to proper chat format
        for item in data:
            if isinstance(item.get('prompt'), str):
                try:
                    item['prompt'] = json.loads(item['prompt'])
                except json.JSONDecodeError:
                    # Fallback: wrap in basic chat format
                    item['prompt'] = [{
                        'role': 'user',
                        'content': item['prompt']
                    }]

        return Dataset.from_list(data)
