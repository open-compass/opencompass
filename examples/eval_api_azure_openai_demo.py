"""
Example configuration of using Azure OpenAI models.

This demonstrates how to use Azure Managed Identity if API keys are not available for authentication.
"""

from mmengine.config import read_base

from opencompass.models import OpenAI, OpenAISDK

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets

# API template for chat models
api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(
        abbr='Azure-GPT-5.1',
        type=OpenAI,
        path='gpt-5.1',
        use_azure_identity=True,  # Enable Azure identity authentication
        tokenizer_path='gpt-5',
        # Azure OpenAI endpoint format:
        openai_api_base='https://{resource-name}.openai.azure.com/openai/deployments/{deployment-name}/chat/completions?api-version=2024-12-01-preview',
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8,
        retry=2,
    ),
    dict(
        abbr='Azure-GPT-5.1-SDK',
        type=OpenAISDK,
        path='gpt-5.1',
        use_azure_identity=True,  # Enable Azure identity authentication
        tokenizer_path='gpt-5',
        # Azure OpenAI endpoint format:
        openai_api_base='https://{resource-name}.openai.azure.com/openai/v1/',
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8,
        retry=2,
    ),
]

# Datasets to evaluate
datasets = gsm8k_datasets
