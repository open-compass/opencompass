# Learn About Config

OpenCompass uses the OpenMMLab modern style configuration files. If you are familiar with the OpenMMLab style
configuration files, you can directly refer to
[A Pure Python style Configuration File (Beta)](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta)
to understand the differences between the new-style and original configuration files. If you have not
encountered OpenMMLab style configuration files before, I will explain the usage of configuration files using
a simple example. Make sure you have installed the latest version of MMEngine to support the
new-style configuration files.

## Basic Format

OpenCompass configuration files are in Python format, following basic Python syntax. Each configuration item
is specified by defining variables. For example, when defining a model, we use the following configuration:

```python
# model_cfg.py
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        path='huggyllama/llama-7b',
        model_kwargs=dict(device_map='auto'),
        tokenizer_path='huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        max_out_len=50,
        run_cfg=dict(num_gpus=8, num_procs=1),
    )
]
```

When reading the configuration file, use `Config.fromfile` from MMEngine for parsing:

```python
>>> from mmengine.config import Config
>>> cfg = Config.fromfile('./model_cfg.py')
>>> print(cfg.models[0])
{'type': HuggingFaceCausalLM, 'path': 'huggyllama/llama-7b', 'model_kwargs': {'device_map': 'auto'}, ...}
```

## Inheritance Mechanism

OpenCompass configuration files use Python's import mechanism for file inheritance. Note that when inheriting
configuration files, we need to use the `read_base` context manager.

```python
# inherit.py
from mmengine.config import read_base

with read_base():
    from .model_cfg import models  # Inherits the 'models' from model_cfg.py
```

Parse the configuration file using `Config.fromfile`:

```python
>>> from mmengine.config import Config
>>> cfg = Config.fromfile('./inherit.py')
>>> print(cfg.models[0])
{'type': HuggingFaceCausalLM, 'path': 'huggyllama/llama-7b', 'model_kwargs': {'device_map': 'auto'}, ...}
```

## Evaluation Configuration Example

```python
# configs/llama7b.py
from mmengine.config import read_base

with read_base():
    # Read the required dataset configurations directly from the preset dataset configurations
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets

# Concatenate the datasets to be evaluated into the datasets field
datasets = [*piqa_datasets, *siqa_datasets]

# Evaluate models supported by HuggingFace's `AutoModelForCausalLM` using `HuggingFaceCausalLM`
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        # Initialization parameters for `HuggingFaceCausalLM`
        path='huggyllama/llama-7b',
        tokenizer_path='huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        # Common parameters for all models, not specific to HuggingFaceCausalLM's initialization parameters
        abbr='llama-7b',            # Model abbreviation for result display
        max_out_len=100,            # Maximum number of generated tokens
        batch_size=16,
        run_cfg=dict(num_gpus=1),   # Run configuration for specifying resource requirements
    )
]
```

## Dataset Configuration File Example

In the above example configuration file, we directly inherit the dataset-related configurations. Next, we will
use the PIQA dataset configuration file as an example to demonstrate the meanings of each field in the dataset
configuration file. If you do not intend to modify the prompt for model testing or add new datasets, you can
skip this section.

The PIQA dataset [configuration file](https://github.com/open-compass/opencompass/blob/main/configs/datasets/piqa/piqa_ppl_1cf9f0.py) is as follows.
It is a configuration for evaluating based on perplexity (PPL) and does not use In-Context Learning.

```python
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

# Reading configurations
# The loaded dataset is usually organized as dictionaries, specifying the input fields used to form the prompt
# and the output field used as the answer in each sample
piqa_reader_cfg = dict(
    input_columns=['goal', 'sol1', 'sol2'],
    output_column='label',
    test_split='validation',
)

# Inference configurations
piqa_infer_cfg = dict(
    # Prompt generation configuration
    prompt_template=dict(
        type=PromptTemplate,
        # Prompt template, the template format matches the inferencer type specified later
        # Here, to calculate PPL, we need to specify the prompt template for each answer
        template={
            0: 'The following makes sense: \nQ: {goal}\nA: {sol1}\n',
            1: 'The following makes sense: \nQ: {goal}\nA: {sol2}\n'
        }),
    # In-Context example configuration, specifying `ZeroRetriever` here, which means not using in-context example.
    retriever=dict(type=ZeroRetriever),
    # Inference method configuration
    #   - PPLInferencer uses perplexity (PPL) to obtain answers
    #   - GenInferencer uses the model's generated results to obtain answers
    inferencer=dict(type=PPLInferencer))

# Metric configuration, using Accuracy as the evaluation metric
piqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

# Dataset configuration, where all the above variables are parameters for this configuration
# It is a list used to specify the configurations of different evaluation subsets of a dataset.
piqa_datasets = [
    dict(
        type=HFDataset,
        path='piqa',
        reader_cfg=piqa_reader_cfg,
        infer_cfg=piqa_infer_cfg,
        eval_cfg=piqa_eval_cfg)
```

For detailed configuration of the **Prompt generation configuration**, you can refer to the [Prompt Template](../prompt/prompt_template.md).

## Advanced Evaluation Configuration

In OpenCompass, we support configuration options such as task partitioner and runner for more flexible and
efficient utilization of computational resources.

By default, we use size-based partitioning for inference tasks. You can specify the sample number threshold
for task partitioning using `--max-partition-size` when starting the task. Additionally, we use local
resources for inference and evaluation tasks by default. If you want to use Slurm cluster resources, you can
use the `--slurm` parameter and the `--partition` parameter to specify the Slurm runner backend when starting
the task.

Furthermore, if the above functionalities do not meet your requirements for task partitioning and runner
backend configuration, you can provide more detailed configurations in the configuration file. Please refer to
[Efficient Evaluation](./evaluation.md) for more information.
