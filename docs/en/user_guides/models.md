# Prepare Models

To support the evaluation of new models in OpenCompass, there are several ways:

1. InternLM-based models
2. HuggingFace-based models
3. API-based models
4. Custom models

## InternLM-based Models

In OpenCompass, we support loading models from internLM for evaluation.
Here is an example internLM model configuration file:

```python
from opencompass.models.intern import intern_model

models = [
    dict(
        type=intern_model,
        path="/program/model/model_test/",
        tokenizer_path='/program/tokenizers/V7.model',
        tokenizer_type='v7',
        model_config = "/program/model/model_config.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1))
]

```

Explanation of some of the parameters:

- `path`: The path for internLM weights, from which OpenCompass will load weights.
- `tokenizer_path`: is the path of the tokenizer, and OpenCompass loads the tokenizer based on this path.
- `tokenizer_type`: specifies the type of tokenizer. Option type include llama, v4, v6, and v7.
- `model_config`: The passed in model configuration path will load the model's config based on the configuration in the file.
Here is a model_ Specific examples of config:

```python
model = dict(
    checkpoint = False,
    num_chunks = 1,
    num_attention_heads = 32,
    embed_split_hidden = True,
    vocab_size = 103168,
    embed_grad_scale = 1,
    parallel_output = True,
    hidden_size = 4096,
    num_layers = 32,
    mlp_ratio = 2.6666666666666665,
    apply_post_layer_norm = False,
    no_bias = True,
    deepnorm = False,
    dtype = "torch.bfloat16",
    norm_type = 'rmsnorm',
    layer_norm_epsilon = 1e-05
)
parallel = dict(
    zero1=1,
)
```
The parallel remains unchanged, always specifying zero1=1, and the model can change the corresponding parameters according to the requirements.

## HuggingFace-based Models

In OpenCompass, we support constructing evaluation models directly from HuggingFace's
`AutoModel.from_pretrained` and `AutoModelForCausalLM.from_pretrained` interfaces. If the model to be
evaluated follows the typical generation interface of HuggingFace models, there is no need to write code. You
can simply specify the relevant configurations in the configuration file.

Here is an example configuration file for a HuggingFace-based model:

```python
# Use `HuggingFace` to evaluate models supported by AutoModel.
# Use `HuggingFaceCausalLM` to evaluate models supported by AutoModelForCausalLM.
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        # Parameters for `HuggingFaceCausalLM` initialization.
        path='huggyllama/llama-7b',
        tokenizer_path='huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        batch_padding=False,
        # Common parameters shared by various models, not specific to `HuggingFaceCausalLM` initialization.
        abbr='llama-7b',            # Model abbreviation used for result display.
        max_out_len=100,            # Maximum number of generated tokens.
        batch_size=16,              # The size of a batch during inference.
        run_cfg=dict(num_gpus=1),   # Run configuration to specify resource requirements.
    )
]
```

Explanation of some of the parameters:

- `batch_padding=False`: If set to False, each sample in a batch will be inferred individually. If set to True,
  a batch of samples will be padded and inferred together. For some models, such padding may lead to
  unexpected results. If the model being evaluated supports sample padding, you can set this parameter to True
  to speed up inference.
- `padding_side='left'`: Perform padding on the left side. Not all models support padding, and padding on the
  right side may interfere with the model's output.
- `truncation_side='left'`: Perform truncation on the left side. The input prompt for evaluation usually
  consists of both the in-context examples prompt and the input prompt. If the right side of the input prompt
  is truncated, it may cause the input of the generation model to be inconsistent with the expected format.
  Therefore, if necessary, truncation should be performed on the left side.

During evaluation, OpenCompass will instantiate the evaluation model based on the `type` and the
initialization parameters specified in the configuration file. Other parameters are used for inference,
summarization, and other processes related to the model. For example, in the above configuration file, we will
instantiate the model as follows during evaluation:

```python
model = HuggingFaceCausalLM(
    path='huggyllama/llama-7b',
    tokenizer_path='huggyllama/llama-7b',
    tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
    max_seq_len=2048,
)
```

## API-based Models

Currently, OpenCompass supports API-based model inference for the following:

- OpenAI (`opencompass.models.OpenAI`)
- More coming soon

Let's take the OpenAI configuration file as an example to see how API-based models are used in the
configuration file.

```python
from opencompass.models import OpenAI

models = [
    dict(
        type=OpenAI,                             # Using the OpenAI model
        # Parameters for `OpenAI` initialization
        path='gpt-4',                            # Specify the model type
        key='YOUR_OPENAI_KEY',                   # OpenAI API Key
        max_seq_len=2048,                        # The max input number of tokens
        # Common parameters shared by various models, not specific to `OpenAI` initialization.
        abbr='GPT-4',                            # Model abbreviation used for result display.
        max_out_len=512,                         # Maximum number of generated tokens.
        batch_size=1,                            # The size of a batch during inference.
        run_cfg=dict(num_gpus=0),                # Resource requirements (no GPU needed)
    ),
]
```

# Custom Models

If the above methods do not support your model evaluation requirements, you can refer to
[Supporting New Models](../advanced_guides/new_model.md) to add support for new models in OpenCompass.
