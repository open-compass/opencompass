# Tutorial for Evaluating Reasoning Models

OpenCompass provides an evaluation tutorial for DeepSeek R1 series reasoning models (mathematical datasets).

- At the model level, we recommend using the sampling approach to reduce repetitions caused by greedy decoding
- For datasets with limited samples, we employ multiple evaluation runs and take the average
- For answer validation, we utilize LLM-based verification to reduce misjudgments from rule-based evaluation

## Installation and Preparation

Please follow OpenCompass's installation guide.

## Evaluation Configuration Setup

We provide example configurations in `examples/eval_deepseek_r1.py`. Below is the configuration explanation:

### Configuration Interpretation

#### 1. Dataset and Validator Configuration

```python
# Configuration supporting multiple runs (example)
from opencompass.configs.datasets.aime2024.aime2024_llmverify_repeat8_gen_e8fcee import aime2024_datasets

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)

# LLM validator configuration. Users need to deploy API services via LMDeploy/vLLM/SGLang or use OpenAI-compatible endpoints
verifier_cfg = dict(
    abbr='qwen2-5-32B-Instruct',
    type=OpenAISDK,
    path='Qwen/Qwen2.5-32B-Instruct',  # Replace with actual path
    key='YOUR_API_KEY',  # Use real API key
    openai_api_base=['http://your-api-endpoint'],  # Replace with API endpoint
    query_per_second=16,
    batch_size=1024,
    temperature=0.001,
    max_out_len=16384
)

# Apply validator to all datasets
for item in datasets:
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = verifier_cfg
```

#### 2. Model Configuration

We provided an example of evaluation based on LMDeploy as the reasoning model backend, users can modify path (i.e., HF path)

```python
# LMDeploy model configuration example
models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-distill-qwen-7b-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        engine_config=dict(session_len=32768, max_batch_size=128, tp=1),
        gen_config=dict(
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=32768
        ),
        max_seq_len=32768,
        batch_size=64,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    ),
    # Extendable 14B/32B configurations...
]
```

#### 3. Evaluation Process Configuration

```python
# Inference configuration
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask))
    
# Evaluation configuration
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=8),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLEvalTask)))
```

#### 4. Summary Configuration

```python
# Multiple runs results average configuration
summary_groups = [
    {
        'name': 'AIME2024-Aveage8',
        'subsets':[[f'aime2024-run{idx}', 'accuracy'] for idx in range(8)]
    },
    # Other dataset average configurations...
]

summarizer = dict(
    dataset_abbrs=[
        ['AIME2024-Aveage8', 'naive_average'],
        # Other dataset metrics...
    ],
    summary_groups=summary_groups
)

# Work directory configuration
work_dir = "outputs/deepseek_r1_reasoning"
```

## Evaluation Execution

### Scenario 1: Model loaded on 1 GPU, data evaluated by 1 worker, using a total of 1 GPU

```bash
opencompass examples/eval_deepseek_r1.py --debug --dump-eval-details
```

Evaluation logs will be output in the command line.

### Scenario 2: Model loaded on 1 GPU, data evaluated by 8 workers, using a total of 8 GPUs

You need to modify the `infer` configuration in the configuration file and set `num_worker` to 8

```python
# Inference configuration
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask))
```

At the same time, remove the `--debug` parameter from the evaluation command

```bash
opencompass examples/eval_deepseek_r1.py --dump-eval-details
```

In this mode, OpenCompass will use multithreading to start `$num_worker` tasks. Specific logs will not be displayed in the command line, instead, detailed evaluation logs will be shown under `$work_dir`.

### Scenario 3: Model loaded on 2 GPUs, data evaluated by 4 workers, using a total of 8 GPUs

Note that in the model configuration, `num_gpus` in `run_cfg` needs to be set to 2 (if using an inference backend, parameters such as `tp` in LMDeploy also need to be modified accordingly to 2), and at the same time, set `num_worker` in the `infer` configuration to 4

```python
models += [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-distill-qwen-14b-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        engine_config=dict(session_len=32768, max_batch_size=128, tp=2),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=32768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=2),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    ),
]
```

```python
# Inference configuration
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask))
```

### Evaluation Results

The evaluation results are displayed as follows:

```bash
dataset                             version    metric         mode    deepseek-r1-distill-qwen-7b-turbomind                                                                                                       ----------------------------------  ---------  -------------  ------  ---------------------------------------                                                                                                     MATH                                -          -              -                                         AIME2024-Aveage8                    -          naive_average  gen     56.25     

```

## Performance Baseline

Since the model uses Sampling for decoding, and the AIME dataset size is small, there may still be a performance fluctuation of 1-3 points even when averaging over 8 evaluations.

| Model                        | Dataset  | Metric   | Value |
| ---------------------------- | -------- | -------- | ----- |
| DeepSeek-R1-Distill-Qwen-7B  | AIME2024 | Accuracy | 56.3  |
| DeepSeek-R1-Distill-Qwen-14B | AIME2024 | Accuracy | 74.2  |
| DeepSeek-R1-Distill-Qwen-32B | AIME2024 | Accuracy | 74.2  |
