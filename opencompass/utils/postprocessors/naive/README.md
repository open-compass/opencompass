## Short Usage Introduction for Naive Model Postprocessor with Custom Model

<!-- Now OC can use  -->

### Step 1: Deploy an API server using vLLM or LMDeploy

```bash
lmdeploy serve api_server meta-llama/Meta-Llama-3-8B-Instruct --model-name llama3-8b-instruct  --server-port 23333 --backend turbomind --tp 1
```

### Step 2: Add Naive Model Postprocessor to the configuration file

Take GSM8K as an example, you can add the following lines to the configuration file and replace the `api_url` with the correct address of the API server.

```python
...
from opencompass.utils.model_postprocessors import navie_model_postprocess
from opencompass.utils.postprocessors.naive import MATH_NAVIE_PROMPT_TEMPLATE

...

gsm8k_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
    # Add the following line to use the naive model postprocessor
    model_postprocessor=dict(
        type=navie_model_postprocess,
        custom_instruction=MATH_NAVIE_PROMPT_TEMPLATE,
        model_name='llama3-8b-instruct',
        api_url='http://0.0.0.0:23333/v1,http://0.0.0.0:23334/v1')
    )
...

```

The prompt for extraction can also be customized by changing the `custom_instruction` parameter. Now support two default templates: `MATH_NAVIE_PROMPT_TEMPLATE` for math problems extraction like GSM8K and MATH, and `OPTION_NAVIE_PROMPT_TEMPLATE` for option problems extraction like MMLU. You can also write your own prompt template, like:

```python
OPTION_NAVIE_PROMPT_TEMPLATE = """
There is a detailed explanation of the final answer you should extract:
1. You should extract the final answer option like 'A', 'B', 'C', 'D' ... from the given output sentences.
2. The question is a single choice question, so the final answer option should be one of the options, not a combination of options.
"""
```

Your prompt should start with `There is a detailed explanation of the final answer you should extract:` and following with your customized instructions.

### Step 3: Run the Evaluation as Usual

Now you can run the evaluation as usual with the configuration file you modified. The evaluation will use the custom model as the post-process model to get the final result. The final result will be the `model_postprocess_accuracy` in the evaluation result, like:

```Markdown
dataset                                            version    metric                      mode      llama-3-8b-instruct-turbomind
-------------------------------------------------  ---------  --------------------------  ------  -------------------------------
gsm8k                                              a58960     accuracy                    gen                               73.46
gsm8k                                              a58960     model_postprocess_accuracy  gen                               78.77
```

## Experiment Results

We have tested the model postprocess method with different models (Qwen2-72B-Chat, Llama3-8b-Chat) as post-process model on the GSM8K, MMLU datasets for `Meta-Llama-3-8B-Instruct` with above settings, and the results are as follows:

```Markdown
| Dataset | Type            | Config ID              | Regex Postprocess Score | Model Postprocess Score (Llama3-8b-Instruct) | Model Postprocess Score (Qwen2-72B-Chat) |
| ------- | --------------- | ------------------------ | ----------------------- | ----------------------- |----------------------- |
| gsm8k   | math            | a58960                   | 73.46               | 79.08                  | 78.77                   |
| mmlu    | option          | 4d595a                   | 67.89               | 65.26                  | 67.94                  |
```

The `metric` column with `model_postprocess_accuracy` is the final result after the `Naive Model Postprocessor` is applied.
