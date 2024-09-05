## Extract Final Answers with Postprocess Models

OpenCompass now support postprocess (extract) prediction answers with postprocess models, to get the true ability level of models. Now, we use [XFinder](https://github.com/IAAR-Shanghai/xFinder) as our first postprocess model to extract the final answers from the model outputs.

We support four types of task types now:

1. **math**: for math questions with numerical pr formula answers, like GSM8k, Math, etc.
2. **alphabet_option**: for alphabet option questions with alphabet answers, like CommonsenseQA, MMLU, etc.
3. **short_text**: for questions answer type is a short text with selected short text answers.

Here are some examples of the question type supported from the official xFinder repo:

```json
[
    {
        "model_name": "Llama3_8B_Instruct",
        "dataset": "MetaMathQA",
        "key_answer_type": "math",
        "question": "Find the smallest, positive four-digit multiple of $15.$",
        "llm_output": "The smallest four-digit multiple of $15$ is $\\boxed{1005}.$",
        "correct_answer": "1005",
        "standard_answer_range": "a(n) number / set / vector / matrix / interval / expression / function / equation / inequality"
    },
    {
        "model_name": "LLaMA2_7B_Chat",
        "dataset": "SIQA_text",
        "key_answer_type": "short_text",
        "question": "Jordan kept Carson up all night talking because Jordan's girlfriend left them and they were quite upset. How would Carson feel as a result? Answer Choices: drowsy / upset their girlfriend left / energetic",
        "llm_output": "Based on the information provided in the question, Carson would likely feel upset because Jordan's girlfriend left them and they were quite upset. Therefore, the correct answer is (B) upset their girlfriend left.",
        "correct_answer": "drowsy",
        "standard_answer_range": [
            "drowsy",
            "upset their girlfriend left",
            "energetic"
        ]
    },
    {
        "model_name": "PHI2",
        "dataset": "CommonsenseQA_alpha",
        "key_answer_type": "alphabet_option",
        "question": "Why would you go to an office? Answer Choices: (A) work (B) school building (C) paper (D) city (E) habit",
        "llm_output": "A) work (B) school building (C) paper (D) city (E) habit\n\nExplanation:\n We are looking for a reason why someone would go to an office. The only answer choice that matches this is option A) work. Therefore, the correct answer is A) work.",
        "correct_answer": "A",
        "standard_answer_range": [
            [
                "A",
                "work"
            ],
            [
                "B",
                "school building"
            ],
            [
                "C",
                "paper"
            ],
            [
                "D",
                "city"
            ],
            [
                "E",
                "habit"
            ]
        ]
    }
]
```

## How to Use Model Postprocess in OpenCompass

### Step 1: Deploy the Postprocess Model Server

For now, there are two xFinder models can use, you can download them from Huggingface model hub:

1. **IAAR-Shanghai/xFinder-qwen1505**
2. **IAAR-Shanghai/xFinder-llama38it**

You can use LMDeploy or vLLM to deploy the xFinder model server, for example, you can use the following command to deploy the xFinder model server with LMDeploy:

```bash
lmdeploy serve api_server IAAR-Shanghai/xFinder-qwen1505  --model-name xFinder-qwen1505  --server-port 23333 --backend turbomind --tp 1
```

### Step 2: Set the Postprocess Model Config in the Dataset Configuration

We make the postprocess as a common postprocess function in OpenCompass, so you can use it by setting the `postprocess` parameter in the `predict` function of OpenCompass. It can be used with the default postprocess regularization extract function at the same time. The only thing you need to do is to deploy the postprocess model server and set the `model_postprocessor` to the original `eval_cfg` in the dataset configuration, like the following example:

```python
from opencompass.utils.model_postprocessors import xfinder_postprocess

...

    model_postprocessor=dict(
        type=xfinder_postprocess,
        question_type='math',
        xfinder_model_name='xFinder-qwen1505',
        xfiner_api_url='http://0.0.0.0:23333/v1,http://0.0.0.0:23334/v1')
```

Explanation of the parameters:

- `question_type`: the type of the question, which can be one of the three types mentioned above.
- `xfinder_model_name`: the name of the model you deploying the model server.
- `xfiner_api_url`: the URL of the model server, you can set multiple URLs with `,` to use multiple model servers, which can accelerate the postprocess speed.

📢：**Please attention following points**:

1. Now only support extract questions with Zero-shot setting.
2. For alphabet_option problems, the option should be like '\\nA. xxx\\nB. xxx\\nC. xxx\\nD. xxx\\nE. xxx\\n ...' or '\\n(A) xxx\\n(B) xxx\\n(C) xxx\\n(D) xxx\\n(E) xxx\\n ...' format, and the correct answer should be the alphabet of the correct answer, like 'A', 'B', 'C', 'D', 'E'.

For more details about the xFinder model, you can refer to the [xFinder](https://github.com/IAAR-Shanghai/xFinder), and for a complete example, you can refer to the following example, which is the configuration of the GSM8K dataset with the xFinder postprocess model:

```python
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_dataset_postprocess, Gsm8kEvaluator
from opencompass.datasets import MATHEvaluator, math_postprocess_v2
from opencompass.utils.model_postprocessors import xfinder_postprocess

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

gsm8k_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
    model_postprocessor=dict(
        type=xfinder_postprocess,
        question_type='math',
        xfinder_model_name='xFinder-qwen1505',
        xfiner_api_url='http://0.0.0.0:23333/v1,http://0.0.0.0:23334/v1')
    )

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
```

For evaluation results, `accuracy` is the result using default postprocess, and `model_postprocess_accuracy` is the result using xFinder postprocess, the gap can be wider when the model is not good answering the questions properly.

You can also use the `--dump-eval-details` command to dump the detailed evaluation details to see the model postprocess results from the `results` folder.

## Results Comparison with Different Question Types

We have tested the model postprocess method with XFinder model on the GSM8K, MMLU, Natural Questions (NQ) datasets for `Meta-Llama-3-8B-Instruct` with above settings, and the results are as follows:

| Dataset | Type            | Config Name              | Regex Postprocess Score | Model Postprocess Score |
| ------- | --------------- | ------------------------ | ----------------------- | ----------------------- |
| gsm8k   | math            | gsm8k_xfinder_gen_a58960 | 73.46                   | 78.09                   |
| nq      | short_text      | nq_xfinder_gen_3dcea1    | 22.33                   | 37.53                   |
| mmlu    | alphabet_option | mmlu_xfinder_gen_4d595a  | 67.89                   | 67.93                   |

## Citation

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}

@misc{yu2024xfinderrobustpinpointanswer,
      title={xFinder: Robust and Pinpoint Answer Extraction for Large Language Models},
      author={Qingchen Yu and Zifan Zheng and Shichao Song and Zhiyu Li and Feiyu Xiong and Bo Tang and Ding Chen},
      year={2024},
      eprint={2405.11874},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.11874},
}

```
