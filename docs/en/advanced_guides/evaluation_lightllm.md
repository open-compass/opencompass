# Evaluation with Lightllm

We now support the evaluation of large language models using [Lightllm](https://github.com/ModelTC/lightllm) for inference. Developed by SenseTime, LightLLM is a Python-based LLM (Large Language Model) inference and serving framework, notable for its lightweight design, easy scalability, and high-speed performance. Lightllm provides support for various large Language models, allowing users to perform model inference through Lightllm, locally deploying it as a service. During the evaluation process, OpenCompass feeds data to Lightllm through an API and processes the response. OpenCompass has been adapted for compatibility with Lightllm, and this tutorial will guide you on using OpenCompass to evaluate models with Lightllm as the inference backend.

## Setup

### Install OpenCompass

Please follow the [instructions](https://opencompass.readthedocs.io/en/latest/get_started/installation.html) to install the OpenCompass and prepare the evaluation datasets.

### Install Lightllm

Please follow the [Lightllm homepage](https://github.com/ModelTC/lightllm) to install the Lightllm. Pay attention to aligning the versions of relevant dependencies, especially the version of the Transformers.

## Evaluation

We use the evaluation of Humaneval with the llama2-7B model as an example.

### Step-1: Deploy the model locally as a service using Lightllm.

```shell
python -m lightllm.server.api_server --model_dir /path/llama2-7B    \
                                     --host 0.0.0.0                 \
                                     --port 1030                    \
                                     --nccl_port 2066               \
                                     --max_req_input_len 4096       \
                                     --max_req_total_len 6144       \
                                     --tp 1                         \
                                     --trust_remote_code            \
                                     --max_total_token_num 120000
```

\*\*Note: \*\* tp can be configured to enable TensorParallel inference on several gpus, suitable for the inference of very large models.

\*\*Note: \*\* The max_total_token_num in the above command will affect the throughput performance during testing. It can be configured according to the documentation on the [Lightllm homepage](https://github.com/ModelTC/lightllm). As long as it does not run out of memory, it is often better to set it as high as possible.

\*\*Note: \*\* If you want to start multiple LightLLM services on the same machine, you need to reconfigure the above port and nccl_port.

You can use the following Python script to quickly test whether the current service has been successfully started.

```python
import time
import requests
import json

url = 'http://localhost:8080/generate'
headers = {'Content-Type': 'application/json'}
data = {
    'inputs': 'What is AI?',
    "parameters": {
        'do_sample': False,
        'ignore_eos': False,
        'max_new_tokens': 1024,
    }
}
response = requests.post(url, headers=headers, data=json.dumps(data))
if response.status_code == 200:
    print(response.json())
else:
    print('Error:', response.status_code, response.text)
```

### Step-2: Evaluate the above model using OpenCompass.

```shell
python run.py configs/eval_lightllm.py
```

You are expected to get the evaluation results after the inference and evaluation.

\*\*Note: \*\*In `eval_lightllm.py`, please align the configured URL with the service address from the previous step.
