# 评测 Lightllm 模型

我们支持评测使用 [Lightllm](https://github.com/ModelTC/lightllm) 进行推理的大语言模型。Lightllm 是由商汤科技开发，是一个基于 Python 的 LLM 推理和服务框架，以其轻量级设计、易于扩展和高速性能而著称，Lightllm 对多种大模型都进行了支持。用户可以通过 Lightllm 进行模型推理，并且以服务的形式在本地起起来，在评测过程中，OpenCompass 通过 api 将数据喂给Lightllm，并对返回的结果进行处理。OpenCompass 对 Lightllm 进行了适配，本教程将介绍如何使用 OpenCompass 来对以 Lightllm 作为推理后端的模型进行评测。

## 环境配置

### 安装 OpenCompass

请根据 OpenCompass [安装指南](https://opencompass.readthedocs.io/en/latest/get_started/installation.html) 来安装算法库和准备数据集。

### 安装 Lightllm

请根据 [Lightllm 主页](https://github.com/ModelTC/lightllm) 来安装 Lightllm。注意对齐相关依赖库的版本，尤其是 transformers 的版本。

## 评测

我们以 llama2-7B 评测 humaneval 作为例子来介绍如何评测。

### 第一步: 将模型通过 Lightllm 在本地以服务的形式起起来

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

**注：** 上述命令可以通过 tp 的数量设置，在 tp 张卡上进行 TensorParallel 推理，适用于较大的模型的推理。

**注：** 上述命令中的 max_total_token_num，会影响测试过程中的吞吐性能，可以根据 [Lightllm 主页](https://github.com/ModelTC/lightllm) 上的文档，进行设置。只要不爆显存，往往设置越大越好。

**注：** 如果要在同一个机器上起多个 Lightllm 服务，需要重新设定上面的 port 和 nccl_port。

可以使用下面的 Python 脚本简单测试一下当前服务是否已经起成功

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

### 第二步: 使用 OpenCompass 评测上述模型

```shell
python run.py configs/eval_lightllm.py
```

当模型完成推理和指标计算后，我们便可获得模型的评测结果。

**注：** `eval_lightllm.py` 中，配置的 url 要和上一步服务地址对齐。
