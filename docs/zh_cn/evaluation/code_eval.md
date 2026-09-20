# 代码评测

这里以 `humaneval` 和 `mbpp` 为例介绍 pass@1 / pass@k 的配置方式；当代码需要真实执行时（多语言 `humaneval-x`、`DS1000`），则通过独立的 Docker 代码评测服务完成，避免在开发环境执行模型生成的代码造成损失。

## pass@1

如果只需要生成单条回复来评测pass@1的性能，可以直接使用[configs/datasets/humaneval/humaneval_gen_8e312c.py](https://github.com/open-compass/opencompass/blob/main/configs/datasets/humaneval/humaneval_gen_8e312c.py) 和 [configs/datasets/mbpp/deprecated_mbpp_gen_1e1056.py](https://github.com/open-compass/opencompass/blob/main/configs/datasets/mbpp/deprecated_mbpp_gen_1e1056.py) 并参考通用的[快速上手教程](../get_started/quick_start.md)即可。

如果要进行多语言评测，可以参考本文[代码执行服务](#代码执行服务)一节。

## pass@k

如果对于单个example需要生成多条回复来评测pass@k的性能，需要参考以下两种情况。这里以10回复为例子：

### 通常情况

对于绝大多数模型来说，模型支持HF的generation中带有`num_return_sequences` 参数，我们可以直接使用来获取多回复。可以参考以下配置文件。

```python
from opencompass.datasets import MBPPDatasetV2, MBPPPassKEvaluator

with read_base():
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets

mbpp_datasets[0]['type'] = MBPPDatasetV2
mbpp_datasets[0]['eval_cfg']['evaluator']['type'] = MBPPPassKEvaluator
mbpp_datasets[0]['reader_cfg']['output_column'] = 'test_column'

datasets = []
datasets += humaneval_datasets
datasets += mbpp_datasets

models = [
    dict(
        type=HuggingFaceCausalLM,
        ...,
        generation_kwargs=dict(
            num_return_sequences=10,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        ),
        ...,
    )
]
```

对于 `mbpp`，在数据集和评测上需要有新的变更，所以同步修改`type`, `eval_cfg.evaluator.type`, `reader_cfg.output_column` 字段来适应新的需求。

另外我们需要模型的回复有随机性，同步需要设置`generation_kwargs`参数。这里注意要设置`num_return_sequences`得到回复数。

注意：`num_return_sequences` 必须大于等于k，本身pass@k是计算的概率估计。

具体可以参考以下配置文件
[examples/eval_code_passk.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_code_passk.py)

### 模型不支持多回复

适用于一些没有设计好的API以及功能缺失的HF模型。这个时候我们需要重复构造数据集来达到多回复的效果。这里可以参考以下配置文件。

```python
from opencompass.datasets import MBPPDatasetV2, MBPPPassKEvaluator

with read_base():
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets

humaneval_datasets[0]['abbr'] = 'openai_humaneval_pass10'
humaneval_datasets[0]['num_repeats'] = 10
mbpp_datasets[0]['abbr'] = 'mbpp_pass10'
mbpp_datasets[0]['num_repeats'] = 10
mbpp_datasets[0]['type'] = MBPPDatasetV2
mbpp_datasets[0]['eval_cfg']['evaluator']['type'] = MBPPPassKEvaluator
mbpp_datasets[0]['reader_cfg']['output_column'] = 'test_column'

datasets = []
datasets += humaneval_datasets
datasets += mbpp_datasets

models = [
    dict(
        type=HuggingFaceCausalLM,
        ...,
        generation_kwargs=dict(
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        ),
        ...,
    )
]
```

由于数据集的prompt并没有修改，我们需要替换对应的字段来达到数据集重复的目的。
需要修改以下字段：

- `num_repeats`: 数据集重复的次数
- `abbr`: 数据集的缩写最好随着重复次数一并修改，因为数据集数量会发生变化，防止与`.cache/dataset_size.json` 中的数值出现差异导致一些潜在的问题。

对于 `mbpp`，同样修改`type`, `eval_cfg.evaluator.type`, `reader_cfg.output_column` 字段。

另外我们需要模型的回复有随机性，同步需要设置`generation_kwargs`参数。

具体可以参考以下配置文件
[examples/eval_code_passk_repeat_dataset.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_code_passk_repeat_dataset.py)

## 代码执行服务

代码需要真实执行的数据集，评测在一个独立的 Docker 服务中进行。OpenCompass 使用的代码评测服务基于 [code-evaluator](https://github.com/open-compass/code-evaluator)项目搭建。

### 支持的数据集

1. humaneval-x

多编程语言的数据集 [humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x)
数据集[下载地址](https://github.com/THUDM/CodeGeeX2/tree/main/benchmark/humanevalx)，请下载需要评测的语言（××.jsonl.gz）文件，并放入`./data/humanevalx`文件夹。

目前支持的语言有`python`, `cpp`, `go`, `java`, `js`。

2. DS1000

Python 多算法库数据集 [ds1000](https://github.com/xlang-ai/DS-1000)
数据集[下载地址](https://github.com/xlang-ai/DS-1000/blob/main/ds1000_data.zip)

目前支持的算法库有`Pandas`, `Numpy`, `Tensorflow`, `Scipy`, `Sklearn`, `Pytorch`, `Matplotlib`。

### 启动代码评测服务

1. 确保您已经安装了 docker，可参考[安装docker文档](https://docs.docker.com/engine/install/)
2. 拉取代码评测服务项目，并构建 docker 镜像

选择你需要的数据集对应的dockerfile，在下面命令中做替换 `humanevalx` 或者 `ds1000`。

```shell
git clone https://github.com/open-compass/code-evaluator.git
docker build -t code-eval-{your-dataset}:latest -f docker/{your-dataset}/Dockerfile .
```

3. 使用以下命令创建容器

```shell
# 输出日志格式
docker run -it -p 5000:5000 code-eval-{your-dataset}:latest python server.py

# 在后台运行程序
# docker run -itd -p 5000:5000 code-eval-{your-dataset}:latest python server.py

# 使用不同的端口
# docker run -itd -p 5001:5001 code-eval-{your-dataset}:latest python server.py --port 5001
```

**注：**

- 如在评测Go的过程中遇到timeout，请在创建容器时候使用以下命令

```shell
docker run -it -p 5000:5000 -e GO111MODULE=on -e GOPROXY=https://goproxy.io code-eval-{your-dataset}:latest python server.py
```

4. 为了确保您能够访问服务，通过以下命令检测推理环境和评测服务访问情况。 (如果推理和代码评测在同一主机中运行服务，就跳过这个操作)

```shell
ping your_service_ip_address
telnet your_service_ip_address your_service_port
```

## 本地代码评测

模型推理和代码评测服务在同一主机，或者同一局域网中，可以直接进行代码推理及评测。**注意：DS1000暂不支持，请走异地评测**

### 配置文件

我们已经提供了 humaneval-x 在 codegeex2 上评估的[配置文件](https://github.com/open-compass/opencompass/blob/main/examples/eval_codegeex2.py)作为参考。
其中数据集以及相关后处理的配置文件为这个[链接](https://github.com/open-compass/opencompass/tree/main/configs/datasets/humanevalx)， 需要注意 humanevalx_eval_cfg_dict 中的evaluator 字段。

```python
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalXDataset, HumanevalXEvaluator

humanevalx_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test')

humanevalx_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024))

humanevalx_eval_cfg_dict = {
    lang : dict(
            evaluator=dict(
                type=HumanevalXEvaluator,
                language=lang,
                ip_address="localhost",    # replace to your code_eval_server ip_address, port
                port=5000),               # refer to https://github.com/open-compass/code-evaluator to launch a server
            pred_role='BOT')
    for lang in ['python', 'cpp', 'go', 'java', 'js']   # do not support rust now
}

humanevalx_datasets = [
    dict(
        type=HumanevalXDataset,
        abbr=f'humanevalx-{lang}',
        language=lang,
        path='./data/humanevalx',
        reader_cfg=humanevalx_reader_cfg,
        infer_cfg=humanevalx_infer_cfg,
        eval_cfg=humanevalx_eval_cfg_dict[lang])
    for lang in ['python', 'cpp', 'go', 'java', 'js']
]
```

### 任务启动

参考[五分钟快速开始](../get_started/quick_start.md)

## 异地代码评测

模型推理和代码评测服务分别在不可访问的不同机器中，需要先进行模型推理，收集代码推理结果。配置文件和推理流程都可以复用上面的教程。

### 收集推理结果（仅针对Humanevalx）

OpenCompass 在 `tools` 中提供了 `collect_code_preds.py` 脚本对推理结果进行后处理并收集，我们只需要提供启动任务时的配置文件，以及指定复用对应任务的工作目录，其参数含义与 `opencompass` 的 `--reuse` 一致，细节可参考[文档](https://opencompass.readthedocs.io/zh-cn/latest/get_started/quick_start.html#id4)。

```shell
python tools/collect_code_preds.py [config] [-r latest]
```

收集到的结果将会按照以下的目录结构保存到 `-r` 对应的工作目录中：

```
workdir/humanevalx
├── codegeex2-6b
│   ├── humanevalx_cpp.json
│   ├── humanevalx_go.json
│   ├── humanevalx_java.json
│   ├── humanevalx_js.json
│   └── humanevalx_python.json
├── CodeLlama-13b
│   ├── ...
├── CodeLlama-13b-Instruct
│   ├── ...
├── CodeLlama-13b-Python
│   ├── ...
├── ...
```

对于 DS1000 只需要拿到 `opencompass` 对应生成的 prediction文件即可。

### 代码评测

#### 以下仅支持Humanevalx

确保代码评测服务启动的情况下，使用 `curl` 提交请求：

```shell
curl -X POST -F 'file=@{result_absolute_path}' -F 'dataset={dataset/language}' {your_service_ip_address}:{your_service_port}/evaluate
```

例如：

```shell
curl -X POST -F 'file=@./examples/humanevalx/python.json' -F 'dataset=humanevalx/python' localhost:5000/evaluate
```

得到结果：

```
"{\"pass@1\": 37.19512195121951%}"
```

另外我们额外提供了 `with-prompt` 选项（默认为True），由于有些模型生成结果包含完整的代码（如WizardCoder），不需要 prompt + prediciton 的形式进行拼接，可以参考以下命令进行评测。

```shell
curl -X POST -F 'file=@./examples/humanevalx/python.json' -F 'dataset=humanevalx/python' -H 'with-prompt: False' localhost:5000/evaluate
```

#### 以下仅支持DS1000

确保代码评测服务启动的情况下，使用 `curl` 提交请求：

```shell
curl -X POST -F 'file=@./internlm-chat-7b-hf-v11/ds1000_Numpy.json' localhost:5000/evaluate
```

DS1000支持额外 debug 参数，注意开启之后会有大量log

- `full`: 额外打印每个错误样本的原始prediction，后处理后的prediction，运行程序以及最终报错。
- `half`: 额外打印每个错误样本的运行程序以及最终报错。
- `error`: 额外打印每个错误样本的最终报错。

```shell
curl -X POST -F 'file=@./internlm-chat-7b-hf-v11/ds1000_Numpy.json' -F 'debug=error' localhost:5000/evaluate
```

另外还可以通过同样的方式修改`num_workers`来控制并行数。

## 进阶教程

除了评测已支持的 `humanevalx` 数据集以外，用户还可能有以下需求:

### 支持新数据集

可以参考[支持新数据集](../extension/new_dataset.md)教程

### 修改后处理

1. 本地评测中，可以按照支持新数据集教程中的后处理部分来修改后处理方法；
2. 异地评测中，可以修改 `tools/collect_code_preds.py` 中的后处理部分；
3. 代码评测服务中，存在部分后处理也可以进行修改，详情参考下一部分教程；

### 代码评测服务 Debug

在支持新数据集或者修改后处理的过程中，可能会遇到需要修改原本的代码评测服务的情况，按照需求修改以下部分

1. 删除 `Dockerfile` 中安装 `code-evaluator` 的部分，在启动容器时将 `code-evaluator` 挂载

```shell
docker run -it -p 5000:5000 -v /local/path/of/code-evaluator:/workspace/code-evaluator code-eval:latest bash
```

2. 安装并启动代码评测服务，此时可以根据需要修改本地 `code-evaluator` 中的代码来进行调试

```shell
cd code-evaluator && pip install -r requirements.txt
python server.py
```
