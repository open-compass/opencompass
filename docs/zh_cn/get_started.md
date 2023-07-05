# 安装

1. 使用以下命令准备 OpenCompass 环境：

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
```

如果你希望自定义 PyTorch 版本或相关的 CUDA 版本，请参考 [官方文档](https://pytorch.org/get-started/locally/) 准备 PyTorch 环境。需要注意的是，OpenCompass 要求 `pytorch>=1.13`。

2. 安装 OpenCompass：

```bash
git clone https://github.com/opencompass/opencompass
cd opencompass
pip install -e .
```

3. 安装 humaneval（可选）

如果你希望在 humaneval 数据集上进行评估，请执行此步骤。

```
git clone https://github.com/openai/human-eval.git
cd human-eval
pip install -r requirments.txt
pip install -e .
cd ..
```

请仔细阅读 `human_eval/execution.py` **第48-57行**的注释，了解执行模型生成的代码可能存在的风险，如果接受这些风险，请取消**第58行**的注释，启用代码执行评测。

# 快速上手

在这一节，我们会以测试 LLaMA-7B 在 SIQA 和 PIQA 上的性能为例，带领你熟悉 OpenCompass 的一些基本功能。在运行前，
请先确保你安装好了 OpenCompass，并在本机或集群上有满足 LLaMA-7B 最低要求的 GPU 计算资源。

## 准备数据集

在仓库目录创建 data 文件夹，并将数据集文件放置在 data 文件夹中

## 准备评测配置文件

创建如下配置文件 `configs/llama.py`:

```python
from mmengine.config import read_base

with read_base():
    # 直接从预设数据集配置中读取需要的数据集配置
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets

# 将需要评测的数据集拼接成 datasets 字段
datasets = [*piqa_datasets, *siqa_datasets]

# 使用 HuggingFaceCausalLM 评测 HuggingFace 中 AutoModelForCausalLM 支持的模型
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        # 以下参数为 HuggingFaceCausalLM 的初始化参数
        path='huggyllama/llama-7b',
        tokenizer_path='huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        # 以下参数为各类模型都有的参数，非 HuggingFaceCausalLM 的初始化参数
        abbr='llama-7b',            # 模型简称，用于结果展示
        max_out_len=100,            # 最长生成 token 数
        batch_size=16,              # 批次大小
        run_cfg=dict(num_gpus=1),   # 运行配置，用于指定资源需求
    )
]
```

## 启动评测

首先，我们可以使用 debug 模式启动任务，以检查模型加载、数据集读取是否出现异常，如未正确读取缓存等。

```shell
python run.py configs/llama.py -w outputs/llama --debug
```

但 `--debug` 模式下只能逐一序列执行任务，因此检查无误后，可关闭 `--debug` 模式，使程序充分利用多卡资源

```shell
python run.py configs/llama.py -w outputs/llama
```

以下是一些与评测相关的参数，可以帮助你根据自己的环境情况配置更高效的推理任务。

- `-w outputs/llama`: 评测日志及结果保存目录
- `-r`: 重启上一次（中断的）评测
- `--mode all`: 指定进行某一阶段的任务
  - all: 进行全阶段评测，包括推理和评估
  - infer: 仅进行各个数据集上的推理
  - eval: 仅基于推理结果进行评估
  - viz: 仅展示评估结果
- `--max-partition-size 2000`: 数据集拆分尺寸，部分数据集可能比较大，利用此参数将其拆分成多个子任务，能有效利用资源。但如果拆分过细，则可能因为模型本身加载时间过长，反而速度更慢
- `--max-num-workers 32`: 最大并行启动任务数，在 Slurm 等分布式环境中，该参数用于指定最大提交任务数；在本地环境中，该参数用于指定最大并行执行的任务数，注意实际并行执行任务数受制于 GPU 等资源数，并不一定为该数字。

如果你不是在本机进行评测，而是使用 slurm 集群，可以指定如下参数：

- `--slurm`: 使用 slurm 在集群提交任务
- `--partition my_part`: slurm 集群分区
- `--retry 2`: 任务出错重试次数

## 获取评测结果

评测完成后，会打印评测结果表格如下：

```text
dataset    version    metric    mode      llama-7b
---------  ---------  --------  ------  ----------
piqa       1cf9f0     accuracy  ppl          77.75
siqa       e78df3     accuracy  gen          36.08
```

另外，会在结果保存目录的 `summary` 文件夹中保存 txt 和 csv 格式的结果文件。
