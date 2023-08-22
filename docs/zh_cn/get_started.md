# 安装

1. 准备 OpenCompass 运行环境：

   ```bash
   conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
   conda activate opencompass
   ```

   如果你希望自定义 PyTorch 版本或相关的 CUDA 版本，请参考 [官方文档](https://pytorch.org/get-started/locally/) 准备 PyTorch 环境。需要注意的是，OpenCompass 要求 `pytorch>=1.13`。

2. 安装 OpenCompass：

   ```bash
   git clone https://github.com/InternLM/opencompass.git
   cd opencompass
   pip install -e .
   ```

3. 安装 humaneval（可选）：

   如果你需要**在 humaneval 数据集上评估模型代码能力**，请执行此步骤，否则忽略这一步。

   <details>
   <summary><b>点击查看详细</b></summary>

   ```bash
   git clone https://github.com/openai/human-eval.git
   cd human-eval
   pip install -r requirements.txt
   pip install -e .
   cd ..
   ```

   请仔细阅读 `human_eval/execution.py` **第48-57行**的注释，了解执行模型生成的代码可能存在的风险，如果接受这些风险，请取消**第58行**的注释，启用代码执行评测。

   </details>

4. 安装 Llama（可选）：

   如果你需要**使用官方实现评测 Llama / Llama-2 / Llama-2-chat 模型**，请执行此步骤，否则忽略这一步。

   <details>
   <summary><b>点击查看详细</b></summary>

   ```bash
   git clone https://github.com/facebookresearch/llama.git
   cd llama
   pip install -r requirements.txt
   pip install -e .
   cd ..
   ```

   你可以在 `configs/models` 下找到所有 Llama / Llama-2 / Llama-2-chat 模型的配置文件示例。([示例](https://github.com/InternLM/opencompass/blob/eb4822a94d624a4e16db03adeb7a59bbd10c2012/configs/models/llama2_7b_chat.py))

   </details>

# 数据集准备

OpenCompass 支持的数据集主要包括两个部分：

1. Huggingface 数据集： [Huggingface Dataset](https://huggingface.co/datasets) 提供了大量的数据集，这部分数据集运行时会**自动下载**。

2. 自建以及第三方数据集：OpenCompass 还提供了一些第三方数据集及自建**中文**数据集。运行以下命令**手动下载解压**。

在 OpenCompass 项目根目录下运行下面命令，将数据集准备至 `${OpenCompass}/data` 目录下：

```bash
wget https://github.com/InternLM/opencompass/releases/download/0.1.1/OpenCompassData.zip
unzip OpenCompassData.zip
```

OpenCompass 已经支持了大多数常用于性能比较的数据集，具体支持的数据集列表请直接在 `configs/datasets` 下进行查找。

# 快速上手

OpenCompass 的评测以配置文件为中心，必须包含 `datasets` 和 `models` 字段，配置需要评测的模型以及数据集，使用入口 'run.py' 启动。

我们会以测试 [OPT-125M](https://huggingface.co/facebook/opt-125m) 以及 [OPT-350M](https://huggingface.co/facebook/opt-350m) 预训练基座模型在 [SIQA](https://huggingface.co/datasets/social_i_qa) 和 [Winograd](https://huggingface.co/datasets/winogrande) 上的性能为例，带领你熟悉 OpenCompass 的一些基本功能。
本次的测试的配置文件为[configs/eval_demo.py](https://github.com/InternLM/opencompass/blob/main/configs/eval_demo.py)。

运行前确保已经安装了 OpenCompass，本实验可以在单张 _GTX-1660-6G_ 显卡上成功运行。
更大参数的模型，如 Llama-7B, 可参考 [configs](https://github.com/InternLM/opencompass/tree/main/configs) 中其他例子。

由于 OpenCompass 默认使用并行的方式进行评测，为了便于及时发现问题，我们可以在首次启动时使用 debug 模式运行，该模式会将任务串行执行，并会实时输出任务的执行进度。

```bash
python run.py configs/eval_demo.py -w outputs/demo --debug
```

如果一切正常，屏幕上会出现 "Starting inference process"：

```bash
Loading cached processed dataset at .cache/huggingface/datasets/social_i_qa/default/0.1.0/674d85e42ac7430d3dcd4de7007feaffcb1527c535121e09bab2803fbcc925f8/cache-742512eab30e8c9c.arrow
[2023-07-12 18:23:55,076] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...
```

此时可以使用 `ctrl+c` 中断 debug 模式的执行，并运行以下命令进行并行评测：

```bash
python run.py configs/eval_demo.py -w outputs/demo
```

运行 demo 期间，我们来介绍一下本案例中的配置内容以及启动选项。

## 步骤详解

### 数据集列表 `datasets`

以下为 `configs/eval_demo.py` 中与数据集相关的配置片段：

```python
from mmengine.config import read_base  # 使用 mmengine.read_base() 读取基础配置

with read_base():
    # 直接从预设数据集配置中读取需要的数据集配置
    from .datasets.winograd.winograd_ppl import winograd_datasets  # 读取 Winograd 的配置，基于 PPL (perplexity) 进行评测
    from .datasets.siqa.siqa_gen import siqa_datasets  # 读取 SIQA 的配置，基于生成式进行评测

datasets = [*siqa_datasets, *winograd_datasets]       # 最后 config 需要包含所需的评测数据集列表 datasets
```

[configs/datasets](https://github.com/InternLM/OpenCompass/blob/main/configs/datasets) 包含各种数据集预先定义好的配置文件；
部分数据集文件夹下有 'ppl' 和 'gen' 两类配置文件，表示使用的评估方式，其中 `ppl` 表示使用判别式评测， `gen` 表示使用生成式评测。

[configs/datasets/collections](https://github.com/InternLM/OpenCompass/blob/main/configs/datasets/collections) 存放了各类数据集集合，方便做综合评测。

更多介绍可查看 [数据集配置](./user_guides/dataset_prepare.md)。

### 模型列表 `models`

OpenCompass 支持直接在配置中指定待测试的模型列表，对于 HuggingFace 模型来说，用户通常无需添加代码。下面为相关的配置片段：

```python
# 提供直接使用 HuggingFaceCausalLM 模型的接口
from opencompass.models import HuggingFaceCausalLM

# OPT-350M
opt350m = dict(
       type=HuggingFaceCausalLM,
       # 以下参数为 HuggingFaceCausalLM 相关的初始化参数
       path='facebook/opt-350m',
       tokenizer_path='facebook/opt-350m',
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True),
       model_kwargs=dict(device_map='auto'),
       # 下列参数为所有模型均需设定的初始化参数，非 HuggingFaceCausalLM 独有
       abbr='opt350m',                    # 模型简称，用于结果展示
       max_seq_len=2048,              # 模型能接受的最大序列长度
       max_out_len=100,                   # 最长生成 token 数
       batch_size=64,                     # 批次大小
       run_cfg=dict(num_gpus=1),          # 运行配置，用于指定资源需求
    )

# OPT-125M
opt125m = dict(
       type=HuggingFaceCausalLM,
       # 以下参数为 HuggingFaceCausalLM 的初始化参数
       path='facebook/opt-125m',
       tokenizer_path='facebook/opt-125m',
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True),
       model_kwargs=dict(device_map='auto'),
       # 下列参数为所有模型均需设定的初始化参数，非 HuggingFaceCausalLM 独有
       abbr='opt125m',                # 模型简称，用于结果展示
       max_seq_len=2048,              # 模型能接受的最大序列长度
       max_out_len=100,               # 最长生成 token 数
       batch_size=128,                # 批次大小
       run_cfg=dict(num_gpus=1),      # 运行配置，用于指定资源需求
    )

models = [opt350m, opt125m]
```

HuggingFace 中的 'facebook/opt-350m' 以及 'facebook/opt-125m' 权重会在运行时自动下载。

关于模型配置的更多介绍可阅读 [准备模型](./user_guides/models.md)。

### 启动评测

配置文件准备完毕后，我们可以使用 debug 模式启动任务，以检查模型加载、数据集读取是否出现异常，如未正确读取缓存等。

```shell
python run.py configs/eval_demo.py -w outputs/demo --debug
```

但 `--debug` 模式下只能逐一序列执行任务，因此检查无误后，可关闭 `--debug` 模式，使程序充分利用多卡资源

```shell
python run.py configs/eval_demo.py -w outputs/demo
```

以下是一些与评测相关的参数，可以帮助你根据自己的环境情况配置更高效的推理任务。

- `-w outputs/demo`: 评测日志及结果保存目录。若不指定，则默认为 `outputs/default`
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
- `--partition(-p) my_part`: slurm 集群分区
- `--retry 2`: 任务出错重试次数

```{tip}
这个脚本同样支持将任务提交到阿里云深度学习中心（DLC）上运行，以及更多定制化的评测策略。请参考 [评测任务发起](./user_guides/experimentation.md#评测任务发起) 了解更多细节。
```

## 评测结果

评测完成后，会打印评测结果表格如下：

```text
dataset    version    metric    mode      opt350m    opt125m
---------  ---------  --------  ------  ---------  ---------
siqa       e78df3     accuracy  gen         21.55      12.44
winograd   b6c7ed     accuracy  ppl         51.23      49.82
```

所有过程的日志，预测，以及最终结果会放在 `outputs/demo/` 目录下。目录结构如下所示：

```text
outputs/default/
├── 20200220_120000
├── 20230220_183030   # 一次实验
│   ├── configs       # 每次实验都会在此处存下用于追溯的 config
│   ├── logs          # 运行日志
│   │   ├── eval
│   │   └── infer
│   ├── predictions   # 储存了每个任务的推理结果
│   ├── results       # 储存了每个任务的评测结果
│   └── summary       # 汇总每次实验的所有评测结果
├── ...
```

打印评测结果的过程可被进一步定制化，用于输出一些数据集的平均分 (例如 MMLU, C-Eval 等)。

关于评测结果输出的更多介绍可阅读 [结果展示](./user_guides/summarizer.md)。

## 更多教程

想要更多了解 OpenCompass, 可以点击下列链接学习。

- [数据集配置](./user_guides/dataset_prepare.md)
- [准备模型](./user_guides/models.md)
- [任务运行和监控](./user_guides/experimentation.md)
- [如何调Prompt](./prompt/overview.md)
- [结果展示](./user_guides/summarizer.md)
- [学习配置文件](./user_guides/config.md)
