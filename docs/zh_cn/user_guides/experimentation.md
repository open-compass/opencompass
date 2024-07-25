# 任务运行和监控

## 评测任务发起

评测任务的程序入口为 `run.py`，使用方法如下：

```shell
python run.py $EXP {--slurm | --dlc | None} [-p PARTITION] [-q QUOTATYPE] [--debug] [-m MODE] [-r [REUSE]] [-w WORKDIR] [-l] [--dry-run] [--dump-eval-details]
```

任务配置 (`$EXP`)：

- `run.py` 允许接受一个 .py 配置文件作为任务相关参数，里面需要包含 `datasets` 和 `models` 字段。

  ```bash
  python run.py configs/eval_demo.py
  ```

- 如果不传入配置文件，用户也可以通过 `--models MODEL1 MODEL2 ...` 和 `--datasets DATASET1 DATASET2 ...` 来指定模型和数据集:

  ```bash
  python run.py --models hf_opt_350m hf_opt_125m --datasets siqa_gen winograd_ppl
  ```

- 对于 HuggingFace 相关模型，用户也可以通过 HuggingFace 参数快速在命令行中定义一个模型，再通过 `--datasets DATASET1 DATASET2 ...` 定义数据集。

  ```bash
  python run.py --datasets siqa_gen winograd_ppl --hf-type base --hf-path huggyllama/llama-7b
  ```

  HuggingFace 全量参数介绍如下：

  - `--hf-path`:  HuggingFace 模型地址
  - `--peft-path`: PEFT 模型地址
  - `--tokenizer-path`: HuggingFace tokenizer 地址（如与模型地址相同，可省略）
  - `--model-kwargs`: 构造 model 的参数
  - `--tokenizer-kwargs`: 构造 tokenizer 的参数
  - `--max-out-len`: 最长生成 token 数
  - `--max-seq-len`: 模型能接受的最大序列长度
  - `--batch-size`: 批次大小
  - `--hf-num-gpus`: 运行模型所需的gpu数

启动方式：

- 本地机器运行: `run.py $EXP`。
- srun运行: `run.py $EXP --slurm -p $PARTITION_name`。
- dlc运行： `run.py $EXP --dlc --aliyun-cfg $AliYun_Cfg`
- 定制化启动: `run.py $EXP`。这里 $EXP 为配置文件，且里面包含 `eval` 和 `infer` 字段，详细配置请参考 [数据分片](./evaluation.md)。

参数解释如下：

- `-p`: 指定 slurm 分区；
- `-q`: 指定 slurm quotatype（默认为 None），可选 reserved, auto, spot。该参数可能仅适用于部分 slurm 的变体；
- `--debug`: 开启时，推理和评测任务会以单进程模式运行，且输出会实时回显，便于调试；
- `-m`: 运行模式，默认为 `all`。可以指定为 `infer` 则仅运行推理，获得输出结果；如果在 `{WORKDIR}` 中已经有模型输出，则指定为 `eval` 仅运行评测，获得评测结果；如果在 `results/` 中已有单项评测结果，则指定为 `viz` 仅运行可视化；指定为 `all` 则同时运行推理和评测。
- `-r`: 重用已有的推理结果。如果后面跟有时间戳，则会复用工作路径下该时间戳的结果；否则则复用指定工作路径下的最新结果。
- `-w`: 指定工作路径，默认为 `./outputs/default`
- `-l`: 打开飞书机器人状态上报。
- `--dry-run`: 开启时，推理和评测任务仅会分发但不会真正运行，便于调试；
- `--dump-eval-details`: 开启时，`results` 下的评测结果中将会包含更加详细的评测结果信息，例如每条样本是否正确等。

以运行模式 `-m all` 为例，整体运行流如下：

1. 读取配置文件，解析出模型、数据集、评估器等配置信息
2. 评测任务主要分为推理 `infer`、评测 `eval` 和可视化 `viz` 三个阶段，其中推理和评测经过 Partitioner 进行任务切分后，交由 Runner 负责并行执行。单个推理和评测任务则被抽象成 `OpenICLInferTask` 和 `OpenICLEvalTask`。
3. 两阶段分别结束后，可视化阶段会读取 `results/` 中的评测结果，生成可视化报告。

## 任务监控：飞书机器人

用户可以通过配置飞书机器人，实现任务状态的实时监控。飞书机器人的设置文档请[参考这里](https://open.feishu.cn/document/ukTMukTMukTM/ucTM5YjL3ETO24yNxkjN?lang=zh-CN#7a28964d)。

配置方式:

1. 打开 `configs/lark.py` 文件，并在文件中加入以下行：

   ```python
   lark_bot_url = 'YOUR_WEBHOOK_URL'
   ```

   通常， Webhook URL 格式如 https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxxxxxxxxxxx 。

2. 在完整的评测配置中继承该文件：

   ```python
     from mmengine.config import read_base

     with read_base():
         from .lark import lark_bot_url

   ```

3. 为了避免机器人频繁发消息形成骚扰，默认运行时状态不会自动上报。有需要时，可以通过 `-l` 或 `--lark` 启动状态上报：

   ```bash
   python run.py configs/eval_demo.py -p {PARTITION} -l
   ```

## 运行结果

所有运行结果会默认放在`outputs/default/`目录下，目录结构如下所示：

```text
outputs/default/
├── 20200220_120000
├── ...
├── 20230220_183030
│   ├── configs
│   ├── logs
│   │   ├── eval
│   │   └── infer
│   ├── predictions
│   │   └── MODEL1
│   └── results
│       └── MODEL1
```

其中，每一个时间戳中存在以下内容：

- configs 文件夹，用于存放以这个时间戳为输出目录的每次运行对应的配置文件；
- logs 文件夹，用于存放推理和评测两个阶段的输出日志文件，各个文件夹内会以模型为子文件夹存放日志；
- predicitions 文件夹，用于存放推理 json 结果，以模型为子文件夹；
- results 文件夹，用于存放评测 json 结果，以模型为子文件夹

另外，所有指定-r 但是没有指定对应时间戳将会按照排序选择最新的文件夹作为输出目录。

## Summerizer介绍 （待更新）
