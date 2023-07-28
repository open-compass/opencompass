# Task Execution and Monitoring

## Launching an Evaluation Task

The program entry for the evaluation task is `run.py`, its usage is as follows:

```shell
python run.py $Config {--slurm | --dlc | None} [-p PARTITION] [-q QUOTATYPE] [--debug] [-m MODE] [-r [REUSE]] [-w WORKDIR] [-l] [--dry-run]
```

Here are some examples for launching the task in different environments:

- Running locally: `run.py $Config`, where `$Config` does not contain fields 'eval' and 'infer'.
- Running with Slurm: `run.py $Config --slurm -p $PARTITION_name`.
- Running on ALiYun DLC: `run.py $Config --dlc --aliyun-cfg $AliYun_Cfg`, tutorial will come later.
- Customized run: `run.py $Config`, where `$Config` contains fields 'eval' and 'infer', and you are able to customize the way how each task will be split and launched. See [Evaluation document](./evaluation.md).

The parameter explanation is as follows:

- `-p`: Specify the slurm partition;
- `-q`: Specify the slurm quotatype (default is None), with optional values being reserved, auto, spot. This parameter may only be used in some slurm variants;
- `--debug`: When enabled, inference and evaluation tasks will run in single-process mode, and output will be echoed in real-time for debugging;
- `-m`: Running mode, default is `all`. It can be specified as `infer` to only run inference and obtain output results; if there are already model outputs in `{WORKDIR}`, it can be specified as `eval` to only run evaluation and obtain evaluation results; if the evaluation results are ready, it can be specified as `viz` to only run visualization, which summarizes the results in tables; if specified as `all`, a full run will be performed, which includes inference, evaluation, and visualization.
- `-r`: Reuse existing inference results, and skip the finished tasks. If followed by a timestamp, the result under that timestamp in the workspace path will be reused; otherwise, the latest result in the specified workspace path will be reused.
- `-w`: Specify the working path, default is `./outputs/default`.
- `-l`: Enable status reporting via Lark bot.
- `--dry-run`: When enabled, inference and evaluation tasks will be dispatched but won't actually run for debugging.

Using run mode `-m all` as an example, the overall execution flow is as follows:

1. Read the configuration file, parse out the model, dataset, evaluator, and other configuration information
2. The evaluation task mainly includes three stages: inference `infer`, evaluation `eval`, and visualization `viz`. After task division by Partitioner, they are handed over to Runner for parallel execution. Individual inference and evaluation tasks are abstracted into `OpenICLInferTask` and `OpenICLEvalTask` respectively.
3. After each stage ends, the visualization stage will read the evaluation results in `results/` to generate a table.

## Task Monitoring: Lark Bot

Users can enable real-time monitoring of task status by setting up a Lark bot. Please refer to [this document](https://open.feishu.cn/document/ukTMukTMukTM/ucTM5YjL3ETO24yNxkjN?lang=zh-CN#7a28964d) for setting up the Lark bot.

Configuration method:

1. Open the `configs/lark.py` file, and add the following line:

   ```python
   lark_bot_url = 'YOUR_WEBHOOK_URL'
   ```

   Typically, the Webhook URL is formatted like this: https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxxxxxxxxxxx .

2. Inherit this file in the complete evaluation configuration:

   ```python
     from mmengine.config import read_base

     with read_base():
         from .lark import lark_bot_url

   ```

3. To avoid frequent messages from the bot becoming a nuisance, status updates are not automatically reported by default. You can start status reporting using `-l` or `--lark` when needed:

   ```bash
   python run.py configs/eval_demo.py -p {PARTITION} -l
   ```

## Run Results

All run results will be placed in `outputs/default/` directory by default, the directory structure is shown below:

```
outputs/default/
├── 20200220_120000
├── ...
├── 20230220_183030
│   ├── configs
│   ├── logs
│   │   ├── eval
│   │   └── infer
│   ├── predictions
│   │   └── MODEL1
│   └── results
│       └── MODEL1
```

Each timestamp contains the following content:

- configs folder, which stores the configuration files corresponding to each run with this timestamp as the output directory;
- logs folder, which stores the output log files of the inference and evaluation phases, each folder will store logs in subfolders by model;
- predictions folder, which stores the inferred json results, with a model subfolder;
- results folder, which stores the evaluated json results, with a model subfolder.

Also, all `-r` without specifying a corresponding timestamp will select the newest folder by sorting as the output directory.

## Introduction of Summerizer (to be updated)
