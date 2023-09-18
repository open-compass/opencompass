# 实用工具

## Prompt Viewer

本工具允许你在不启动完整训练流程的情况下，直接查看生成的 prompt。如果传入的配置仅为数据集配置（如 `configs/datasets/nq/nq_gen_3dcea1.py`），则展示数据集配置中定义的原始 prompt。若为完整的评测配置（包含模型和数据集），则会展示所选模型运行时实际接收到的 prompt。

运行方式：

```bash
python tools/prompt_viewer.py CONFIG_PATH [-n] [-a] [-p PATTERN]
```

- `-n`: 不进入交互模式，默认选择第一个 model （如有）和 dataset。
- `-a`: 查看配置中所有模型和所有数据集组合接收到的 prompt。
- `-p PATTERN`: 不进入交互模式，选择所有与传入正则表达式匹配的数据集。

## Case Analyzer

本工具在已有评测结果的基础上，产出推理错误样本以及带有标注信息的全量样本。

运行方式：

```bash
python tools/case_analyzer.py CONFIG_PATH [-w WORK_DIR]
```

- `-w`：工作路径，默认为 `'./outputs/default'`。

## Lark Bot

用户可以通过配置飞书机器人，实现任务状态的实时监控。飞书机器人的设置文档请[参考这里](https://open.feishu.cn/document/ukTMukTMukTM/ucTM5YjL3ETO24yNxkjN?lang=zh-CN#7a28964d)。

配置方式:

- 打开 `configs/secrets.py` 文件，并在文件中加入以下行：

  ```python
  lark_bot_url = 'YOUR_WEBHOOK_URL'
  ```

  通常， Webhook URL 格式如 https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxxxxxxxxxxx 。

- 在完整的评测配置中继承该文件：

  ```python
    _base_ = [
        'secrets.py',
        ...
    ]
  ```

  实例可见 `configs/eval.py`。

- 为了避免机器人频繁发消息形成骚扰，默认运行时状态不会自动上报。有需要时，可以通过 `-l` 或 `--lark` 启动状态上报：

  ```bash
  python run.py configs/eval_demo.py -l
  ```

## API Model Tester

本工具可以快速测试 API 模型的功能是否正常。

运行方式：

```bash
python tools/test_api_model.py [CONFIG_PATH] -n
```

## Prediction Merger

本工具可以合并由于 `partitioner` 而产生的分片推理结果。

运行方式：

```bash
python tools/prediction_merger.py CONFIG_PATH [-w WORK_DIR]
```

- `-w`：工作路径，默认为 `'./outputs/default'`。

## List Configs

本工具可以列出或搜索所有可用的模型和数据集配置，且支持模糊搜索，便于结合 `run.py` 使用。

运行方式：

```bash
python tools/list_configs.py [PATTERN1] [PATTERN2] [...]
```

若运行时不加任何参数，则默认列出所有在 `configs/models` 和 `configs/dataset` 下的模型配置。

用户同样可以传入任意数量的参数，脚本会列出所有跟传入字符串相关的配置，支持模糊搜索及 * 号匹配。如下面的命令会列出所有跟 `mmlu` 和 `llama` 相关的配置：

```bash
python tools/list_configs.py mmlu llama
```

它的输出可以是：

```text
+-----------------+-----------------------------------+
| Model           | Config Path                       |
|-----------------+-----------------------------------|
| hf_llama2_13b   | configs/models/hf_llama2_13b.py   |
| hf_llama2_70b   | configs/models/hf_llama2_70b.py   |
| hf_llama2_7b    | configs/models/hf_llama2_7b.py    |
| hf_llama_13b    | configs/models/hf_llama_13b.py    |
| hf_llama_30b    | configs/models/hf_llama_30b.py    |
| hf_llama_65b    | configs/models/hf_llama_65b.py    |
| hf_llama_7b     | configs/models/hf_llama_7b.py     |
| llama2_13b_chat | configs/models/llama2_13b_chat.py |
| llama2_70b_chat | configs/models/llama2_70b_chat.py |
| llama2_7b_chat  | configs/models/llama2_7b_chat.py  |
+-----------------+-----------------------------------+
+-------------------+---------------------------------------------------+
| Dataset           | Config Path                                       |
|-------------------+---------------------------------------------------|
| cmmlu_gen         | configs/datasets/cmmlu/cmmlu_gen.py               |
| cmmlu_gen_ffe7c0  | configs/datasets/cmmlu/cmmlu_gen_ffe7c0.py        |
| cmmlu_ppl         | configs/datasets/cmmlu/cmmlu_ppl.py               |
| cmmlu_ppl_fd1f2f  | configs/datasets/cmmlu/cmmlu_ppl_fd1f2f.py        |
| mmlu_gen          | configs/datasets/mmlu/mmlu_gen.py                 |
| mmlu_gen_23a9a9   | configs/datasets/mmlu/mmlu_gen_23a9a9.py          |
| mmlu_gen_5d1409   | configs/datasets/mmlu/mmlu_gen_5d1409.py          |
| mmlu_gen_79e572   | configs/datasets/mmlu/mmlu_gen_79e572.py          |
| mmlu_gen_a484b3   | configs/datasets/mmlu/mmlu_gen_a484b3.py          |
| mmlu_ppl          | configs/datasets/mmlu/mmlu_ppl.py                 |
| mmlu_ppl_ac766d   | configs/datasets/mmlu/mmlu_ppl_ac766d.py          |
+-------------------+---------------------------------------------------+
```

## Dataset Suffix Updater

本工具可以快速修改 `configs/dataset` 目录下的配置文件后缀，使其符合提示词哈希命名规范。

运行方式：

```bash
python tools/update_dataset_suffix.py
```
