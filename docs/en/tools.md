# Useful Tools

## Prompt Viewer

This tool allows you to directly view the generated prompt without starting the full training process. If the passed configuration is only the dataset configuration (such as `configs/datasets/nq/nq_gen.py`), it will display the original prompt defined in the dataset configuration. If it is a complete evaluation configuration (including the model and the dataset), it will display the prompt received by the selected model during operation.

Running method:

```bash
python tools/prompt_viewer.py CONFIG_PATH [-n] [-a] [-p PATTERN]
```

- `-n`: Do not enter interactive mode, select the first model (if any) and dataset by default.
- `-a`: View the prompts received by all models and all dataset combinations in the configuration.
- `-p PATTERN`: Do not enter interactive mode, select all datasets that match the input regular expression.

## Case Analyzer (To be updated)

Based on existing evaluation results, this tool produces inference error samples and full samples with annotation information.

Running method:

```bash
python tools/case_analyzer.py CONFIG_PATH [-w WORK_DIR]
```

- `-w`: Work path, default is `'./outputs/default'`.

## Lark Bot

Users can configure the Lark bot to implement real-time monitoring of task status. Please refer to [this document](https://open.feishu.cn/document/ukTMukTMukTM/ucTM5YjL3ETO24yNxkjN?lang=zh-CN#7a28964d) for setting up the Lark bot.

Configuration method:

- Open the `configs/secrets.py` file, and add the following line to the file:

```python
lark_bot_url = 'YOUR_WEBHOOK_URL'
```

- Normally, the Webhook URL format is like https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxxxxxxxxxxx .

- Inherit this file in the complete evaluation configuration

- To avoid the bot sending messages frequently and causing disturbance, the running status will not be reported automatically by default. If necessary, you can start status reporting through `-l` or `--lark`:

```bash
python run.py configs/eval_demo.py -l
```

## API Model Tester

This tool can quickly test whether the functionality of the API model is normal.

Running method:

```bash
python tools/test_api_model.py [CONFIG_PATH] -n
```

## Prediction Merger

This tool can merge patitioned predictions.

Running method:

```bash
python tools/prediction_merger.py CONFIG_PATH [-w WORK_DIR]
```

- `-w`: Work path, default is `'./outputs/default'`.

## List Configs

This tool can list or search all available model and dataset configurations. It supports fuzzy search, making it convenient for use in conjunction with `run.py`.

Usage:

```bash
python tools/list_configs.py [PATTERN1] [PATTERN2] [...]
```

If executed without any parameters, it will list all model configurations in the `configs/models` and `configs/dataset` directories by default.

Users can also pass any number of parameters. The script will list all configurations related to the provided strings, supporting fuzzy search and the use of the * wildcard. For example, the following command will list all configurations related to `mmlu` and `llama`:

```bash
python tools/list_configs.py mmlu llama
```

Its output could be:

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

This tool can quickly modify the suffixes of configuration files located under the `configs/dataset` directory, aligning them with the naming conventions based on prompt hash.

How to run:

```bash
python tools/update_dataset_suffix.py
```
