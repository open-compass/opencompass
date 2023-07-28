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
