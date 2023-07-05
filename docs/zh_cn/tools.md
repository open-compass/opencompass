# 实用工具

## Prompt Viewer

本工具允许你在不启动完整训练流程的情况下，直接查看模型会接收到的 prompt。

运行方式：

```bash
python tools/prompt_viewer.py [CONFIG_PATH]
```

## Case Analyzer

本工具在已有评测结果的基础上，产出推理错误样本以及带有标注信息的全量样本

运行方式：

```bash
python tools/case_analyzer.py [CONFIG_PATH] [-w WORK_DIR]
```

- `-w`：工作路径，默认为 `'./outputs/default'`。

更多细节见 [飞书文档](https://aicarrier.feishu.cn/docx/SgrLdwinion00Kxkzh2czz29nIh)

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
  python run.py configs/eval_demo.py -p {PARTITION} -l
  ```

## API Model Tests

本工具可以快速测试 API Wrapper 的功能是否正常。

运行方式：

```bash
python tools/test_api_model.py [CONFIG_PATH]
```
