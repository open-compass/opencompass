# 通知与任务监控

OpenCompass 可以通过飞书机器人报告任务状态。Webhook 属于凭证，不应提交到仓库。

在私有配置文件中声明：

```python
lark_bot_url = 'YOUR_WEBHOOK_URL'
```

在实验配置中使用 `read_base()` 导入：

```python
from mmengine.config import read_base

with read_base():
    from .secrets import lark_bot_url
```

启动时显式打开通知：

```bash
opencompass my_eval.py --lark
```

通知只反映调度状态，不能替代结果验收。任务结束后仍应检查日志、预测数量、评测结果和汇总文件。使用并发评测监听时，还需关注工作目录中的心跳与推理状态文件。
