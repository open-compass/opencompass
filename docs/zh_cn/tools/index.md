# 实用工具总览

OpenCompass 的 `tools/` 目录提供配置发现、Prompt 预览、预测检查、API 测试和结果分析脚本。工具读取的配置与目录格式可能随版本变化，使用前应运行 `python tools/<脚本>.py --help`。

- [配置发现与比较](config_discovery.md)
- [预测结果检查与错误分析](prediction_analysis.md)
- [重复输出与响应长度分析](repeat_and_length.md)
- [接口与消息格式测试](api_and_message_test.md)
- [通知与任务监控](monitoring.md)

Prompt 预览的完整流程参阅[提示词预览与调试](../prompt/debugging.md)，任务中断后的处理参阅[任务恢复与复用](../execution/reuse_and_resume.md)。
