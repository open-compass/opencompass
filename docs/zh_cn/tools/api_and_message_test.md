# 接口与消息格式测试

## API Model Tester

```bash
python tools/test_api_model.py path/to/model_config.py -n
```

该工具构建 API 模型并发送内置测试 Prompt，用于验证配置解析、鉴权、消息协议和基础生成。它不能代替正式 Dataset 的 Prompt 检查，也不能证明限流下的高并发稳定性。

## ChatML 格式检查

仓库的 `tools/chatml_format_test.py` 用于检查相关消息格式。运行前先查看当前参数：

```bash
python tools/chatml_format_test.py --help
```

## 测试顺序

1. 单请求验证 endpoint、模型名和密钥；
2. 用 [Prompt Viewer](../prompt/debugging.md) 检查正式 Dataset 输入；
3. 小样本完整执行推理与评测；
4. 逐步增加内部 `max_workers` 和 Runner 并发；
5. 记录 429、超时、空回复、重试次数与实际费用。

密钥应通过环境变量或受控密钥文件提供，不要写入测试输出和 Git 配置。
