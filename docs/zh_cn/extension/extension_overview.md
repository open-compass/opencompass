# 扩展机制总览

OpenCompass 的主要组件通过注册表和配置构建。扩展前先判断需求属于哪一层：

| 需求                      | 扩展位置                                |
| ------------------------- | --------------------------------------- |
| 新数据格式或下载逻辑      | Dataset                                 |
| 新 Prompt、检索或生成流程 | PromptTemplate / Retriever / Inferencer |
| 新答案抽取或指标          | Postprocessor / Evaluator               |
| 新模型协议或运行时        | Model                                   |
| 新任务切分与执行环境      | Partitioner / Runner / Task             |
| 新榜单聚合口径            | Summarizer                              |

优先复用已有组件并新增配置；只有现有接口不能表达行为时才增加 Python 类。扩展代码应注册到对应 Registry，并配套最小配置、单元测试、相关文档。

- 新数据：参阅[新增数据集](new_dataset.md)和[快速评测自有数据](custom_dataset.md)；
- 新模型：参阅[新增模型后端](new_model.md)；
- 新评测组件：参阅[新增评测器与汇总器](new_evaluator_and_summarizer.md)。
