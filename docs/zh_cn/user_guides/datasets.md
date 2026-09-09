# 数据集选择与配置

OpenCompass 的一个“数据集配置”同时定义数据读取、模型输入和评分规则。同名数据集可以有多个配置变体，不能只根据原始数据名称比较结果。

## 查找和选择配置

```bash
python tools/list_configs.py mmlu gsm8k
```

配置文件通常位于 `opencompass/configs/datasets/<数据集>/`。文件名中的 `gen`、`ppl`、`rawprompt`、few-shot 数量和哈希用于区分评测方案；不带哈希的文件通常是便于引用的入口，但仍应打开文件确认其指向与内容。

不带参数运行 `list_configs.py` 可以查看当前安装版本能够发现的全部模型、数据集和汇总配置：

```bash
python tools/list_configs.py
```

输出第一列的简称可以直接传给 `opencompass --datasets ...`。数据集数量和配置变体会持续变化，因此文档不再维护一份容易过期的静态清单；应以当前代码中的 `opencompass/configs/datasets/` 和工具输出为准。

选择配置时需确认以下信息：

- 数据来源、版本、split 和样本范围；
- 输入字段、答案字段与媒体资源；
- Prompt 类型、few-shot 数量和推理方式；
- Evaluator、答案抽取及后处理规则；
- 是否依赖 Judge、代码沙箱或官方评测服务；
- 数据目录、缓存变量和离线运行要求。

## 数据集配置结构

```python
datasets = [
    dict(
        type=MyDataset,
        abbr='my-dataset',
        path='data/or/hub-id',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
```

- `type` 和 `path` 决定数据从哪里、怎样加载；
- `reader_cfg` 声明输入列、答案列、split 和样本范围；
- `infer_cfg` 声明 Prompt、few-shot Retriever 与 Gen/PPL Inferencer；
- `eval_cfg` 声明预测/答案后处理和 Evaluator。

完整自定义方法参阅[新增数据集](../extension/new_dataset.md)，快速评测 JSON/JSONL/CSV 等自有数据参阅[快速评测自有数据](../extension/custom_dataset.md)。

## 重复运行

CLI 的 `--dataset-num-runs N` 会复制数据集配置并执行多次评测；配置中的 `n`/`k` 还可被部分稳健性指标使用。重复运行只有在生成参数允许随机性、服务端存在波动或指标明确需要多次样本时才有意义。应同时保存每次运行结果和聚合方法，不能只报告平均值。

## 数据位置

数据可能来自本地文件、OpenCompass 数据包、Hugging Face、ModelScope 或数据集自己的下载逻辑。缓存和离线规则详见[数据来源、缓存与离线运行](data_and_cache.md)。
