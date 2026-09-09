# 数据来源、缓存与离线运行

OpenCompass 没有一个覆盖所有数据集的统一下载器。实际读取行为由数据集配置的 `type`、`path` 及其 Dataset 类共同决定。

## 两类常见缓存变量

| 变量                 | 使用者                           | 典型作用                                      |
| -------------------- | -------------------------------- | --------------------------------------------- |
| `COMPASS_DATA_CACHE` | OpenCompass 的 `get_data_path()` | 为相对的 OpenCompass 数据路径添加缓存根目录   |
| `HF_DATASETS_CACHE`  | Hugging Face `datasets`          | 存放 `load_dataset()` 下载和生成的 Arrow 缓存 |
| `HF_HOME`            | Hugging Face Hub 生态            | 统一模型、Hub 和数据集缓存根目录              |

例如配置传入 `./data/fold`，并由 Dataset 类调用 `get_data_path()` 时，设置 `COMPASS_DATA_CACHE=/cache/compass` 后会尝试读取 `/cache/compass/./data/fold`。如果目标不存在，OpenCompass 只会在内置下载映射能够识别该数据集时自动下载；否则会报错。

如果 Dataset 类直接调用：

```python
from datasets import load_dataset

load_dataset('organization/dataset-name')
```

则 Hugging Face 按自身规则使用 `HF_DATASETS_CACHE`/`HF_HOME`。`COMPASS_DATA_CACHE` 不会自动重定向这类缓存。

## 推荐目录设置

共享环境中可以分别设置：

```bash
export COMPASS_DATA_CACHE=/shared/opencompass-cache
export HF_HOME=/shared/huggingface-cache
export HF_DATASETS_CACHE=/shared/huggingface-cache/datasets
```

确保运行用户拥有读写权限，并避免多个不兼容版本同时改写同一缓存。只读缓存适合稳定生产环境，但首次生成 Arrow 文件或解压数据时仍需要可写空间。

## 自动加载数据

本地缺少数据文件时，OpenCompass 不会统一转向网络下载，实际行为由数据集加载代码的实现决定，分为三类：

1. **自动下载**：OpenCompass 内置数据集通过 `get_data_path()` 解析路径，仅当数据集被收录在内部下载映射表中时才会自动下载；未收录的数据集会直接报错，传入的路径也不会被当作 Hugging Face 仓库名处理。
2. **交由 Hugging Face 处理**：直接调用 `load_dataset('org/name')` 的数据集，其下载与缓存由 Hugging Face 负责（使用 `HF_HOME`、`HF_DATASETS_CACHE`），不受 `COMPASS_DATA_CACHE` 影响。
3. **加载失败**：绝对路径以及映射表之外的相对路径在本地缺失时直接报错终止。

判断某个数据集属于哪一类，应查阅其 Dataset 类的 `load()` 实现，确认配置实际传入的 `path`、数据加载走上述哪条路径、本地缺失时的行为，以及除主数据文件外是否还依赖说明文档、压缩包或官方评测资源。
