# 数据集下载与缓存

OpenCompass 没有一个覆盖所有数据集的统一下载器。实际读取行为由数据集配置的 `type`、`path` 及其 Dataset 类共同决定。

## 自动加载数据

数据集没有统一的下载方式，可能从 OpenCompass OSS、Hugging Face、ModelScope 或数据集官方地址获取，也可能要求用户预先准备本地文件。实际行为由数据集配置及其 Dataset 类的 `load()` 实现决定。

对于调用 `get_data_path()` 的数据集，配置中的 `path` 通常是逻辑标识；OpenCompass 会根据 `opencompass/utils/datasets_info.py` 中的映射，将其转换为本地路径、Hugging Face 数据集 ID 或 ModelScope 数据集 ID。如果数据集仅提供 Hugging Face 或 ModelScope 下载源，应通过 `DATASET_SOURCE` 选择对应路由：

```bash
export DATASET_SOURCE=HF          # 使用 Hugging Face
# 或
export DATASET_SOURCE=ModelScope  # 使用 ModelScope
```

变量值区分大小写。未设置 `DATASET_SOURCE` 时，`get_data_path()` 默认解析为本地路径；文件不存在且该数据集已登记 OSS 下载地址时，OpenCompass 会尝试从 OSS 下载。并非所有 Dataset 类都支持全部路由，使用前应检查其 `load()` 实现及 `datasets_info.py` 中相应来源的 ID 是否存在。直接调用 `load_dataset()` 等第三方接口的数据集不经过这套路由，无需专门设置 `DATASET_SOURCE`。

## 数据缓存环境变量

下载来源与缓存目录是两个不同概念：`DATASET_SOURCE` 决定从哪里加载，以下变量决定数据保存或查找的位置。

| 变量                 | 作用                                                                                                                         |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `COMPASS_DATA_CACHE` | OpenCompass 本地数据的缓存根目录。`get_data_path()` 会将映射得到的相对本地路径拼接到该目录下，OSS 数据也会下载并解压到其中。 |
| `HF_DATASETS_CACHE`  | Hugging Face `datasets` 的缓存目录，用于保存下载数据及生成的 Arrow 文件；不影响 OpenCompass 本地数据路径。                   |
| `LMUData`            | VLMEvalKit 的数据根目录，用于保存官方 TSV 数据和图片等多模态资源；未设置时默认为相对启动目录的 `data/vlmevalkit`。           |

共享环境中可以分别设置：

```bash
export COMPASS_DATA_CACHE=/shared/opencompass-cache
export HF_DATASETS_CACHE=/shared/huggingface-cache/datasets
export LMUData=/shared/vlmevalkit-cache
```

`LMUData` 仅适用于通过 VLMEvalKit 桥接的数据集，不会改变其他多模态数据集的路径，详细机制参阅[使用 VLMEvalKit 数据集与官方评测](../evaluation/vlmevalkit.md)。确保运行用户对首次下载、解压或生成缓存的目录拥有写权限；离线运行前应确认数据文件和其他相关资源均已准备完成。
