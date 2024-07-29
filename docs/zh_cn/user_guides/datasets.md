# 配置数据集

本节教程主要关注如何选择和配置所需要的数据集。请确保你已按照[数据集准备](../get_started/installation.md#数据集准备)中的步骤下载好数据集。

## 数据集配置文件目录结构

首先简单介绍一下 OpenCompass `configs/datasets` 目录下的结构，如下所示：

```text
configs/datasets/
├── agieval
├── apps
├── ARC_c
├── ...
├── CLUE_afqmc  # 数据集
│   ├── CLUE_afqmc_gen_901306.py  # 不同版本数据集配置文件
│   ├── CLUE_afqmc_gen.py
│   ├── CLUE_afqmc_ppl_378c5b.py
│   ├── CLUE_afqmc_ppl_6507d7.py
│   ├── CLUE_afqmc_ppl_7b0c1e.py
│   └── CLUE_afqmc_ppl.py
├── ...
├── XLSum
├── Xsum
└── z_bench
```

在 `configs/datasets` 目录结构下，我们直接展平所有数据集，在各个数据集对应的文件夹下存在多个数据集配置。

数据集配置文件名由以下命名方式构成 `{数据集名称}_{评测方式}_{prompt版本号}.py`，以 `CLUE_afqmc/CLUE_afqmc_gen_db509b.py` 为例，该配置文件则为中文通用能力下的 `CLUE_afqmc` 数据集，对应的评测方式为 `gen`，即生成式评测，对应的prompt版本号为 `db509b`；同样的， `CLUE_afqmc_ppl_00b348.py` 指评测方式为`ppl`即判别式评测，prompt版本号为 `00b348` 。

除此之外，不带版本号的文件，例如： `CLUE_afqmc_gen.py` 则指向该评测方式最新的prompt配置文件，通常来说会是精度最高的prompt。

## 数据集选择

在各个数据集配置文件中，数据集将会被定义在 `{}_datasets` 变量当中，例如下面 `CLUE_afqmc/CLUE_afqmc_gen_db509b.py` 中的 `afqmc_datasets`。

```python
afqmc_datasets = [
    dict(
        abbr="afqmc-dev",
        type=AFQMCDatasetV2,
        path="./data/CLUE/AFQMC/dev.json",
        reader_cfg=afqmc_reader_cfg,
        infer_cfg=afqmc_infer_cfg,
        eval_cfg=afqmc_eval_cfg,
    ),
]
```

以及 `CLUE_cmnli/CLUE_cmnli_ppl_b78ad4.py` 中的 `cmnli_datasets`。

```python
cmnli_datasets = [
    dict(
        type=HFDataset,
        abbr='cmnli',
        path='json',
        split='train',
        data_files='./data/CLUE/cmnli/cmnli_public/dev.json',
        reader_cfg=cmnli_reader_cfg,
        infer_cfg=cmnli_infer_cfg,
        eval_cfg=cmnli_eval_cfg)
]
```

以上述两个数据集为例， 如果用户想同时评测这两个数据集，可以在 `configs` 目录下新建一个配置文件，我们使用  `mmengine` 配置中直接import的机制来构建数据集部分的参数，如下所示：

```python
from mmengine.config import read_base

with read_base():
    from .datasets.CLUE_afqmc.CLUE_afqmc_gen_db509b import afqmc_datasets
    from .datasets.CLUE_cmnli.CLUE_cmnli_ppl_b78ad4 import cmnli_datasets

datasets = []
datasets += afqmc_datasets
datasets += cmnli_datasets
```

用户可以根据需要，选择不同能力不同数据集以及不同评测方式的配置文件来构建评测脚本中数据集的部分。

有关如何启动评测任务，以及如何评测自建数据集可以参考相关文档。
