# Configure Datasets

This tutorial mainly focuses on selecting datasets supported by OpenCompass and preparing their configs files. Please make sure you have downloaded the datasets following the steps in [Dataset Preparation](../get_started/installation.md#dataset-preparation).

## Directory Structure of Dataset Configuration Files

First, let's introduce the structure under the `configs/datasets` directory in OpenCompass, as shown below:

```
configs/datasets/
├── agieval
├── apps
├── ARC_c
├── ...
├── CLUE_afqmc  # dataset
│   ├── CLUE_afqmc_gen_901306.py  # different version of config
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

In the `configs/datasets` directory structure, we flatten all datasets directly, and there are multiple dataset configurations within the corresponding folders for each dataset.

The naming of the dataset configuration file is made up of `{dataset name}_{evaluation method}_{prompt version number}.py`. For example, `CLUE_afqmc/CLUE_afqmc_gen_db509b.py`, this configuration file is the `CLUE_afqmc` dataset under the Chinese universal ability, the corresponding evaluation method is `gen`, i.e., generative evaluation, and the corresponding prompt version number is `db509b`; similarly, `CLUE_afqmc_ppl_00b348.py` indicates that the evaluation method is `ppl`, i.e., discriminative evaluation, and the prompt version number is `00b348`.

In addition, files without a version number, such as: `CLUE_afqmc_gen.py`, point to the latest prompt configuration file of that evaluation method, which is usually the most accurate prompt.

## Dataset Selection

In each dataset configuration file, the dataset will be defined in the `{}_datasets` variable, such as `afqmc_datasets` in `CLUE_afqmc/CLUE_afqmc_gen_db509b.py`.

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

And `cmnli_datasets` in `CLUE_cmnli/CLUE_cmnli_ppl_b78ad4.py`.

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

Take these two datasets as examples. If users want to evaluate these two datasets at the same time, they can create a new configuration file in the `configs` directory. We use the import mechanism in the `mmengine` configuration to build the part of the dataset parameters in the evaluation script, as shown below:

```python
from mmengine.config import read_base

with read_base():
    from .datasets.CLUE_afqmc.CLUE_afqmc_gen_db509b import afqmc_datasets
    from .datasets.CLUE_cmnli.CLUE_cmnli_ppl_b78ad4 import cmnli_datasets

datasets = []
datasets += afqmc_datasets
datasets += cmnli_datasets
```

Users can choose different abilities, different datasets and different evaluation methods configuration files to build the part of the dataset in the evaluation script according to their needs.

For information on how to start an evaluation task and how to evaluate self-built datasets, please refer to the relevant documents.
