# Preparing and Selecting Datasets

This section of the tutorial mainly focuses on how to prepare the datasets supported by OpenCompass and build configuration files to complete dataset selection.

## Directory Structure of Dataset Configuration Files

First, let's introduce the structure under the `configs/datasets` directory in OpenCompass, as shown below:

```
configs/datasets/
├── ChineseUniversal  # Ability dimension
│   ├── CLUE_afqmc  # Dataset under this dimension
│   │   ├── CLUE_afqmc_gen_db509b.py  # Different configuration files for this dataset
│   │   ├── CLUE_afqmc_gen.py
│   │   ├── CLUE_afqmc_ppl_00b348.py
│   │   ├── CLUE_afqmc_ppl_2313cf.py
│   │   └── CLUE_afqmc_ppl.py
│   ├── CLUE_C3
│   │   ├── ...
│   ├── ...
├── Coding
├── collections
├── Completion
├── EnglishUniversal
├── Exam
├── glm
├── LongText
├── MISC
├── NLG
├── QA
├── Reasoning
├── Security
└── Translation
```

In the `configs/datasets` directory structure, we have divided the datasets into over ten dimensions based on ability dimensions, such as: Chinese and English Universal, Exam, QA, Reasoning, Security, etc. Each dimension contains a series of datasets, and there are multiple dataset configurations in the corresponding folder of each dataset.

The naming of the dataset configuration file is made up of `{dataset name}_{evaluation method}_{prompt version number}.py`. For example, `ChineseUniversal/CLUE_afqmc/CLUE_afqmc_gen_db509b.py`, this configuration file is the `CLUE_afqmc` dataset under the Chinese universal ability, the corresponding evaluation method is `gen`, i.e., generative evaluation, and the corresponding prompt version number is `db509b`; similarly, `CLUE_afqmc_ppl_00b348.py` indicates that the evaluation method is `ppl`, i.e., discriminative evaluation, and the prompt version number is `00b348`.

In addition, files without a version number, such as: `CLUE_afqmc_gen.py`, point to the latest prompt configuration file of that evaluation method, which is usually the most accurate prompt.

## Dataset Preparation

The datasets supported by OpenCompass mainly include two parts:

1. Huggingface Dataset

[Huggingface Dataset](https://huggingface.co/datasets) provides a large number of datasets. OpenCompass has supported most of the datasets commonly used for performance comparison, please refer to `configs/dataset` for the specific list of supported datasets.

2. OpenCompass Self-built Datasets

In addition to supporting Huggingface's existing datasets, OpenCompass also provides some self-built CN datasets. In the future, a dataset-related Repo will be provided for users to download and use. Following the instructions in the document to place the datasets uniformly in the `./data` directory can complete dataset preparation.

It is important to note that the Repo not only contains self-built datasets, but also includes some HF-supported datasets for testing convenience.

## Dataset Selection

In each dataset configuration file, the dataset will be defined in the `{}_datasets` variable, such as `afqmc_datasets` in `ChineseUniversal/CLUE_afqmc/CLUE_afqmc_gen_db509b.py`.

```python
afqmc_datasets = [
    dict(
        abbr="afqmc-dev",
        type=AFQMCDataset_V2,
        path="./data/CLUE/AFQMC/dev.json",
        reader_cfg=afqmc_reader_cfg,
        infer_cfg=afqmc_infer_cfg,
        eval_cfg=afqmc_eval_cfg,
    ),
]
```

And `afqmc_datasets` in `ChineseUniversal/CLUE_cmnli/CLUE_cmnli_ppl_b78ad4.py`.

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
