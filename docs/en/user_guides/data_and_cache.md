# Dataset Download and Caching

OpenCompass does not provide one unified downloader covering every dataset. Actual loading behavior is jointly determined by `type`, `path`, and the Dataset class in the dataset configuration.

## Automatic Data Loading

Datasets do not share one download mechanism. Data may come from OpenCompass OSS, Hugging Face, ModelScope, or an official dataset URL, and some datasets require users to prepare local files in advance. The dataset configuration and the Dataset class's `load()` implementation determine the actual behavior.

For datasets that call `get_data_path()`, the configured `path` is usually a logical identifier. OpenCompass uses the mapping in `opencompass/utils/datasets_info.py` to resolve it to a local path, a Hugging Face dataset ID, or a ModelScope dataset ID. If a dataset provides only a Hugging Face or ModelScope source, select that route with `DATASET_SOURCE`:

```bash
export DATASET_SOURCE=HF          # Use Hugging Face
# or
export DATASET_SOURCE=ModelScope  # Use ModelScope
```

Values are case-sensitive. When `DATASET_SOURCE` is unset, `get_data_path()` resolves to a local path by default. If the file does not exist and an OSS URL is registered for the dataset, OpenCompass attempts to download it from OSS. Not every Dataset class supports every route; check its `load()` implementation and whether the corresponding source ID exists in `datasets_info.py`. Datasets that directly call third-party interfaces such as `load_dataset()` do not use this routing mechanism and do not require a dedicated `DATASET_SOURCE` setting.

## Data-Cache Environment Variables

The download source and cache directory are separate concepts: `DATASET_SOURCE` determines where data is loaded from, while the following variables determine where data is stored or located.

| Variable             | Purpose                                                                                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `COMPASS_DATA_CACHE` | Root directory for local OpenCompass data. `get_data_path()` appends the resolved relative local path to this directory; OSS data is also downloaded and extracted here. |
| `HF_DATASETS_CACHE`  | Cache directory for Hugging Face `datasets`, including downloaded data and generated Arrow files. It does not affect OpenCompass local-data paths.                       |
| `LMUData`            | VLMEvalKit data root for official TSV files, images, and other multimodal resources. When unset, it defaults to `data/vlmevalkit` relative to the launch directory.      |

In a shared environment, configure the directories separately:

```bash
export COMPASS_DATA_CACHE=/shared/opencompass-cache
export HF_DATASETS_CACHE=/shared/huggingface-cache/datasets
export LMUData=/shared/vlmevalkit-cache
```

`LMUData` applies only to datasets bridged through VLMEvalKit and does not change the paths of other multimodal datasets. See [Using VLMEvalKit Datasets and Official Evaluation](../evaluation/vlmevalkit.md) for details. Ensure that the runtime user can write to directories used for first-time downloads, extraction, or cache generation. Before offline execution, verify that all data files and related resources have been prepared.
