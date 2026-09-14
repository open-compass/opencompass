# Data Sources, Caches, and Offline Operation

OpenCompass does not provide one unified downloader covering every dataset. Actual loading behavior is jointly determined by `type`, `path`, and the Dataset class in the dataset configuration.

## Common Cache Variables

| Variable             | Consumer                      | Typical purpose                                                   |
| -------------------- | ----------------------------- | ----------------------------------------------------------------- |
| `COMPASS_DATA_CACHE` | OpenCompass `get_data_path()` | Adds a cache root to relative OpenCompass data paths              |
| `HF_DATASETS_CACHE`  | Hugging Face `datasets`       | Stores downloads and generated Arrow caches from `load_dataset()` |
| `HF_HOME`            | Hugging Face Hub ecosystem    | Provides a shared root for model, Hub, and dataset caches         |

For example, if a configuration passes `./data/fold` and its Dataset class calls `get_data_path()`, setting `COMPASS_DATA_CACHE=/cache/compass` makes it try `/cache/compass/./data/fold`. If the target does not exist, OpenCompass downloads it automatically only when the built-in download mapping recognizes the dataset; otherwise it raises an error.

If the Dataset class directly calls:

```python
from datasets import load_dataset

load_dataset('organization/dataset-name')
```

Hugging Face uses `HF_DATASETS_CACHE`/`HF_HOME` according to its own rules. `COMPASS_DATA_CACHE` does not redirect this cache automatically.

## Recommended Directory Settings

In a shared environment, configure them separately:

```bash
export COMPASS_DATA_CACHE=/shared/opencompass-cache
export HF_HOME=/shared/huggingface-cache
export HF_DATASETS_CACHE=/shared/huggingface-cache/datasets
```

Ensure that the runtime user has read/write permission, and avoid letting incompatible versions modify the same cache simultaneously. A read-only cache is suitable for a stable production environment, but generating Arrow files or extracting data for the first time still requires writable space.

## Automatic Data Loading

When local data files are missing, OpenCompass does not uniformly fall back to downloading them. Behavior depends on the dataset loader and falls into three categories:

1. **Automatic download:** An OpenCompass built-in dataset resolves its path through `get_data_path()`. It downloads automatically only if the dataset is present in the internal download mapping; an unlisted dataset raises an error, and the supplied path is not treated as a Hugging Face repository name.
2. **Delegation to Hugging Face:** A dataset that directly calls `load_dataset('org/name')` delegates download and caching to Hugging Face (`HF_HOME` and `HF_DATASETS_CACHE`) and is unaffected by `COMPASS_DATA_CACHE`.
3. **Loading failure:** A missing absolute path, or a missing relative path outside the mapping, immediately terminates with an error.

To determine the category of a dataset, inspect the `load()` implementation of its Dataset class. Confirm the effective `path`, which loading path above it follows, what happens when local data is absent, and whether it needs supplementary documents, archives, or official evaluation resources in addition to the main data file.
