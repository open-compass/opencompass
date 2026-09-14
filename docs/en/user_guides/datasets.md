# Dataset Selection and Configuration

In OpenCompass, one “dataset configuration” defines data loading, model input, and scoring rules together. A dataset with the same name can have multiple configuration variants, so results cannot be compared by raw dataset name alone.

## Finding and Selecting Configurations

```bash
python tools/list_configs.py mmlu gsm8k
```

Configuration files are usually under `opencompass/configs/datasets/<dataset>/`. Components such as `gen`, `ppl`, `rawprompt`, the few-shot count, and a hash in the filename distinguish evaluation protocols. A file without a hash is normally a convenient import entry point, but you should still open it and verify its target and contents.

Run `list_configs.py` without arguments to see every model, dataset, and summarizer configuration discoverable by the installed version:

```bash
python tools/list_configs.py
```

Abbreviations in the first output column can be passed directly to `opencompass --datasets ...`. Dataset count and configuration variants change continuously, so this documentation no longer maintains a static list that would quickly become stale. Treat `opencompass/configs/datasets/` and the tool output in the current code as authoritative.

Confirm the following before selecting a configuration:

- Data source, version, split, and sample range.
- Input fields, answer fields, and media resources.
- Prompt type, few-shot count, and inference method.
- Evaluator, answer extraction, and postprocessing rules.
- Dependencies on a Judge, code sandbox, or official evaluation service.
- Data directories, cache variables, and offline-operation requirements.

## Dataset Configuration Structure

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

- `type` and `path` determine where and how data is loaded.
- `reader_cfg` declares input columns, answer columns, split, and sample range.
- `infer_cfg` declares the prompt, few-shot Retriever, and Gen/PPL Inferencer.
- `eval_cfg` declares prediction/reference postprocessing and the Evaluator.

For complete customization, see [Adding a Dataset](../extension/new_dataset.md). To quickly evaluate your own JSON, JSONL, or CSV data, see [Quickly Evaluating Your Own Data](../extension/custom_dataset.md).

## Repeated Runs

The CLI option `--dataset-num-runs N` copies dataset configurations and evaluates them multiple times. The configuration fields `n`/`k` are also consumed by some robustness metrics. Repeated runs are meaningful only when generation arguments permit randomness, the service can vary, or the metric explicitly requires multiple samples. Preserve both each run and the aggregation method; do not report only the average.

## Data Locations

Data can come from local files, OpenCompass data packages, Hugging Face, ModelScope, or dataset-specific download logic. See [Data Sources, Caches, and Offline Operation](data_and_cache.md) for caching and offline rules.
