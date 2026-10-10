# Discovering and Comparing Configurations

## Listing and Searching Configurations

```bash
python tools/list_configs.py
python tools/list_configs.py qwen mmlu
```

The script searches model, dataset, and summarizer configurations and prints the abbreviations accepted by `opencompass --models/--datasets/--summarizer`. You should still inspect each matched file, especially its prompt, model version, and evaluator.

## Comparing Configuration Directories

```bash
python tools/compare_configs.py folder_a folder_b \
    --extensions .py .json .md \
    --ignore folder_a/generated
```

This tool compares two directories by relative path. A file found only in the first directory, or a same-named file with different content, causes failure; a file found only in the second directory produces a notice. It is useful for checking synchronization between configuration sets, but it does not understand semantic differences after Python configuration inheritance is resolved.

To review the effective configuration of a single experiment, inspect the `configs/` snapshot in its output directory together with `--config-verbose`.
