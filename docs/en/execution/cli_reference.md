# Command-Line Reference

The basic form of the `opencompass` command is:

```bash
opencompass [config] [options]
```

This page reflects the current implementation in `opencompass/cli/main.py`. If the code changes, treat the output of the following command as authoritative:

```bash
opencompass --help
```

## Configuration Entry Points and Lookup

| Argument                           | Default   | Description                                                                                                                                                   |
| ---------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config`                           | None      | Optional positional argument specifying the path to a Python configuration file.                                                                              |
| `-h`, `--help`                     | —         | Display help information.                                                                                                                                     |
| `--models MODEL [MODEL ...]`       | None      | Find and load one or more model configurations by name.                                                                                                       |
| `--datasets DATASET [DATASET ...]` | None      | Find and load one or more dataset configurations by name.                                                                                                     |
| `--summarizer SUMMARIZER`          | `example` | Select a result-summary configuration in shorthand configuration mode. Use `filename/config_key` to select a specific configuration object.                   |
| `--config-dir DIR`                 | `configs` | Specify a custom configuration root. OpenCompass searches its `models/`, `datasets/`, `dataset_collections/`, and `summarizers/` subdirectories while retaining the built-in search paths. |

When `config` is supplied, OpenCompass reads that file first. Shorthand construction arguments such as `--models`, `--datasets`, `--summarizer`, `--hf-*`, and `--custom-dataset-*` do not replace its model or dataset configurations. Without a configuration file, use one of these entry points:

- load existing configurations with `--models` and `--datasets`;
- construct a Hugging Face model with `--hf-path` and select datasets with `--datasets`;
- select a model with `--models` or `--hf-path` and construct a custom dataset with `--custom-dataset-path`.

## Execution Stages and Working Directory

| Argument                              | Default           | Description                                                                                                                                        |
| ------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-m`, `--mode {all,infer,eval,viz}`   | `all`             | Select the execution stage: `all` runs the complete workflow, `infer` runs inference only, `eval` runs scoring only, and `viz` runs summarization only. |
| `-r`, `--reuse [TIMESTAMP]`           | None              | Reuse an experiment directory for the specified timestamp. If the timestamp is omitted, use the last directory by name in the working directory.   |
| `-w`, `--work-dir DIR`                | `outputs/default` | Set the experiment output root. Artifacts are stored in its timestamped subdirectory.                                                              |
| `--debug`                             | `False`           | Enable debug mode. The Runner executes tasks sequentially and displays logs directly, which is useful for diagnosing a first run.                  |
| `--dry-run`                           | `False`           | Parse the configuration and partition tasks without starting inference or evaluation. This also enables debug-level logging.                       |
| `-a`, `--accelerator {vllm,lmdeploy}` | None              | Attempt to convert supported local Hugging Face model configurations to vLLM or LMDeploy for one-stop deployment and evaluation. Unsupported model types remain unchanged and produce a warning. |
| `--config-verbose`                    | `False`           | Print the loaded and processed experiment configuration.                                                                                           |
| `-l`, `--lark`                        | `False`           | Enable Lark bot task notifications. The configuration must also provide `lark_bot_url`.                                                            |

## Task Partitioning and Runner

| Argument                  | Default | Description                                                                                                                                                              |
| ------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--max-num-workers N`     | `1`     | Set the maximum concurrent task count for a default Runner supplied by the entry point and set `num_worker` for the default inference Partitioner. An explicitly configured stage is normally not overridden. |
| `--max-workers-per-gpu N` | `1`     | Set the maximum number of concurrent tasks per GPU for an automatically generated `LocalRunner`.                                                                         |
| `--slurm -p PARTITION`    | `False` | Force `SlurmRunner` and replace existing `infer` and `eval` execution configurations. `-p/--partition` is required. `-q/--quotatype`, `--qos`, and `--retry` (default: 2) set the quota type, Quality of Service, and retry count. This option is mutually exclusive with `--dlc`. |
| `--dlc --aliyun-cfg PATH` | `False` | Force the Alibaba Cloud PAI-DLC Runner and replace existing `infer` and `eval` execution configurations. `--aliyun-cfg` selects the DLC configuration file and defaults to `~/.aliyun.cfg`; the path must exist. `--retry` (default: 2) sets the retry count. This option is mutually exclusive with `--slurm`. |

## Inference, Evaluation, and Analysis Outputs

| Argument                       | Default | Description                                                                                                                                                         |
| ------------------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dump-eval-details [BOOL]`   | `True`  | Save per-sample evaluation details. Use `--dump-eval-details False` to disable them; omitting the value keeps it `True`.                                            |
| `--dump-res-length`            | `False` | Pass the response-length statistics flag to inference tasks. Support depends on the Inferencer.                                                                     |
| `--dump-only-message-path DIR` | None    | Export constructed messages without requesting the model. Currently supported only by `GenInferencer`.                                                              |
| `--dump-extract-rate`          | `False` | Instruct evaluation tasks to calculate and save the answer extraction rate.                                                                                         |
| `--analysis-repeat`            | `False` | Analyze repeated predictions during summarization and write a repeated-output analysis file.                                                                        |
| `--dataset-num-runs N`         | `1`     | In shorthand CLI configuration mode, set `n` and `k` to `N` in every loaded dataset configuration. This argument is not applied when an explicit configuration file is supplied. |

## Quickly Constructing a Hugging Face Model

The following arguments apply only when neither `config` nor `--models` supplies a model and `--hf-path` is used to construct one:

| Argument                                        | Default            | Description                                                                                                  |
| ----------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------ |
| `--hf-type {base,chat}`                         | `chat`             | Select the base-model or chat-model wrapper.                                                                 |
| `--hf-path PATH`                                | None               | Specify a Hugging Face model path or repository ID.                                                          |
| `--model-kwargs KEY=VALUE [KEY=VALUE ...]`      | `{}`               | Pass model-loading arguments.                                                                                |
| `--tokenizer-path PATH`                         | Same as model path | Specify a tokenizer path or repository ID.                                                                   |
| `--tokenizer-kwargs KEY=VALUE [KEY=VALUE ...]`  | `{}`               | Pass tokenizer-loading arguments.                                                                            |
| `--peft-path PATH`                              | None               | Specify a PEFT weights path.                                                                                 |
| `--peft-kwargs KEY=VALUE [KEY=VALUE ...]`       | `{}`               | Pass PEFT-loading arguments.                                                                                 |
| `--generation-kwargs KEY=VALUE [KEY=VALUE ...]` | `{}`               | Pass generation arguments.                                                                                   |
| `--max-seq-len N`                               | None               | Set the maximum sequence length supported by the model.                                                      |
| `--max-out-len N`                               | `256`              | Set the maximum output length.                                                                               |
| `--min-out-len N`                               | `1`                | Set the minimum output length.                                                                               |
| `--batch-size N`                                | `8`                | Set the inference batch size.                                                                                |
| `--hf-num-gpus N`                               | `1`                | Set the number of GPUs used by a Hugging Face model task.                                                    |
| `--pad-token-id N`                              | None               | Specify the padding token ID.                                                                                |
| `--stop-words WORD [WORD ...]`                  | Empty list         | Specify one or more stop words.                                                                              |
| `--num-gpus N`                                  | —                  | Deprecated. The current version raises an error when this argument is supplied; use `--hf-num-gpus` instead. |

## Quickly Constructing a Custom Dataset

The following arguments generate a dataset configuration from a local file:

| Argument                                  | Default       | Description                                                                                 |
| ----------------------------------------- | ------------- | ------------------------------------------------------------------------------------------- |
| `--custom-dataset-path PATH`              | None          | Specify a custom dataset file. Required when neither `config` nor `--datasets` is supplied. |
| `--custom-dataset-meta-path PATH`         | None          | Specify the metadata file for the custom dataset.                                           |
| `--custom-dataset-data-type {mcq,qa}`     | Auto-detected | Select multiple-choice or question-answer data.                                             |
| `--custom-dataset-infer-method {gen,ppl}` | Auto-detected | Select generative or PPL inference.                                                         |

## Result Persistence Arguments

| Argument                    | Default | Description                                                                                                                                                            |
| --------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-sp`, `--station-path DIR` | None    | Specify a shared result directory. When this argument or `station_path` in the configuration is set, results are saved to the shared directory after the run.          |
| `--read-from-station`       | `False` | Read existing results from the shared directory before execution, write them into the current experiment's `results/`, and skip model-dataset combinations whose results already exist. |
| `--station-overwrite`       | `False` | Allow existing files to be overwritten when saving results to the shared directory.                                                                                    |
