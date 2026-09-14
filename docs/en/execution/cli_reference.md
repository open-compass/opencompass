# Command-Line Reference

The final authority for the installed version is:

```bash
opencompass --help
```

## Experiment Entry Point

| Argument       | Meaning                                                |
| -------------- | ------------------------------------------------------ |
| `config`       | Python experiment configuration file; optional         |
| `--models`     | Select models by name from configuration directories   |
| `--datasets`   | Select datasets by name from configuration directories |
| `--summarizer` | Select a summarizer configuration by name              |
| `--config-dir` | Additional configuration search directory              |

Without a configuration file, provide `--models` together with `--datasets`, or construct a Hugging Face model with `--hf-path` and select a dataset.

## Execution and Output

| Argument                      | Meaning                                                             |
| ----------------------------- | ------------------------------------------------------------------- |
| `--dry-run`                   | Parse configuration and partition tasks without running inference   |
| `--debug`                     | Run in one process and display logs in the terminal                 |
| `--mode {all,infer,eval,viz}` | Select execution stage                                              |
| `--reuse [TIMESTAMP]`         | Reuse a selected or latest timestamp directory                      |
| `--work-dir`                  | Output root                                                         |
| `--config-verbose`            | Print the final configuration                                       |
| `--dump-eval-details False`   | Disable per-sample evaluation details, which are enabled by default |
| `--dump-res-length`           | Record response lengths                                             |
| `--analysis-repeat`           | Analyze repeated predictions during summarization                   |
| `--dump-extract-rate`         | Report answer extraction rate                                       |

## Concurrency and Backends

| Argument                        | Meaning                                               |
| ------------------------------- | ----------------------------------------------------- |
| `--max-num-workers`             | Maximum concurrent tasks for the default Runner       |
| `--max-workers-per-gpu`         | Maximum LocalRunner tasks per GPU                     |
| `--slurm -p PARTITION`          | Use Slurm                                             |
| `--dlc --aliyun-cfg PATH`       | Use DLC                                               |
| `--accelerator {vllm,lmdeploy}` | Attempt to convert a supported HF model configuration |
| `--retry`                       | Failure retry count for the default Slurm/DLC Runner  |

## Quickly Constructing a Hugging Face Model

Common arguments include `--hf-type`, `--hf-path`, `--tokenizer-path`, `--model-kwargs`, `--tokenizer-kwargs`, `--generation-kwargs`, `--max-seq-len`, `--max-out-len`, `--batch-size`, and `--hf-num-gpus`. Put complex or reproducibility-critical model settings in a configuration file.

`--dataset-num-runs N` copies every dataset configuration and runs it N times. When generation is stochastic, record each result and the aggregation method.
