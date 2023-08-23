# Results Summary

After the evaluation is complete, the results need to be printed on the screen or saved. This process is controlled by the summarizer.

```{note}
If the summarizer appears in the overall config, all the evaluation results will be output according to the following logic.
If the summarizer does not appear in the overall config, the evaluation results will be output in the order they appear in the `dataset` config.
```

## Example

A typical summarizer configuration file is as follows:

```python
summarizer = dict(
    dataset_abbrs = [
        'race',
        'race-high',
        'race-middle',
    ],
    summary_groups=[
        {'name': 'race', 'subsets': ['race-high', 'race-middle']},
    ]
)
```

The output is:

```text
dataset      version    metric         mode      internlm-7b-hf
-----------  ---------  -------------  ------  ----------------
race         -          naive_average  ppl                76.23
race-high    0c332f     accuracy       ppl                74.53
race-middle  0c332f     accuracy       ppl                77.92
```

The summarizer tries to read the evaluation scores from the `{work_dir}/results/` directory using the `models` and `datasets` in the config as the full set. It then displays them in the order of the `summarizer.dataset_abbrs` list. Moreover, the summarizer tries to compute some aggregated metrics using `summarizer.summary_groups`. The `name` metric is only generated if and only if all values in `subsets` exist. This means if some scores are missing, the aggregated metric will also be missing. If scores can't be fetched by the above methods, the summarizer will use `-` in the respective cell of the table.

In addition, the output consists of multiple columns:

- The `dataset` column corresponds to the `summarizer.dataset_abbrs` configuration.
- The `version` column is the hash value of the dataset, which considers the dataset's evaluation method, prompt words, output length limit, etc. Users can verify whether two evaluation results are comparable using this column.
- The `metric` column indicates the evaluation method of this metric. For specific details, [metrics](./metrics.md).
- The `mode` column indicates how the inference result is obtained. Possible values are `ppl` / `gen`. For items in `summarizer.summary_groups`, if the methods of obtaining `subsets` are consistent, its value will be the same as subsets, otherwise it will be `mixed`.
- The subsequent columns represent different models.

## Field Description

The fields of summarizer are explained as follows:

- `dataset_abbrs`: (list, optional) Display list items. If omitted, all evaluation results will be output.
- `summary_groups`: (list, optional) Configuration for aggregated metrics.

The fields in `summary_groups` are:

- `name`: (str) Name of the aggregated metric.
- `subsets`: (list) Names of the metrics that are aggregated. Note that it can not only be the original `dataset_abbr` but also the name of another aggregated metric.
- `weights`: (list, optional) Weights of the metrics being aggregated. If omitted, the default is to use unweighted averaging.

Please note that we have stored the summary groups of datasets like MMLU, C-Eval, etc., under the `configs/summarizers/groups` path. It's recommended to consider using them first.
