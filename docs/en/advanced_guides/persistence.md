# Evaluation Results Persistence

## Introduction

Normally, the evaluation results of OpenCompass will be saved to your work directory. But in some cases, there may be a need for data sharing among users or quickly browsing existing public evaluation results. Therefore, we provide an interface that can quickly transfer evaluation results to external public data stations, and on this basis, provide functions such as uploading, overwriting, and reading.

## Quick Start

### Uploading

By adding `args` to the evaluation command or adding configuration in the Eval script, the results of evaluation can be stored in the path you specify. Here are the examples:

(Approach 1) Add an `args` option to the command and specify your public path address.

```bash
opencompass  ...  -sp '/your_path'
```

(Approach 2) Add configuration in the Eval script.

```pythonE
station_path = '/your_path'
```

### Overwriting

The above storage method will first determine whether the same task result already exists in the data station based on the `abbr` attribute in the model and dataset configuration before uploading data. If results already exists, cancel this storage. If you need to update these results, please add the `station-overwrite` option to the command, here is an example:

```bash
opencompass  ...  -sp '/your_path' --station-overwrite
```

### Reading

You can directly read existing results from the data station to avoid duplicate evaluation tasks. The read results will directly participate in the 'summarize' step. When using this configuration, only tasks that do not store results in the data station will be initiated. Here is an example:

```bash
opencompass  ...  -sp '/your_path' --read-from-station
```

### Command Combination

1. Only upload the results under your latest working directory to the data station, without supplementing tasks that missing results:

```bash
opencompass  ...  -sp '/your_path' -r latest -m viz
```

## Storage Format of the Data Station

In the data station, the evaluation results are stored as `json` files for each `model-dataset` pair. The specific directory form is `/your_path/dataset_name/model_name.json `. Each `json` file stores a dictionary corresponding to the results, including `predictions`, `results`, and `cfg`, here is an example:

```pythonE
Result = {
    'predictions': List[Dict],
    'results': Dict,
    'cfg': Dict = {
        'models': Dict,
        'datasets': Dict,
        (Only subjective datasets)'judge_models': Dict
    }
}
```

Among this three keys, `predictions` records the predictions of the model on each item of data in the dataset. `results` records the total score of the model on the dataset. `cfg` records detailed configurations of the model and the dataset in this evaluation task.
