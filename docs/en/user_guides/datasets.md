# Dataset Selection and Configuration

An OpenCompass dataset configuration defines data loading, model input construction, and scoring rules together. The same dataset may have multiple configuration variants, and different evaluation protocols may produce different results.

## Selecting a Configuration

```bash
python tools/list_configs.py mmlu gsm8k  # Find configurations related to MMLU and GSM8K
```

Dataset configuration files are usually located under `opencompass/configs/datasets/<dataset>/`. Their filenames commonly include identifiers such as `gen`, `ppl`, `rawprompt`, the number of few-shot examples, and a hash to distinguish evaluation protocols.

Before selecting a configuration, verify the following:

- data source, version, split, and sample range;
- input fields, answer field, and any multimodal input;
- prompt type, number of few-shot examples, and inference method;
- Evaluator and post-processing rules, including any dependency on a Judge model or an external evaluation service.

## Dataset Configuration Structure

A dataset configuration consists of data-loading arguments and three sections: `reader_cfg`, `infer_cfg`, and `eval_cfg`.

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

- `type`: the dataset class registered with OpenCompass. It loads the source data into a Hugging Face `Dataset` or `DatasetDict`.
- `abbr`: the short name used in task directories and summary results. Use distinguishable abbreviations for different evaluation configurations of the same source data.
- `path`: a dataset path or repository identifier. Depending on the dataset class, arguments such as `name`, `split`, or `task` may also be accepted to select a subset.
- `reader_cfg`: specifies the fields, data splits, and sample ranges used in evaluation.
- `infer_cfg`: specifies prompt construction, few-shot example retrieval, and the inference method.
- `eval_cfg`: specifies prediction post-processing and scoring rules.

### `reader_cfg`: Fields and Data Splits

The basic form of `reader_cfg` is as follows:

```python
reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='train',
    test_split='test',
    train_range=None,
    test_range='[:100]',
)
```

The fields have the following meanings:

- `input_columns`: fields used to construct the model input, such as the question, options, or context.
- `output_column`: the field containing the reference answer. Set it to `None` for tasks that do not require reference answers.
- `train_split` and `test_split`: respectively specify the split from which the Retriever selects few-shot examples and the split on which inference and scoring are performed. Their defaults are `train` and `test`. These fields may be omitted when the data has only one split.
- `train_range` and `test_range`: limit the samples selected from the corresponding splits. `None` uses all samples; an integer selects a fixed number after shuffling; a float between `0` and `1` selects that proportion; and a slice string such as `'[:100]'` or `'[100:200]'` selects an interval in the original order. These fields may be omitted for full evaluation.

After completing the configuration, verify that `input_columns`, `output_column`, and every placeholder in the prompt exist in the loaded data. To quickly test the first several samples, use `test_range='[:N]'`.

### `infer_cfg`: Prompt, Retrieval, and Inference

The most common structure for generative evaluation is:

```python
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role='user', content='{question}\nPlease provide the answer.'),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

`infer_cfg` commonly contains the following fields:

- `prompt_template`: converts each dataset sample into model input through a prompt template.
- `retriever`: determines which few-shot examples are selected from the training split. `ZeroRetriever` retrieves no examples; `FixKRetriever`, `RandomRetriever`, and other Retrievers select examples according to their respective strategies.
- `inferencer`: determines the inference method. `GenInferencer` asks the model to generate an answer directly and accepts inference arguments such as `max_out_len` and `stopping_criteria`. A `max_out_len` explicitly set here takes precedence over the default in the model configuration.

For prompt placeholders, conversation messages, and few-shot insertion, see [Prompt Templates](../prompt/raw_prompt_template.md). After changing a template, use [Prompt Preview and Debugging](../prompt/debugging.md) to inspect the final input.

A common PPL evaluation configuration is:

```python
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'yes': '{question} yes',
            'no': '{question} no',
        },
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)
```

The keys of the candidate templates must cover the label values in `output_column`. Alternatively, candidate labels may be specified explicitly through the `labels` argument of `PPLInferencer`. PPL evaluation requires a model backend capable of computing log-likelihoods for its input; not every API model provides this capability.

### `eval_cfg`: Post-processing and Scoring

When model outputs cannot be compared directly with reference answers, each side can be post-processed before scoring. For example:

```python
from opencompass.datasets import (Gsm8kEvaluator,
                                  gsm8k_dataset_postprocess,
                                  gsm8k_postprocess)

eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)
```

The fields serve the following purposes:

- `evaluator`: the scorer configuration. `type` selects the Evaluator, and all remaining fields are passed as initialization arguments. Common scoring methods include accuracy, exact match, mathematical answer verification, code execution, and model-based judging.
- `pred_postprocessor`: processes model predictions before scoring, for example by extracting an option letter, a number, or an answer enclosed in a particular tag.
- `dataset_postprocessor`: processes reference answers from `output_column` before scoring so that their format matches the processed predictions.
- `pred_role`: extracts content for a specified role from the output of a local chat model. Use it only when the model defines a corresponding `meta_template`.

An Evaluator `type` is required by the basic scoring workflow; all other fields are optional.

For data caching and offline behavior, see [Data Sources, Caching, and Offline Operation](data_and_cache.md). For the complete dataset extension procedure, see [Adding a Dataset](../extension/new_dataset.md).

## Dataset Statistics

The following table is generated from `dataset-index.yml` in the repository root. It lists the datasets registered with OpenCompass, their categories, resource links, and recommended configurations, and supports fuzzy search.

```{include} ../dataset_statistics.inc
```
