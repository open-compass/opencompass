# CircularEval

## Background

For multiple-choice questions, when a Language Model (LLM) provides the correct option, it does not necessarily imply a true understanding and reasoning of the question. It could be a guess. To differentiate these scenarios and reduce LLM bias towards options, CircularEval (CircularEval) can be utilized. A multiple-choice question is augmented by shuffling its options, and if the LLM correctly answers all variations of the augmented question, it is considered correct under CircularEval.

## Adding Your Own CircularEval Dataset

Generally, to evaluate a dataset using CircularEval, both its loading and evaluation methods need to be rewritten. Modifications are required in both the OpenCompass main library and configuration files. We will use C-Eval as an example for explanation.

OpenCompass main library:

```python
from opencompass.datasets.ceval import CEvalDataset
from opencompass.datasets.circular import CircularDatasetMeta

class CircularCEvalDataset(CEvalDataset, metaclass=CircularDatasetMeta):
    # The overloaded dataset class
    dataset_class = CEvalDataset

    # Splits of the DatasetDict that need CircularEval. For CEvalDataset, which loads [dev, val, test], we only need 'val' and 'test' for CircularEval, not 'dev'
    default_circular_splits = ['val', 'test']

    # List of keys to be shuffled
    default_option_keys = ['A', 'B', 'C', 'D']

    # If the content of 'answer_key' is one of ['A', 'B', 'C', 'D'], representing the correct answer. This field indicates how to update the correct answer after shuffling options. Choose either this or default_answer_key_switch_method
    default_answer_key = 'answer'

    # If 'answer_key' content is not one of ['A', 'B', 'C', 'D'], a function can be used to specify the correct answer after shuffling options. Choose either this or default_answer_key
    # def default_answer_key_switch_method(item, circular_pattern):
    #     # 'item' is the original data item
    #     # 'circular_pattern' is a tuple indicating the order after shuffling options, e.g., ('D', 'A', 'B', 'C') means the original option A is now D, and so on
    #     item['answer'] = circular_pattern['ABCD'.index(item['answer'])]
    #     return item
```

`CircularCEvalDataset` accepts the `circular_pattern` parameter with two values:

- `circular`: Indicates a single cycle. It is the default value. ABCD is expanded to ABCD, BCDA, CDAB, DABC, a total of 4 variations.
- `all_possible`: Indicates all permutations. ABCD is expanded to ABCD, ABDC, ACBD, ACDB, ADBC, ADCB, BACD, ..., a total of 24 variations.

Additionally, we provide a `CircularEvaluator` to replace `AccEvaluator`. This Evaluator also accepts `circular_pattern`, and it should be consistent with the above. It produces the following metrics:

- `acc_{origin|circular|all_possible}`: Treating each question with shuffled options as separate, calculating accuracy.
- `perf_{origin|circular|all_possible}`: Following Circular logic, a question is considered correct only if all its variations with shuffled options are answered correctly, calculating accuracy.
- `more_{num}_{origin|circular|all_possible}`: According to Circular logic, a question is deemed correct if the number of its variations answered correctly is greater than or equal to num, calculating accuracy.

OpenCompass configuration file:

```python
from mmengine.config import read_base
from opencompass.datasets.circular import CircularCEvalDataset

with read_base():
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets

for d in ceval_datasets:
    # Overloading the load method
    d['type'] = CircularCEvalDataset
    # Renaming for differentiation from non-circular evaluation versions
    d['abbr'] = d['abbr'] + '-circular-4'
    # Overloading the evaluation method
    d['eval_cfg']['evaluator'] = {'type': CircularEvaluator}

# The dataset after the above operations looks like this:
# dict(
#     type=CircularCEvalDataset,
#     path='./data/ceval/formal_ceval',  # Unchanged
#     name='computer_network',  # Unchanged
#     abbr='ceval-computer_network-circular-4',
#     reader_cfg=dict(...),  # Unchanged
#     infer_cfg=dict(...),  # Unchanged
#     eval_cfg=dict(evaluator=dict(type=CircularEvaluator), ...),
# )
```

Additionally, for better presentation of results in CircularEval, consider using the following summarizer:

```python


from mmengine.config import read_base
from opencompass.summarizers import CircularSummarizer

with read_base():
    from ...summarizers.groups.ceval.ceval_summary_groups

new_summary_groups = []
for item in ceval_summary_groups:
    new_summary_groups.append(
        {
            'name': item['name'] + '-circular-4',
            'subsets': [i + '-circular-4' for i in item['subsets']],
        }
    )

summarizer = dict(
    type=CircularSummarizer,
    # Select specific metrics to view
    metric_types=['acc_origin', 'perf_circular'],
    dataset_abbrs = [
        'ceval-circular-4',
        'ceval-humanities-circular-4',
        'ceval-stem-circular-4',
        'ceval-social-science-circular-4',
        'ceval-other-circular-4',
    ],
    summary_groups=new_summary_groups,
)
```

For more complex evaluation examples, refer to this sample code: https://github.com/open-compass/opencompass/tree/main/configs/eval_circular.py
