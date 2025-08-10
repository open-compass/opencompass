# 循环评测

## 背景

对于选择题而言，当 LLM 给出正确的选项，并不一定代表着它能真正地理解题意并经过推理得出答案，它也有可能是蒙对的。为了将这两种情形区分开，同时也为了降低 LLM 对选项的偏见，我们可以尝试使用循环评测 (CircularEval)。我们会将一道选择题按照打乱选项的方式进行增广，若 LLM 可以在增广后的每道题上均得到正确的答案，那么我们认为在循环评测的意义下，这道题被做对了。

## 新增自己的循环评测数据集

一般来说，为了将一个数据集使用循环评测的方式进行评测，它的加载方式和评测方式是需要被重写的，OpenCompass 主库和配置文件均需要进行修改。后续我们以 C-Eval 为例进行讲解。

OpenCompass 主库：

```python
from opencompass.datasets.ceval import CEvalDataset
from opencompass.datasets.circular import CircularDatasetMeta

class CircularCEvalDataset(CEvalDataset, metaclass=CircularDatasetMeta):
    # 被重载的数据集类
    dataset_class = CEvalDataset

    # 若原 load 方法得到一 DatasetDict，其哪些 split 需要被循环评测。CEvalDataset load 得到 [dev, val, test]，我们只需要对 val 和 test 进行循环评测，dev 不需要
    default_circular_splits = ['val', 'test']

    # 需要被打乱的 key 列表
    default_option_keys = ['A', 'B', 'C', 'D']

    # 若 answer_key 的内容属于是 ['A', 'B', 'C', 'D'] 之一，并表示正确答案。该字段表示打乱选项后，需要如何更新正确答案。与 default_answer_key_switch_method 二选一
    default_answer_key = 'answer'

    # 如果 answer_key 的内容不属于 ['A', 'B', 'C', 'D'] 之一，那么可以使用函数的方式来指定打乱选项后的正确答案。与 default_answer_key 二选一
    # def default_answer_key_switch_method(item, circular_pattern):
    #     # item 是原本的数据项
    #     # circular_pattern 是一个 tuple，表示打乱选项后的顺序，例如 ('D', 'A', 'B', 'C') 表示原来的 A 选项变成了 D，原来的 B 选项变成了 A，以此类推
    #     item['answer'] = circular_pattern['ABCD'.index(item['answer'])]
    #     return item
```

`CircularCEvalDataset` 会接受 `circular_pattern` 参数，它有两个取值:

- `circular`: 表示单项循环。默认为该值。ABCD 会被扩充为 ABCD, BCDA, CDAB, DABC, 共 4 种
- `all_possible`: 表示全排列。ABCD 会被扩充为 ABCD, ABDC, ACBD, ACDB, ADBC, ADCB, BACD, ..., 共 24 种

另外我们提供了一个 `CircularEvaluator` 用于替换 `AccEvaluator`，该 Evaluator 同样接受 `circular_pattern`，该参数应与上述保持一致。它会产出以下指标：

- `acc_{origin|circular|all_possible}`: 将打乱后选项顺序后的题目视作多道单独的题目，计算准确率
- `perf_{origin|circular|all_possible}`: 按照 circular 的逻辑，若选项打乱后的题目都回答正确，才会视为这道题正确，计算准确率
- `more_{num}_{origin|circular|all_possible}`: 按照 circular 的逻辑，若选项打乱后的题目回答正确的数量大于等于 num，就会视为这道题正确，计算准确率

OpenCompass 配置文件：

```python
from mmengine.config import read_base
from opencompass.datasets.circular import CircularCEvalDataset

with read_base():
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets

for d in ceval_datasets:
    # 重载 load 方法
    d['type'] = CircularCEvalDataset
    # 为了与非循环评测版本做区分而进行改名
    d['abbr'] = d['abbr'] + '-circular-4'
    # 重载评测方法
    d['eval_cfg']['evaluator'] = {'type': CircularEvaluator}

# 上述操作后的 dataset 形如下：
# dict(
#     type=CircularCEvalDataset,
#     path='./data/ceval/formal_ceval',  # 未改变
#     name='computer_network',  # 未改变
#     abbr='ceval-computer_network-circular-4',
#     reader_cfg=dict(...),  # 未改变
#     infer_cfg=dict(...),  # 未改变
#     eval_cfg=dict(evaluator=dict(type=CircularEvaluator), ...),
# )
```

另外评测时为了针对循环评测有更良好的结果呈现，建议考虑使用以下 summarizer

```python
from mmengine.config import read_base
from opencompass.summarizers import CircularSummarizer

with read_base():
    from ...summarizers.groups.ceval import ceval_summary_groups

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
    # 选择具体看哪些指标
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

更多复杂的评测案例可以参考这个样例代码: https://github.com/open-compass/opencompass/tree/main/configs/eval_circular.py
