# 提示词攻击

OpenCompass 支持[PromptBench](https://github.com/microsoft/promptbench)的提示词攻击。其主要想法是评估提示指令的鲁棒性，也就是说，当攻击或修改提示以指导任务时，希望该任务能尽可能表现的像像原始任务一样好。

## 环境安装

提示词攻击需要依赖 `PromptBench` 中的组件，所以需要先配置好环境。

```shell
git clone https://github.com/microsoft/promptbench.git
pip install textattack==0.3.8
export PYTHONPATH=$PYTHONPATH:promptbench/
```

## 如何攻击

### 增加数据集配置文件

我们将使用GLUE-wnli数据集作为示例，大部分配置设置可以参考[config.md](../user_guides/config.md)获取帮助。

首先，我们需要支持基本的数据集配置，你可以在`configs`中找到现有的配置文件，或者根据[new-dataset](./new_dataset.md)支持你自己的配置。

以下面的`infer_cfg`为例，我们需要定义提示模板。`adv_prompt`是实验中要被攻击的基本提示占位符。`sentence1`和`sentence2`是此数据集的输入。攻击只会修改`adv_prompt`字段。

然后，我们应该使用`AttackInferencer`与`original_prompt_list`和`adv_key`告诉推理器在哪里攻击和攻击什么文本。

更多详细信息可以参考`configs/datasets/promptbench/promptbench_wnli_gen_50662f.py`配置文件。

```python
original_prompt_list = [
    'Are the following two sentences entailment or not_entailment? Answer me with "A. entailment" or "B. not_entailment", just one word. ',
    "Does the relationship between the given sentences represent entailment or not_entailment? Respond with 'A. entailment' or 'B. not_entailment'.",
    ...,
]

wnli_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt="""{adv_prompt}
Sentence 1: {sentence1}
Sentence 2: {sentence2}
Answer:"""),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=AttackInferencer,
        original_prompt_list=original_prompt_list,
        adv_key='adv_prompt'))
```

### Add a eval config

我们应该在此处使用 `OpenICLAttackTask` 来进行攻击任务。还应该使用 `NaivePartitioner`，因为攻击实验将重复运行整个数据集近百次以搜索最佳攻击，为方便起见我们不希望拆分数据集。

```note
由于上述提到的重复搜索，请选择小型数据集（样本少于1000）进行攻击，否则时间成本将非常大。
```

在 `attack` 配置中还有其他几个选项：

- `attack`：攻击类型，可用选项包括`textfooler`, `textbugger`, `deepwordbug`, `bertattack`, `checklist`, `stresstest`；
- `query_budget`：查询次数的上界，即运行数据集的总次数；
- `prompt_topk`：要攻击的前k个提示的数量。在大多数情况下，原始提示列表大于10，运行整个集合是耗时的。

```python
# Please run whole dataset at a time, aka use `NaivePartitioner` only
# Please use `OpenICLAttackTask` if want to perform attack experiment
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=8,
        task=dict(type=OpenICLAttackTask),
        retry=0),
)

attack = dict(
    attack='textfooler',
    query_budget=100,
    prompt_topk=2,
)
```

### 运行试验

请当运行攻击实验的时候请使用 `--mode infer` 选项，并需要指定`PYTHONPATH`。

```shell
python run.py configs/eval_attack.py --mode infer
```

所有结果都将保存在名为“attack”的文件夹中。
内容包括原始提示的准确性和受到攻击的提示的准确性，以及前k个提示下降的准确性，例如：

```
Prompt: Assess the connection between the following sentences and classify it as 'A. entailment' or 'B. not_entailment'., acc: 59.15%
Prompt: Does the relationship between the given sentences represent entailment or not_entailment? Respond with 'A. entailment' or 'B. not_entailment'., acc: 57.75%
Prompt: Analyze the two provided sentences and decide if their relationship is 'A. entailment' or 'B. not_entailment'., acc: 56.34%
Prompt: Identify whether the given pair of sentences demonstrates entailment or not_entailment. Answer with 'A. entailment' or 'B. not_entailment'., acc: 54.93%
...
Original prompt: Assess the connection between the following sentences and classify it as 'A. entailment' or 'B. not_entailment'.
Attacked prompt: b"Assess the attach between the following sentences and sorted it as 'A. entailment' or 'B. not_entailment'."
Original acc: 59.15%, attacked acc: 40.85%, dropped acc: 18.31%
```
