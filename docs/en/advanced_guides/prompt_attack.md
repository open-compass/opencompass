# Prompt Attack

We support prompt attack following the idea of [PromptBench](https://github.com/microsoft/promptbench). The main purpose here is to evaluate the robustness of prompt instruction, which means when attack/modify the prompt to instruct the task, how well can this task perform as the original task.

## Set up environment

Some components are necessary to prompt attack experiment, therefore we need to set up environments.

```shell
git clone https://github.com/microsoft/promptbench.git
pip install textattack==0.3.8
export PYTHONPATH=$PYTHONPATH:promptbench/
```

## How to attack

### Add a dataset config

We will use GLUE-wnli dataset as example, most configuration settings can refer to [config.md](../user_guides/config.md) for help.

First we need support the basic dataset config, you can find the existing config files in `configs` or support your own config according to [new-dataset](./new_dataset.md)

Take the following `infer_cfg` as example, we need to define the prompt template. `adv_prompt` is the basic prompt placeholder to be attacked in the experiment. `sentence1` and `sentence2` are the input columns of this dataset. The attack will only modify the `adv_prompt` here.

Then, we should use `AttackInferencer` with `original_prompt_list` and `adv_key` to tell the inferencer where to attack and what text to be attacked.

More details can refer to `configs/datasets/promptbench/promptbench_wnli_gen_50662f.py` config file.

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

We should use `OpenICLAttackTask` here for attack task. Also `NaivePartitioner` should be used because the attack experiment will run the whole dataset repeatedly for nearly hurdurds times to search the best attack, we do not want to split the dataset for convenience.

```note
Please choose a small dataset(example < 1000) for attack, due to the aforementioned repeated search, otherwise the time cost is enumerous.
```

There are several other options in `attack` config:

- `attack`: attack type, available options includes `textfooler`, `textbugger`, `deepwordbug`, `bertattack`, `checklist`, `stresstest`;
- `query_budget`: upper boundary of queries, which means the total numbers of running the dataset;
- `prompt_topk`: number of topk prompt to be attacked. In most case, the original prompt list is great than 10, running the whole set is time consuming.

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

### Run the experiment

Please use `--mode infer` when run the attack experiment, and set `PYTHONPATH` env.

```shell
python run.py configs/eval_attack.py --mode infer
```

All the results will be saved in `attack` folder.
The content includes the original prompt accuracy and the attacked prompt with dropped accuracy of `topk` prompt, for instance:

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
