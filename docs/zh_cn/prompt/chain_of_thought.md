# Chain of Thought

## 背景

CoT（思维链）是帮助大型语言模型解决如数学问题和关系推理问题等复杂问题的有效方式，在OpenCompass中，我们支持多种类型的CoT方法。

![image](https://github.com/open-compass/opencompass/assets/28834990/45d60e0e-02a1-49aa-b792-40a1f95f9b9e)

## 1. 零样本思维链

可以通过在数据集配置中简单地添加 “Let's think step by step"，来更改数据集配置的 PromptTemplate，从而实现 零样本 CoT prompt 以进行评估：

```python
qa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="Answer the question:\nQ: {question}?\nLet's think step by step:\n"
    ),
    retriever=dict(type=ZeroRetriever)
)
```

## 2. 小样本思维链

小样本思维链可以使大型语言模型更容易跟随预设的指示并得到更好的答案。对于小样本思维链，按照以下配置将思维链模板添加到 `PromptTemplate` 中，可以创建一个 one-shot prompt：

```python
qa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=
'''Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?
Let's think step by step
Answer:
Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.
His team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers
They scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.
All together his team scored 50+24+10= 84 points
Mark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.
His opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.
They also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.
All together Mark's opponents scored 100+12+5=117 points
The total score for the game is both team's scores added together, so it is 84+117=201 points
The answer is 201

Question: {question}\nLet's think step by step:\n{answer}
'''),
    retriever=dict(type=ZeroRetriever)
)
```

## 3. Self-Consistency

SC (Self-Consistency) 方法是在 [此文章](https://arxiv.org/abs/2203.11171) 中提出的，该方法会为问题生成多条不同的推理路径，并对生成的答案进行众数投票。这种方法在复杂推理任务中表现出了显著的能力，但由于需要推理多次来采样多条推理链，所以可能会消耗很多的时间和资源。在 OpenCompass 中，您可以通过在数据集配置中将 `GenInferencer` 替换为 `SCInferencer` 并设置相应的参数参数来简单地实现 SC 方法，例如：

```python
# 此SC版gsm8k测试配置可以在： opencompass.configs.datasets.gsm8k.gsm8k_gen_a3e34a.py 中找到。
gsm8k_infer_cfg = dict(
    inferencer=dict(
        type=SCInferencer, # 替换 GenInferencer 为 SCInferencer
        generation_kwargs=dict(do_sample=True, temperature=0.7, top_k=40),  # 设置采样参数以确保模型生成不同的输出，目前仅适用于从HuggingFace加载的模型。
        infer_type='SC',
        sc_size = SAMPLE_SIZE
    )
)
gsm8k_eval_cfg = dict(sc_size=SAMPLE_SIZE)
```

```{note}
注意，OpenCompass 默认使用 argmax 的方式采样下一个 token，因此若不指定采样参数，模型每次的推理结果将会是完全一致的，多轮评测将会失效。
```

其中 `SAMPLE_SIZE` 是推理路径的数量，较高的值通常会带来更高的性能。SC方法的原论文中展示了不同推理任务间推理路径数量与性能之间的关系：

![image](https://github.com/open-compass/opencompass/assets/28834990/05c7d850-7076-43ca-b165-e6251f9b3001)

从图中可以看出，在不同的推理任务中，随着推理路径数量的增加，性能呈现出增长的趋势。但是，对于某些任务，增加推理路径的数量可能达到一个极限，进一步增加推理路径的数量可能不会带来更多的性能提升。因此，需要在具体任务中进行实验和调整，找到最适合任务的推理路径数量。

## 4. Tree-of-Thoughts

相比一般的CoT方法采样一条推理路径，ToT(Tree-of-Thoughts)允许语言模型同时考虑多种不同的推理路径，通过对推理过程进行自我评估，以及在必要时进行前瞻或回溯以做出全局选择。具体的，分为下面四个阶段：

**1. 问题分解 (Thought Decomposition)**

根据问题的特点，将问题分解成多个中间步骤。每个步骤可以是短语、算式或写作计划，这取决于问题的性质。

**2. 推理过程生成 (Thought Generation)**

假设解决问题需要k个步骤，有两种方法生成推理内容：

- 独立采样：对于每个状态，模型会独立地从CoT提示中完整抽取k个推理内容，不依赖于其他的推理内容。
- 顺序生成：顺序地使用“提示”来逐步引导推理内容生成，每个推理内容都可能依赖于前一个推理内容。

**3. 启发式评估 (Heuristic Evaluation)**

使用启发式方法评估每个生成的推理内容对问题解决的贡献，这种自我评估基于语言模型的自我反馈，如设计Prompt让模型对多个生成结果进行打分。

**4. 选择搜索算法 (Search Algorithm)**

根据生成和评估推理内容的方法，选择适当的搜索算法。例如，可以使用广度优先搜索（BFS）或深度优先搜索（DFS）等算法来系统地探索思考树，并进行前瞻和回溯。

在OpenCompass中，需要根据需要设置ToT参数，以下是[ToT论文](https://arxiv.org/pdf/2305.10601.pdf)中24点游戏的样例配置，目前支持Huggingface模型进行ToT推理：

```python
# 此 ToT Game24 配置可以在以下路径找到：opencompass/configs/datasets/game24/game24_gen_8dfde3.py。
from opencompass.datasets import (Game24Dataset, game24_postprocess,
                                  Game24Evaluator, Game24PromptWrapper)

generation_kwargs = dict(temperature=0.7)

game24_infer_cfg = dict(
        prompt_template=dict(
        type=PromptTemplate,
        template='{input}'), #直接传入input内容，因为Prompt需要分段指定
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ToTInferencer, # 替换GenInferencer为ToTInferencer
                    generation_kwargs=generation_kwargs,
                    method_generate='propose',  # 生成推理内容的方法，可以是独立采样（sample）或顺序生成（propose）
                    method_evaluate='value', # 评估推理内容的方法，可以是投票 （vote）或打分（value）
                    method_select='greedy', # 选择推理内容的方法，可以是贪心（greedy）或随机（sample）
                    n_evaluate_sample=3,
                    n_select_sample=5,
                    task_wrapper=dict(type=Game24PromptWrapper) # 该Wrapper类包含每个步骤的Prompt和推理内容的生成及评估方法，需要根据任务进行自定义
                    ))

```

如果要在自定义的数据集上使用ToT方法，相比普通评测方式，需要在`opencompass.datasets.YourDataConfig.py`中额外设置`YourDataPromptWrapper`类，以进行ToT中的推理生成和启发式评估。对于类似游戏24点的推理任务，具体可以参考`opencompass/datasets/game24.py`。
