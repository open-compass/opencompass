# Chain of Thought

## Background

During the process of reasoning, CoT (Chain of Thought) method is an efficient way to help LLMs deal complex questions, for example: math problem and relation inference. In OpenCompass, we support multiple types of CoT method.

![image](https://github.com/open-compass/opencompass/assets/28834990/45d60e0e-02a1-49aa-b792-40a1f95f9b9e)

## 1. Zero Shot CoT

You can change the `PromptTemplate` of the dataset config, by simply add *Let's think step by step* to realize a Zero-Shot CoT prompt for your evaluation:

```python
qa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="Answer the question:\nQ: {question}?\nLet's think step by step:\n"
    ),
    retriever=dict(type=ZeroRetriever)
)
```

## 2. Few Shot CoT

Few-shot CoT can make LLMs easy to follow your instructions and get better answers. For few-shot CoT, add your CoT template to `PromptTemplate` like following config to create a one-shot prompt:

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

The SC (Self-Consistency) method is proposed in [this paper](https://arxiv.org/abs/2203.11171), which will sample multiple reasoning paths for the question, and make majority voting to the generated answers for LLMs. This method displays remarkable proficiency among reasoning tasks with high accuracy but may consume more time and resources when inferencing, because of the majority voting strategy. In OpenCompass, You can easily implement the SC method by replacing `GenInferencer` with `SCInferencer` in the dataset configuration and setting the corresponding parameters like:

```python
# This SC gsm8k config can be found at: opencompass.configs.datasets.gsm8k.gsm8k_gen_a3e34a.py
gsm8k_infer_cfg = dict(
    inferencer=dict(
        type=SCInferencer, # Replace GenInferencer with SCInferencer.
        generation_kwargs=dict(do_sample=True, temperature=0.7, top_k=40),  # Set sample parameters to make sure model generate various output, only works for models load from HuggingFace now.
        infer_type='SC',
        sc_size = SAMPLE_SIZE
    )
)
gsm8k_eval_cfg = dict(sc_size=SAMPLE_SIZE)
```

```{note}
OpenCompass defaults to use argmax for sampling the next token. Therefore, if the sampling parameters are not specified, the model's inference results will be completely consistent each time, and multiple rounds of evaluation will be ineffective.
```

Where `SAMPLE_SIZE` is the number of reasoning paths in Self-Consistency, higher value usually outcome higher performance. The following figure from the original SC paper demonstrates the relation between reasoning paths and performance in several reasoning tasks:

![image](https://github.com/open-compass/opencompass/assets/28834990/05c7d850-7076-43ca-b165-e6251f9b3001)

From the figure, it can be seen that in different reasoning tasks, performance tends to improve as the number of reasoning paths increases. However, for some tasks, increasing the number of reasoning paths may reach a limit, and further increasing the number of paths may not bring significant performance improvement. Therefore, it is necessary to conduct experiments and adjustments on specific tasks to find the optimal number of reasoning paths that best suit the task.

## 4. Tree-of-Thoughts

In contrast to the conventional CoT approach that considers only a single reasoning path, Tree-of-Thoughts (ToT) allows the language model to explore multiple diverse reasoning paths simultaneously. The model evaluates the reasoning process through self-assessment and makes global choices by conducting lookahead or backtracking when necessary. Specifically, this process is divided into the following four stages:

**1. Thought Decomposition**

Based on the nature of the problem, break down the problem into multiple intermediate steps. Each step can be a phrase, equation, or writing plan, depending on the nature of the problem.

**2. Thought Generation**

Assuming that solving the problem requires k steps, there are two methods to generate reasoning content:

- Independent sampling: For each state, the model independently extracts k reasoning contents from the CoT prompts, without relying on other reasoning contents.
- Sequential generation: Sequentially use "prompts" to guide the generation of reasoning content, where each reasoning content may depend on the previous one.

**3. Heuristic Evaluation**

Use heuristic methods to evaluate the contribution of each generated reasoning content to problem-solving. This self-evaluation is based on the model's self-feedback and involves designing prompts to have the model score multiple generated results.

**4. Search Algorithm Selection**

Based on the methods of generating and evaluating reasoning content, select an appropriate search algorithm. For example, you can use breadth-first search (BFS) or depth-first search (DFS) algorithms to systematically explore the thought tree, conducting lookahead and backtracking.

In OpenCompass, ToT parameters need to be set according to the requirements. Below is an example configuration for the 24-Point game from the [official paper](https://arxiv.org/pdf/2305.10601.pdf). Currently, ToT inference is supported only with Huggingface models:

```python
# This ToT Game24 config can be found at: opencompass/configs/datasets/game24/game24_gen_8dfde3.py.
from opencompass.datasets import (Game24Dataset, game24_postprocess,
                                  Game24Evaluator, Game24PromptWrapper)

generation_kwargs = dict(temperature=0.7)

game24_infer_cfg = dict(
        prompt_template=dict(
        type=PromptTemplate,
        template='{input}'), # Directly pass the input content, as the Prompt needs to be specified in steps
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ToTInferencer, # Replace GenInferencer with ToTInferencer
                    generation_kwargs=generation_kwargs,
                    method_generate='propose',  # Method for generating reasoning content, can be independent sampling (sample) or sequential generation (propose)
                    method_evaluate='value', # Method for evaluating reasoning content, can be voting (vote) or scoring (value)
                    method_select='greedy', # Method for selecting reasoning content, can be greedy (greedy) or random (sample)
                    n_evaluate_sample=3,
                    n_select_sample=5,
                    task_wrapper=dict(type=Game24PromptWrapper) # This Wrapper class includes the prompts for each step and methods for generating and evaluating reasoning content, needs customization according to the task
                    ))
```

If you want to use the ToT method on a custom dataset, you'll need to make additional configurations in the `opencompass.datasets.YourDataConfig.py` file to set up the `YourDataPromptWrapper` class. This is required for handling the thought generation and heuristic evaluation step within the ToT framework. For reasoning tasks similar to the game 24-Point, you can refer to the implementation in `opencompass/datasets/game24.py` for guidance.
