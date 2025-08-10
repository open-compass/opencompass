# Prompt Template

## Background

In language model evaluation, we often construct prompts from the original dataset according to certain rules to enable the model to answer questions as required.

Typically, we place instructions at the beginning of the prompt, followed by several in-context examples, and finally, we include the question. For example:

```text
Solve the following questions.
1+1=?
2
3+9=?
12
5+6=?
```

Extensive experiments have shown that even with the same original test questions, different ways of constructing the prompt can affect the model's performance. Factors that may influence this include:

- The composition of the prompt itself, including instructions, in-context examples, and the format of the question.
- The selection of in-context examples, including the number and method of selection.
- The manner in which the prompt is used. Should the model complete the prompt based on the given context, or should it choose the best prompt among the candidate prompts?

OpenCompass defines the prompt construction strategy in the `infer_cfg` section of the dataset configuration. A typical `infer_cfg` is shown below:

```python
infer_cfg = dict(
    ice_template=dict(  # Template used to construct In Context Examples (ice).
        type=PromptTemplate,
        template='{question}\n{answer}'
    ),
    prompt_template=dict(  # Template used to construct the main prompt.
        type=PromptTemplate,
        template='Solve the following questions.\n</E>{question}\n{answer}',
        ice_token="</E>"
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1]),  # Definition of how to retrieve in-context examples.
    inferencer=dict(type=GenInferencer),  # Method used to generate predictions.
)
```

In this document, we will mainly introduce the definitions of `ice_template`, `prompt_template`, and `inferencer`. For information on the `retriever`, please refer to other documents.

Let's start by introducing the basic syntax of the prompt.

## String-Based Prompt

String-based prompt is a classic form of template. Consider the following template:

```python
prompt_template=dict(
    type=PromptTemplate,
    template="{anything}\nQuestion: {question}\nAnswer: {answer}"
)
```

At runtime, the fields within the `{}` will be replaced with corresponding fields from the data sample. If a field does not exist in the data sample, it will be kept as is in the output.

For example, let's consider a data example as follows:

```python
example = {
    'question': '1+1=?',
    'answer': '2',  # Assume the answer is in the reader_cfg.output_column
    'irrelevant_infos': 'blabla',
}
```

After filling in the template, the result will be:

```text
{anything}
Question: 1+1=?
Answer:
```

As you can see, the actual answer for the question, represented by the field `answer`, does not appear in the generated result. This is because OpenCompass will mask fields that are written in `reader_cfg.output_column` to prevent answer leakage. For detailed explanations on `reader_cfg`, please refer to the relevant documentation on dataset configuration.

## Dialogue-Based Prompt

In practical testing, making models perform simple completions may not effectively test the performance of chat-based models. Therefore, we prefer prompts that take the form of dialogues. Additionally, different models have varying definitions of dialogue formats. Hence, we need prompts generated from the dataset to be more versatile, and the specific prompts required by each model can be generated during testing.

To achieve this, OpenCompass extends the string-based prompt to dialogue-based prompt. Dialogue-based prompt is more flexible, as it can combine with different [meta_templates](./meta_template.md) on the model side to generate prompts in various dialogue formats. It is applicable to both base and chat models, but their definitions are relatively complex.

Now, let's assume we have a data sample as follows:

```python
example = {
    'question': '1+1=?',
    'answer': '2',  # Assume the answer is in the reader_cfg.output_column
    'irrelavent_infos': 'blabla',
}
```

Next, let's showcase a few examples:

`````{tabs}

````{tab} Single-round Dialogue
```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        round=[
            dict(role="HUMAN", prompt="Question: {question}"),
            dict(role="BOT", prompt="Answer: {answer}"),
        ]
    )
)
```

The intermediate result obtained by OpenCompass after filling the data into the template is:

```python
PromptList([
    dict(role='HUMAN', prompt='Question: 1+1=?'),
    dict(role='BOT', prompt='Answer: '),
])
```

````

````{tab} Multi-round Dialogue
```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        round=[
            dict(role="HUMAN", prompt="Question: 2+2=?"),
            dict(role="BOT", prompt="Answer: 4"),
            dict(role="HUMAN", prompt="Question: 3+3=?"),
            dict(role="BOT", prompt="Answer: 6"),
            dict(role="HUMAN", prompt="Question: {question}"),
            dict(role="BOT", prompt="Answer: {answer}"),
        ]
    )
)
```

The intermediate result obtained by OpenCompass after filling the data into the template is:

```python
PromptList([
    dict(role='HUMAN', prompt='Question: 2+2=?'),
    dict(role='BOT', prompt='Answer: 4'),
    dict(role='HUMAN', prompt='Question: 3+3=?'),
    dict(role='BOT', prompt='Answer: 6'),
    dict(role='HUMAN', prompt='Question: 1+1=?'),
    dict(role='BOT', prompt='Answer: '),
])
```
````


````{tab} Dialogue with sys instruction

```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        begin=[
            dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
        ],
        round=[
            dict(role="HUMAN", prompt="Question: {question}"),
            dict(role="BOT", prompt="Answer: {answer}"),
        ]
    )
)
```

The intermediate result obtained by OpenCompass after filling the data into the template is:

```python
PromptList([
    dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
    dict(role='HUMAN', prompt='Question: 1+1=?'),
    dict(role='BOT', prompt='Answer: '),
])
```

During the processing of a specific meta template, if the definition includes the SYSTEM role, the template designated for the SYSTEM role will be used for processing. On the other hand, if the SYSTEM role is not defined, the template assigned to the fallback_role role will be utilized, which, in this example, corresponds to the HUMAN role.

````

`````

In dialogue-based templates, prompts are organized in the form of conversations between different roles (`role`). In the current predefined dataset configuration of OpenCompass, some commonly used roles in a prompt include:

- `HUMAN`: Represents a human, usually the one asking questions.
- `BOT`: Represents the language model, usually the one providing answers.
- `SYSTEM`: Represents the system, typically used at the beginning of prompts to give instructions.

Furthermore, unlike string-based templates, the prompts generated by dialogue-based templates are transformed into an intermediate structure called PromptList. This structure will be further combined with the model-side [meta_templates](./meta_template.md) to assemble the final prompt. If no meta template is specified, the prompts in the PromptList will be directly concatenated into a single string.

```{note}
The content within the PromptList in the example above is not the final input to the model and depends on the processing of the meta template. One potential source of misunderstanding is that in generative evaluations, the prompt of the last `BOT` role, `Answer: `, **will not** be inputted to the model. This is because API models generally cannot customize the initial part of model-generated responses. Therefore, this setting ensures consistency in the evaluation behavior between language models and API models. For more information, please refer to the documentation on [meta template](./meta_template.md).
```

<details>
<summary>Expand the complete parameter descriptions</summary>

- `begin`, `end`: (list, optional) The beginning and end of the prompt, typically containing system-level instructions. Each item inside can be **a dictionary or a string**.

- `round`: (list) The format of the dialogue in the template. Each item in the list must be a dictionary.

Each dictionary has the following parameters:

- `role` (str): The role name participating in the dialogue. It is used to associate with the names in meta_template but does not affect the actual generated prompt.

- `fallback_role` (str): The default role name to use in case the associated role is not found in the meta_template. Defaults to None.

- `prompt` (str): The dialogue content for the role.

</details>

## Prompt Templates and `inferencer`

Once we understand the basic definition of prompt templates, we also need to organize them according to the type of `inferencer`.

OpenCompass mainly supports two types of inferencers: `GenInferencer` and `PPLInferencer`, corresponding to two different inference methods.

`GenInferencer` corresponds to generative inference. During inference, the model is asked to continue generating text based on the input prompt. In this case, the `template` represents a single template for each sentence, for example:

`````{tabs}

````{group-tab} String-based Prompt
```python
prompt_template=dict(
    type=PromptTemplate,
    template='Solve the following questions.\n{question}\n{answer}'
)
```
````

````{group-tab} Dialogue-Based Prompt
```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        begin=[
            dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
        ],
        round=[
            dict(role="HUMAN", prompt="{question}"),
            dict(role="BOT", prompt="{answer}"),
        ]
    )
)
```
````

`````

Then, the model's inference result will be a continuation of the concatenated string.

For `PPLInferencer`, it corresponds to discriminative inference. During inference, the model is asked to compute the perplexity (PPL) for each input string and select the item with the lowest perplexity as the model's inference result. In this case, `template` is a `dict` representing the template for each sentence, for example:

`````{tabs}

````{group-tab} String-based Prompt
```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        "A": "Question: Which is true?\nA. {A}\nB. {B}\nC. {C}\nAnswer: A",
        "B": "Question: Which is true?\nA. {A}\nB. {B}\nC. {C}\nAnswer: B",
        "C": "Question: Which is true?\nA. {A}\nB. {B}\nC. {C}\nAnswer: C",
        "UNK": "Question: Which is true?\nA. {A}\nB. {B}\nC. {C}\nAnswer: None of them is true.",
    )
)
```
````

````{group-tab} Dialogue-Based Prompt
```python
prompt_template=dict(
    type=PromptTemplate,
    template=dict(
        "A": dict(
            round=[
                dict(role="HUMAN", prompt="Question: Which is true?\nA. {A}\nB. {B}\nC. {C}"),
                dict(role="BOT", prompt="Answer: A"),
            ]
        ),
        "B": dict(
            round=[
                dict(role="HUMAN", prompt="Question: Which is true?\nA. {A}\nB. {B}\nC. {C}"),
                dict(role="BOT", prompt="Answer: B"),
            ]
        ),
        "C": dict(
            round=[
                dict(role="HUMAN", prompt="Question: Which is true?\nA. {A}\nB. {B}\nC. {C}"),
                dict(role="BOT", prompt="Answer: C"),
            ]
        ),
        "UNK": dict(
            round=[
                dict(role="HUMAN", prompt="Question: Which is true?\nA. {A}\nB. {B}\nC. {C}"),
                dict(role="BOT", prompt="Answer: None of them is true."),
            ]
        ),
    )
)
```
````

`````

In this case, the model's inference result will be one of the four keys in the `template` ("A" / "B" / "C" / "UNK").

## `ice_template` and `prompt_template`

In OpenCompass, for 0-shot evaluation, we usually only need to define the `prompt_template` field to complete prompt construction. However, for few-shot evaluation, we also need to define the `ice_template` field, which manages the prompt templates corresponding to the in-context examples during context learning.

Both `ice_template` and `prompt_template` follow the same syntax and rules. The complete prompt construction process can be represented using the following pseudo-code:

```python
def build_prompt():
    ice = ice_template.format(*ice_example)
    prompt = prompt_template.replace(prompt_template.ice_token, ice).format(*prompt_example)
    return prompt
```

Now, let's assume there are two training data (ex1, ex2) and one testing data (ex3):

```python
ex1 = {
    'question': '2+2=?',
    'answer': '4',
    'irrelavent_infos': 'blabla',
}
ex2 = {
    'question': '3+3=?',
    'answer': '6',
    'irrelavent_infos': 'blabla',
}
ex3 = {
    'question': '1+1=?',
    'answer': '2',  # Assume the answer is in the reader_cfg.output_column
    'irrelavent_infos': 'blabla',
}
```

Next, let's take a look at the actual effects of different prompt construction methods:

`````{tabs}

````{group-tab} String-based Prompt

Template configurations are as follows:

```python
infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template='{question}\n{answer}'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template='Solve the following questions.\n</E>{question}\n{answer}'
        ice_token='</E>',
    )
)
```

The resulting strings are as follows:

```text
Solve the following questions.
2+2=?
4
3+3=?
6
1+1=?

```

````

````{group-tab} Dialogue-Based Prompt

Template configurations are as follows:

```python
infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="{question}"),
                dict(role="BOT", prompt="{answer}"),
            ]
        )
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
                '</E>',
            ],
            round=[
                dict(role="HUMAN", prompt="{question}"),
                dict(role="BOT", prompt="{answer}"),
            ],
        ),
        ice_token='</E>',
    )
)
```

The intermediate results obtained by OpenCompass after filling the data into the templates are as follows:

```python
PromptList([
    dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following questions.'),
    dict(role='HUMAN', prompt='2+2=?'),
    dict(role='BOT', prompt='4'),
    dict(role='HUMAN', prompt='3+3=?'),
    dict(role='BOT', prompt='6'),
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt=''),
])
```
````

`````

### Abbreviated Usage

It is worth noting that, for the sake of simplicity in the configuration file, the `prompt_template` field can be omitted. When the `prompt_template` field is omitted, the `ice_template` will be used as the `prompt_template` as well, to assemble the complete prompt. The following two `infer_cfg` configurations are equivalent:

<table class="docutils">
  <thead>
  <tr>
      <th>Complete Form</th>
      <th>Abbreviated Form</th>
  <tbody>
  <tr>
  <td>

```python
infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template="Q: {question}\nA: {answer}",
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template="</E>Q: {question}\nA: {answer}",
        ice_token="</E>",
    ),
    # ...
)
```

</td>
  <td>

```python
infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template="</E>Q: {question}\nA: {answer}",
        ice_token="</E>",
    ),
    # ...
)
```

</td>
  </tr>
  </thead>
  </table>

More generally, even in the case of 0-shot learning (i.e., when `retriever` is `ZeroRetriver`), this mechanism still applies. Therefore, the following configuration is also valid:

```python
datasets = [
    dict(
        infer_cfg=dict(
            ice_template=dict(
                type=PromptTemplate,
                template="Q: {question}\nA: {answer}",
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )
    ),
]
```

## Usage Suggestion

It is suggested to use the [Prompt Viewer](../tools.md) tool to visualize the completed prompts, confirm the correctness of the templates, and ensure that the results meet expectations.
