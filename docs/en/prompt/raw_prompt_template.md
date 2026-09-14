# Prompt Templates: RawPromptTemplate by Default

Prompt configuration determines how one data sample becomes the input actually received by a model. Even with the same model, raw data, and Evaluator, a different prompt, few-shot example set, or model protocol can substantially change the result.

Evaluation normally places instructions and in-context examples before the question, then chooses generative completion, PPL, or another inference method. OpenCompass therefore defines input construction under `infer_cfg` in the dataset configuration and separates it into two layers:

```text
data sample
  ↓ dataset-side template (RawPromptTemplate by default / traditional PromptTemplate)
normalized text or role/content messages
  ↓ Model-Side Conversation Template Protocol (MetaTemplate) / tokenizer chat template
input actually received by the model
```

Prefer `RawPromptTemplate` for a new conversational evaluation configuration. It directly represents `role/content` messages and is easy to compare with an API or tokenizer chat template. `PromptTemplate` remains useful for maintaining old configurations, PPL multi-candidate templates, and complex ICE orchestration. Model-specific role markers and special tokens belong in the [Model-Side Conversation Template Protocol](meta_template.md).

## RawPromptTemplate (Default)

## Basic Usage

```python
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role='system', content='You are a helpful assistant.'),
            dict(
                role='user',
                content=(
                    '{problem}\n'
                    'Put the final answer in \\boxed{{}}.'
                ),
            ),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

`{problem}` comes from a dataset sample field. Rendering produces a message list, which is then passed to the model backend. Confirm that every placeholder appears in `reader_cfg.input_columns` or in a retrieved example.

## Three Element Types in `messages`

The `messages` list accepts three kinds of elements:

| Element | Purpose |
| --- | --- |
| `dict(role=..., content=...)` | A normal message; `{field}` in `content` is replaced by the sample field |
| `dict(expand_column='xxx')` | Reads a message list from sample field `xxx` and expands it in place |
| `'</E>'` (string) | Insertion point for few-shot examples (ICE), described below |

### `expand_column`: Expanding a Dataset Message Column

When an input is an entire conversation stored in a dataset field rather than one or two messages, insert it as a whole with `expand_column`. For example, a sample's `dialogue` field may be:

```python
# One row in the dataset
{
    'dialogue': [
        {'role': 'user', 'content': 'Write a product description in no more than 100 words.'},
        {'role': 'assistant', 'content': ''},
        {'role': 'user', 'content': 'Rewrite the preceding paragraph in a more formal tone.'},
        {'role': 'assistant', 'content': ''},
    ]
}
```

Configure it as:

```python
reader_cfg = dict(input_columns=['dialogue'], output_column='reference')

infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[{'expand_column': 'dialogue'}],
        format_variables=False,
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

An `expand_column` element has no `role` or `content`. During rendering, it reads the field and inserts each message in order. `format_variables=False` indicates that these messages are ready-made data and should not undergo placeholder substitution, which is suitable for already rendered conversations and ChatML data.

### Inserting Few-Shot Examples (ICE)

The string `'</E>'` in the main template (the default `ice_token`) marks where few-shot examples are inserted. A separate `ice_template` describes the format of each example. The following configuration comes from `opencompass/configs/datasets/SciReasoner/mol_biotext_rawprompt_gen.py`:

```python
infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'system',
             'content': 'There is a single choice question about chemistry. '
                        'Answer the question directly.'},
            '</E>',
            {'role': 'user', 'content': 'Query: {input}'},
        ],
    ),
    ice_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': 'Query: {input}'},
            {'role': 'assistant', 'content': '{output}'},
        ],
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0]),
    inferencer=dict(type=GenInferencer),
)
```

Rendering works as follows:

- The retriever selects k few-shot examples (`FixKRetriever` uses fixed indices; alternatives include `TopkRetriever` and `RandomRetriever`).
- `ice_template` renders each example into messages (here, one user question and one assistant answer).
- All example messages replace `'</E>'` in order. The final input remains one message list: system + k example groups + current question.

## Multi-Turn Input and Inference

A multi-turn evaluation requires the model to answer continuously in one conversation while retaining previous user messages and model replies. A later turn depends on generation from the preceding turn, so tasks cannot be arbitrarily split into independent samples.

The data is normally organized with user turns from the dataset and empty assistant `content` fields as generation slots. The inferencer advances turn by turn, writes each generation into its assistant slot, and accumulates context. The complete `infer_cfg` for the multi-turn instruction-following dataset MultiIF is:

```python
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

multiif_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[{'expand_column': 'dialogue'}],
        format_variables=False,
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        multiround=True,
    ),
)
```

`multiround=True` on `GenInferencer` enables turn-by-turn inference. It generates whenever it encounters an empty assistant turn and retains the result in context for later turns. The evaluation configuration must identify which role's output to read; MultiIF uses `eval_cfg=dict(evaluator=..., pred_role='BOT')`. If the model backend automatically inserts a system message or generation prompt, also ensure that it is not duplicated by the dataset template.

## Differences from PromptTemplate

Traditional `PromptTemplate` uses structures such as `begin`, `round`, and `ice_token` to express roles and few-shot concatenation before mapping them into model input. `RawPromptTemplate` makes configuration content correspond directly to the final messages.

Prefer `RawPromptTemplate` when:

- Adding a dataset for a chat model or OpenAI-compatible API.
- Input is already system/user/assistant messages, or a whole conversation is stored in a dataset field.
- You want less role mapping and easier message-by-message inspection.

Continue using [PromptTemplate](#prompttemplate-traditional-template) when:

- Maintaining an existing stable configuration.
- Few-shot orchestration needs more than one `</E>` insertion point.
- Using a traditional inference method, such as PPL, that needs multiple candidate templates.

## Boundary Between Dataset and Model Templates

The Dataset-side RawPromptTemplate expresses how to ask the question. The model-side [Model-Side Conversation Template Protocol](meta_template.md) or tokenizer chat template expresses how that model encodes roles. Do not write model-specific special tokens on both sides or accidentally append two system prompts.

Some API model configurations can add messages through `meta_template`, for example:

```python
models = [
    dict(
        type=OpenAISDK,
        abbr='my-api-model',
        path='my-model',
        key='ENV',
        openai_api_base='https://example.com/v1',
        meta_template=[
            dict(role='system', content='Additional system instruction.'),
        ],
        max_seq_len=32768,
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=0),
    )
]
```

Before use, inspect how the concrete model class reads `key` and `meta_template`; fields differ between API backends.

## PromptTemplate (Traditional Template)

`PromptTemplate` is the traditional OpenCompass template system. New configurations use RawPromptTemplate by default. The following syntax remains available for maintaining an existing configuration, constructing PPL multi-candidate input, or complex ICE orchestration.

### String-Based Prompt

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

### Dialogue-Based Prompt

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

### Prompt Templates and `inferencer`

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

### `ice_template` and `prompt_template`

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

#### Abbreviated Usage

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

### Usage Suggestion

Use [Prompt Viewer](../tools/index.md) to visualize the assembled prompts, confirm that the template is correct, and verify that the result meets expectations.

## Migrating from the Traditional Template

Check the following in order during migration:

1. Convert system instructions in `begin` into system messages.
2. Convert each HUMAN/BOT turn into user/assistant messages.
3. Preserve field placeholders and verify brace escaping.
4. Preview several final inputs.
5. Compare predictions before and after migration on a small sample, confirming that few-shot order, stop words, and answer extraction are unchanged.

A more direct template format does not make evaluation semantics automatically equivalent. Migration involving few-shot examples, repeated sampling, or a special model protocol requires sample-by-sample comparison.

## Debugging

After changing a template, preview the final input and inspect role order, few-shot insertion position, special tokens, and truncation. See [Prompt Debugging and Prompt Viewer](debugging.md).
