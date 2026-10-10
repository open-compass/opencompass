# Prompt Templates

Prompt configuration determines how each data sample is converted into the input that the model actually receives. Even when the model, source data, and Evaluator are identical, changes to the prompt, few-shot examples, or model protocol can substantially affect results.

Evaluation commonly places instructions and in-context examples before the question and then uses generative completion, PPL, or another inference method. OpenCompass therefore defines input construction in the dataset configuration's `infer_cfg` and divides it into two layers:

```text
Data sample
  ↓ Dataset-side template (RawPromptTemplate, standard / PromptTemplate, legacy)
Unified text or role/content messages
  ↓ Model-side conversation protocol (MetaTemplate) / tokenizer chat template
Input actually received by the model
```

The standard `RawPromptTemplate` directly represents `role`/`content` messages and is recommended for new chat or API evaluation configurations. The legacy `PromptTemplate` mainly supports existing one-stop deployment and evaluation configurations and base-model evaluations that use PPL or similar inference methods. To append content from the model configuration, see [Model-side Conversation Template Protocol](meta_template.md).

## Standard Prompt Template

### Basic Usage

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

`{problem}` refers to a field in the dataset sample. Rendering the template produces a message list, which is then passed to the model backend. Verify that every placeholder is present in `reader_cfg.input_columns` or in the retrieved examples.

### Three Types of `messages` Elements

The `messages` list accepts three element types:

| Element                       | Purpose                                                                                 |
| ----------------------------- | --------------------------------------------------------------------------------------- |
| `dict(role=..., content=...)` | A regular message. `{field}` placeholders in `content` are replaced with sample fields. |
| `dict(expand_column='xxx')`   | Reads a message list from the sample's `xxx` field and expands it in place.             |
| `'</E>'` (string)             | The insertion point for in-context examples (ICE), described below.                     |

#### `expand_column`: Expand a Message Column from the Dataset

When the input is a complete conversation stored in one dataset field rather than one or two messages, use `expand_column` to insert it in full. For example, a sample may contain this `dialogue` field:

```python
# One row in the dataset
{
    'dialogue': [
        {'role': 'user', 'content': 'Write a product description in no more than 100 words.'},
        {'role': 'assistant', 'content': ''},
        {'role': 'user', 'content': 'Rewrite the preceding text in a more formal tone.'},
        {'role': 'assistant', 'content': ''},
    ]
}
```

Configure it as follows:

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

An `expand_column` element has no `role` or `content`. During rendering, the template reads the named field and inserts its messages in order. `format_variables=False` indicates that these messages are already prepared and should not undergo placeholder substitution, which is appropriate for pre-rendered conversations and ChatML data.

#### Insert Few-shot Examples (ICE)

The `'</E>'` string in the main template—the default `ice_token`—marks where few-shot examples are inserted. A separate `ice_template` defines the format of each example. The following configuration is adapted from `opencompass/configs/datasets/SciReasoner/mol_biotext_rawprompt_gen.py`:

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

Rendering proceeds as follows:

- the Retriever selects `k` examples (`FixKRetriever` selects fixed indices; alternatives include `TopkRetriever` and `RandomRetriever`);
- `ice_template` renders each example as a message group—one user question and one assistant answer here;
- all example messages replace `'</E>'` in the main template in order, leaving one final message list: system message + `k` examples + current question.

### Multi-turn Input and Inference

Multi-turn evaluation requires the model to answer successive turns in the same conversation while retaining preceding user messages and model responses. Because the input for each later turn depends on the previous generation, the task cannot be divided into independent samples arbitrarily.

The dataset commonly stores user turns as data and leaves assistant `content` empty as a generation slot. The Inferencer advances the conversation one turn at a time, writes each generation into its assistant turn, and retains it in the context. The complete `infer_cfg` for the MultiIF multi-turn instruction-following dataset is:

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

`GenInferencer(multiround=True)` enables turn-by-turn inference. Each empty assistant turn triggers one generation, whose content remains in the context for subsequent turns.

## Legacy Prompt Template

The legacy `PromptTemplate` mainly supports existing one-stop deployment and evaluation configurations and base-model evaluations that use PPL inference. Prefer `RawPromptTemplate` for new chat or API evaluation configurations.

`template` may be a string, or a legacy chat structure organized with `begin`, `round`, and `end`. Each message is described by `role` and `prompt`; `fallback_role` specifies an alternative role when the model-side template does not define the requested `role`. A role-based template can be converted to model input with the [Model-side Conversation Template Protocol](meta_template.md). Without a model-side template, its `prompt` values are concatenated in order.

A regular generation task can be configured as follows:

```python
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt='You are a helpful assistant.',
                ),
            ],
            round=[
                dict(role='HUMAN', prompt='Question: {question}'),
                dict(role='BOT', prompt='Answer: {answer}'),
            ],
            end=[],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

`begin` contains system instructions, `round` describes the user input and model-response position, and `end` contains any content appended after the conversation. A generation task normally begins generating at the final `BOT` role, so `end` is usually empty or omitted. If `{answer}` corresponds to `reader_cfg.output_column`, it is cleared during inference to prevent reference-answer leakage.

Unlike generative inference, `PPLInferencer` requires `template` to be a dictionary providing one complete input for each candidate answer and selects the candidate with the lowest PPL:

```python
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A': (
                'Question: {question}\n'
                'A. {A}\nB. {B}\nC. {C}\n'
                'Answer: A'
            ),
            'B': (
                'Question: {question}\n'
                'A. {A}\nB. {B}\nC. {C}\n'
                'Answer: B'
            ),
            'C': (
                'Question: {question}\n'
                'A. {A}\nB. {B}\nC. {C}\n'
                'Answer: C'
            ),
        },
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)
```

For few-shot input, use an `ice_template` with the same syntax as `prompt_template` to describe retrieved examples, and use `ice_token` to specify their insertion point:

```python
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever

infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Question: {question}'),
                dict(role='BOT', prompt='Answer: {answer}'),
            ],
        ),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt='Answer the following questions.',
                ),
                '</E>',
            ],
            round=[
                dict(role='HUMAN', prompt='Question: {question}'),
                dict(role='BOT', prompt='Answer: {answer}'),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1]),
    inferencer=dict(type=GenInferencer),
)
```

`FixKRetriever` selects examples 0 and 1 from the index set. `ice_template` renders them as two question-answer groups, which replace `</E>` in `prompt_template`. The current evaluation sample is still rendered by `prompt_template.round`, and its reference answer is cleared during inference.
