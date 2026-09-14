# Model-Side Conversation Template Protocol

This chapter explains how to configure the Model-Side Conversation Template Protocol (Meta Template) for language models and API models. Before reading, see the [Traditional Dialogue-Based Prompt](raw_prompt_template.md#dialogue-based-prompt) in the merged template page. The configuration field remains `meta_template`.

The Model-Side Conversation Template Protocol is tied to the model configuration and combines with the Dataset-side dialogue template at runtime to produce the prompt best suited to the current model:

```python
# Pass the meta_template field into the model
models = [
    dict(
        type='AnyModel',
        meta_template = ...,  # meta template
    )
]
```

```{note}
In some cases, such as testing a base model, no instruction needs to be injected into the normal dialogue and the model-side protocol can be empty. The prompt received by the model is then defined only by the dataset configuration and is a regular string. If the dataset configuration uses a dialogue template, turns from different roles are concatenated with \n.
```

## Application to Language Models

The following figure shows several ways data becomes a prompt through the Dataset-side prompt template and Model-Side Conversation Template Protocol in 2-shot learning. Use it as a reference for the following sections.

![](https://user-images.githubusercontent.com/22607038/251195073-85808807-6359-44df-8a19-9f5d00c591ec.png)

The following examples explain how to define the model-side protocol.

Suppose that according to the dialogue template of the dataset, the following dialogue was produced:

```python
PromptList([
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='2'),
    dict(role='HUMAN', prompt='2+2=?'),
    dict(role='BOT', prompt='4'),
])
```

We want to pass this dialogue to a model that has already gone through SFT. The model's agreed dialogue begins with the speech of different roles with `<Role Name>:` and ends with a special token and \\n. Here is the complete string the model expects to receive:

```Plain
<HUMAN>: 1+1=?<eoh>
<BOT>: 2<eob>
<HUMAN>: 2+2=?<eoh>
<BOT>: 4<eob>
```

In the meta template, we only need to abstract the format of each round of dialogue into the following configuration:

```python
# model meta template
meta_template = dict(
    round=[
          dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n'),
          dict(role='BOT', begin='<BOT>: ', end='<eob>\n'),
    ],
 )
```

______________________________________________________________________

Some datasets may introduce SYSTEM-level roles:

```python
PromptList([
    dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following math questions'),
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='2'),
    dict(role='HUMAN', prompt='2+2=?'),
    dict(role='BOT', prompt='4'),
])
```

Assuming the model also accepts the SYSTEM role, and expects the input to be:

```
<SYSTEM>: Solve the following math questions<eosys>\n
<HUMAN>: 1+1=?<eoh>\n
<BOT>: 2<eob>\n
<HUMAN>: 2+2=?<eoh>\n
<BOT>: 4<eob>\n
end of conversation
```

We can put the definition of the SYSTEM role into `reserved_roles`. Roles in `reserved_roles` will not appear in regular conversations, but they allow the dialogue template of the dataset configuration to call them in `begin` or `end`.

```python
# model meta template
meta_template = dict(
    round=[
          dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n'),
          dict(role='BOT', begin='<BOT>: ', end='<eob>\n'),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<SYSTEM>: ', end='<eosys>\n'),],
 ),
```

If the model does not accept the SYSTEM role, it is not necessary to configure this item, and it can still run normally. In this case, the string received by the model becomes:

```
<HUMAN>: Solve the following math questions<eoh>\n
<HUMAN>: 1+1=?<eoh>\n
<BOT>: 2<eob>\n
<HUMAN>: 2+2=?<eoh>\n
<BOT>: 4<eob>\n
end of conversation
```

This is because in the predefined datasets in OpenCompass, each `SYSTEM` speech has a `fallback_role='HUMAN'`, that is, if the `SYSTEM` role in the meta template does not exist, the speaker will be switched to the `HUMAN` role.

______________________________________________________________________

Some models may need to consider embedding other strings at the beginning or end of the conversation, such as system instructions:

```
Meta instruction: You are now a helpful and harmless AI assistant.
<SYSTEM>: Solve the following math questions<eosys>\n
<HUMAN>: 1+1=?<eoh>\n
<BOT>: 2<eob>\n
<HUMAN>: 2+2=?<eoh>\n
<BOT>: 4<eob>\n
end of conversation
```

In this case, we can specify these strings by specifying the begin and end parameters.

```python
meta_template = dict(
    round=[
          dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n'),
          dict(role='BOT', begin='<BOT>: ', end='<eob>\n'),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<SYSTEM>: ', end='<eosys>\n'),],
    begin="Meta instruction: You are now a helpful and harmless AI assistant.",
    end="end of conversation",
 ),
```

______________________________________________________________________

In **generative** task evaluation, we will not directly input the answer to the model, but by truncating the prompt, while retaining the previous text, we leave the answer output by the model blank.

```
Meta instruction: You are now a helpful and harmless AI assistant.
<SYSTEM>: Solve the following math questions<eosys>\n
<HUMAN>: 1+1=?<eoh>\n
<BOT>: 2<eob>\n
<HUMAN>: 2+2=?<eoh>\n
<BOT>:
```

We only need to set the `generate` field in BOT's configuration to True, and OpenCompass will automatically leave the last utterance of BOT blank:

```python
meta_template = dict(
    round=[
          dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n'),
          dict(role='BOT', begin='<BOT>: ', end='<eob>\n', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<SYSTEM>: ', end='<eosys>\n'),],
    begin="Meta instruction: You are now a helpful and harmless AI assistant.",
    end="end of conversation",
 ),
```

Note that `generate` only affects generative inference. When performing discriminative inference, the prompt received by the model is still complete.

### Full Definition

```bash
models = [
    dict(meta_template = dict(
            begin="Meta instruction: You are now a helpful and harmless AI assistant.",
            round=[
                    dict(role='HUMAN', begin='HUMAN: ', end='<eoh>\n'),  # begin and end can be a list of strings or integers.
                    dict(role='THOUGHTS', begin='THOUGHTS: ', end='<eot>\n', prompt='None'), # Here we can set the default prompt, which may be overridden by the specific dataset
                    dict(role='BOT', begin='BOT: ', generate=True, end='<eob>\n'),
            ],
            end="end of conversion",
            reserved_roles=[dict(role='SYSTEM', begin='SYSTEM: ', end='\n'),],
            eos_token_id=10000,
         ),
     )
]
```

The `meta_template` is a dictionary that can contain the following fields:

- `begin`, `end`: (str, optional) The beginning and ending of the prompt, typically some system-level instructions.

- `round`: (list) The template format of each round of dialogue. The content of the prompt for each round of dialogue is controlled by the dialogue template configured in the dataset.

- `reserved_roles`: (list, optional) Specify roles that do not appear in `round` but may be used in the dataset configuration, such as the `SYSTEM` role.

- `eos_token_id`: (int, optional): Specifies the ID of the model's eos token. If not set, it defaults to the eos token id in the tokenizer. Its main role is to trim the output of the model in generative tasks, so it should generally be set to the first token id of the end corresponding to the item with generate=True.

The `round` of the `meta_template` specifies the format of each role's speech in a round of dialogue. It accepts a list of dictionaries, each dictionary's keys are as follows:

- `role` (str): The name of the role participating in the dialogue. This string does not affect the actual prompt.

- `begin`, `end` (str): Specifies the fixed beginning or end when this role speaks.

- `prompt` (str): The role's prompt. It is allowed to leave it blank in the meta template, but in this case, it must be specified in the prompt of the dataset configuration.

- `generate` (bool): When specified as True, this role is the one the model plays. In generation tasks, the prompt received by the model will be cut off at the `begin` of this role, and the remaining content will be filled by the model.

## Application to API Models

The Model-Side Conversation Template Protocol for an API model is similar to that for a regular model, but simpler. Use one of the following configurations to evaluate an API model in a multi-turn dialogue, as appropriate:

```bash
# If the API model does not support system instructions
meta_template=dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
)

# If the API model supports system instructions
meta_template=dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)
```

### Principle

Even though different API models accept different data structures, there are commonalities overall. Interfaces that accept dialogue history generally allow users to pass in prompts from the following three roles:

- User
- Robot
- System (optional)

In this regard, OpenCompass has preset three `api_role` values for API models: `HUMAN`, `BOT`, `SYSTEM`, and stipulates that in addition to regular strings, the input accepted by API models includes a middle format of dialogue represented by `PromptList`. The API model will repackage the dialogue in a multi-turn dialogue format and send it to the backend. However, to activate this feature, users need to map the roles `role` in the dataset prompt template to the corresponding `api_role` in the above meta template. The following figure illustrates the relationship between the input accepted by the API model and the Prompt Template and Meta Template.

![](https://user-images.githubusercontent.com/22607038/251195872-63aa7d30-045a-4837-84b5-11b09f07fb18.png)

## Debugging

To debug a prompt, prepare the configuration and use `tools/prompt_viewer.py` to preview what the model actually receives. See [Prompt Preview and Debugging](debugging.md) for details.
