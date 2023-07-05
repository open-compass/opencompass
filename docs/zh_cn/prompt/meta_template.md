# Meta Prompt

## 背景

在 LLM 的实际 finetune 中，我们常常会根据实际的要求注入一些预定义的字符串，以求模型能按照自然语言的格式输出指定的内容。在评测时，我们也需要按照 finetune 时设定的格式输入问题，模型才能发挥出其最大的性能。因此，我们需要对 OpenICL 原本的 prompt 设计作一次增强，才能满足相应需求。

## Model - Meta Template

此前， prompt template 的设定绑定在数据集配置中。现在考虑到不同模型的 instruction 可能会有所不同，我们往 model config 中新增 `meta_template` 字段，允许用户指定与模型密切相关的 instruction。

```Python
models = [
    dict(type='LLM',
         # ...
         meta_template = dict(
            begin="meta instruction\nYou are an AI assistant.\n",
            round=[
                    dict(role='HUMAN', begin='<|HUMAN|>:', end='脷\n'),  # begin and end can be a list of strings or integers.
                    dict(role='THOUGHTS', begin='<|Inner Thoughts|>:', end='茔\n', prompt='None'),
                    dict(role='COMMANDS', begin='<|Commands|>:', end='蝮\n', prompt='None'),
                    dict(role='RESULTS', begin='<|Results|>:', end='兒\n', prompt='None'),  # Here we can set the default prompt, which may be overridden by the speicfic dataset
                    dict(role='BOT', begin='<|MOSS|>:', generate=True, end='氡\n'),
            ],
            end="end of conversion",
            reserved_roles=[dict(role='SYSTEM', begin='<|SYSTEM|>: ', end='\n'),],
            # the token to stop the generation tasks (TODO: support string)
            eos_token_id=65605,
         ),
     )
]
```

这里，meta_template 是一个**字典**，该字典可以包含以下数个字段：

- `begin`，`end` ：(str，可选) prompt 的开头，通常是一些 meta instruction。

- `round`：(list，可选) 约定了每一轮对话的 prompt 格式。每轮对话的 prompt 内容由 dataset config 中的 prompt template 控制（下文会详述）。如果不指定，则该字段将会直接被 dataset config 中的 prompt template 替换。

- (str，可选)：收尾的 instruction。

- `reserved_roles` （list，可选）指定了在 meta template 中并未出现的预留角色。这里面定义的角色有可能在 dataset config 的 begin 或 end 中用到，例如 `SYSTEM` 角色。

- `eos_token_id` （int, 可选）：指定了该模型在生成式任务中 eos token 的 id。如果不设置，则默认为 tokenizer 中的 eos token id。

`round` 指定了每轮对话中每个角色说话的格式，通常接受一个列表，内容可以是 **str 或 dict**。每个字典接受以下关键字：

- `role`（str）: 对话中的角色，也可以认为是这个 prompt 的 identifier。该字符串并不影响实际的 prompt，仅用于在 dataset_config 中的指定对应项，并对其 prompt 内容进行覆盖。

- `begin`, `end` (str): 指定该角色在说话时的开头或结尾。

- `prompt` (str)：prompt 的内容，遵循 `ICLPromptTemplate` 的格式规范。如果在 meta_prompt_template 中未指定，则必须在 dataset config 中的 prompt template 中指定。

- `generate` (bool): 指定为 True 时，该角色即为模型在生成任务中开始生成输出的位点。在生成任务中生成对应 prompt 时，prompt template 只会生成到该角色的 begin，剩下的内容由模型补全。

在上面的例子中，最后的 meta prompt 将会是：

```
meta instructionYou are an AI assistant.
<|HUMAN|>: 脷\n
<|Inner Thoughts|>: None茔\n<|Commands|>: None蝮\n<|Results|>: None兒\n
<|MOSS|>: 氡\n
end of conversion
```

特别地，在生成式任务中，prompt 仅会生成到 \<|MOSS|>: 后：

```
meta instructionYou are an AI assistant.
<|HUMAN|>: 脷\n
<|Inner Thoughts|>: None茔\n<|Commands|>: None蝮\n<|Results|>: None兒\n
<|MOSS|>:
```

接下来我们在 dataset config 中进行进一步约定。

## Dataset: Prompt Template

在 model 配置中约定了该 model 所需的 meta template 后，dataset 中 prompt template 的格式也会有所变化。同时，该方向尽可能地保持了 prompt 的 backward compatibility。

在改动前，`PromptTemplate` 接受 str 或 dict 作为输入。其中，dict 形式的输入将 label string 映射到对应的 prompt (str)上，通常用作为 `PPLInferencer` 的输入。因而本质上，`PromptTemplate` 的旧版实现里表示 prompt 的方式只有 `str` 一种。

而改动后的 prompt template 允许接受的 prompt 基本形式从 str 扩展到了 dict。

这个 dict 的格式与 meta template 相似，用户也可以指定 `begin`, `end` 和 `round` 关键字：

```Python
mmlu_prompt_template = dict(
    type='PromptTemplate',
    template=dict(
        begin=[dict(role='SYSTEM', fallback_role='HUMAN', prompt='The following are '
            'multiple choice questions (with answers) about physics.'),
            '</E>',
        ],
        round=[
            dict(role='HUMAN', prompt='</input>\nA. </A>\nB. </B>\nC. </C>\nD. </D>\nAnswer: '),
            dict(role='BOT', prompt='</target>'),
        ],
        end="end of dataset prompt template."
    ),
        column_token_map={
            'input': '</input>',
            'A': '</A>',
            'B': '</B>',
            'C': '</C>',
            'D': '</D>',
            'target': '</target>'
        },
        ice_token='</E>',
    )

```

其中，`round`用于指定在每轮对话中角色的 prompt 格式，同时也是为了呼应和补全 meta template 中的配置，因此，其接受的参数和规则均与 meta template 中的 `round` 一致。**在实际运行时，两处 prompt 的配置将会融合，同时如果某一字段被重复定义，则以 dataset config 中定义为准。**

而 `begin` 和 `end` 则除了支持 str 类型的输入，也支持 list 类型的输入，在其中用户可以通过组合 dict 和字符串实现对系统角色的融合。留意到例子中引入了 `fallback_role` 的设定，意味着若系统在 meta template 中 reserved_roles 中找不到 `role` 中的角色时，会自动替换成 `fallback_role` 中的角色。这个特征的设立是为了尽可能确保 prompt 模板的通用性。

结合 meta template，最终生成的 prompt 模板为：

```Plain
meta instruction
You are an AI assistant.
<|SYSTEM|>: The following are multiple choice questions (with answers) about college biology.
<|HUMAN|>: Which of the following is NOT a characteristic of an oligotrophic lake?
A. Low nutrient levels
B. High altitudes
C. Shallow water
D. Sand or gravel bottom
Answer: 脷\n
<|Inner Thoughts|>: None茔\n
<|Commands|>: None蝮\n
<|Results|>: None兒\n
<|MOSS|>: A氡\n
end of dataset prompt template.
end of conversion
```

特别地，由于这种 prompt 的数据结构（dict）与旧版的 label -> prompt 映射相同，本实现仅在字典的 keys 为 {`begin`, `round`, `end`} 的子集时将 prompt 的输入以新版规则进行解码，否则依然将字典以 label -> prompt 的形式进行解码。此外，该方案也允许新版 prompt 字典嵌套在旧版的 label -> prompt 字典中。例如，以下表达方式也是合法的 （摘自 `configs/datasets/mmlu.py`）：

```Python
prompt_template={
        target:
        dict(
            begin=[dict(role='SYSTEM', fallback_role='HUMAN', prompt='The following are '
                'multiple choice questions (with answers) about '
                f'{name.replace("_", " ")}.\n'),
                '</E>',
            ],
            round=[
                dict(role='HUMAN', prompt='</input>\nA. </A>\nB. </B>\nC. </C>\nD. </D>\nAnswer: '),
                dict(role='BOT', prompt=f'{target}'),
            ]
        )
        for target in ['A', 'B', 'C', 'D']  # use the actual answer
    }
```

### 无 meta template 时

为了保证后向兼容性，当用户未在 model config 中指定 meta template 时，`ICLPromptTemplate` 会将每个 dict 按照 `begin`, `prompt`, `end` 的顺序拼接为普通字符串。

### 多轮对话例子

在某些时候，一轮完整的交互中可能需要包含多轮对话。用户可以参考 `configs/datasets/gsm8k.py` 配置自己的模板。
