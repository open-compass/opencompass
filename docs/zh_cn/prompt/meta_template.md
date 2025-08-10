# Meta Template

## 背景

在 LLM 的 Supervised Fine-Tuning (SFT) 过程中，我们常常会根据实际的要求往对话内注入一些预定义的字符串，以求模型能按照一定的要求输出内容。例如，在一些 `chat` 模型的微调中，我们可能会在每段对话的开头加入系统层级的指令，并约定一套的格式表示用户与模型之间的对话。在一段对话中，模型期望文本的格式可能如下：

```Bash
Meta instruction: You are now a helpful and harmless AI assistant.
HUMAN: Hi!<eoh>\n
Bot: Hello! How may I assist you?<eob>\n
```

在评测时，我们也需要按照约定的格式输入问题，模型才能发挥出其最大的性能。

此外， API 模型也存在着类似的情况。一般 API 的对话模型都允许用户在调用时传入历史对话，还有些模型也允许传入 SYSTEM 层级的指令。为了更好地评测 API 模型的能力，我们希望在评测 API 模型时可以尽量让数据更贴合 API 模型本身的多轮对话模板，而并非把所有内容塞进一段指令当中。

因此，我们需要针对不同模型指定不同的解析模板。在 OpenCompass 中，我们将这套解析模板其称为 **Meta Template**。Meta Template 与模型的配置相绑定，在运行时与数据集的对话式模板相结合，最终产生最适合当前模型的 prompt。

```Python
# 指定时只需要把 meta_template 字段传入模型
models = [
    dict(
        type='AnyModel',
        meta_template = ...,  # meta tmplate
    )
]
```

接下来，我们会介绍 Meta Template 在两种模型上的配置方法。建议读者在阅读本章前，先了解[对话式模板](./prompt_template.md#对话式-prompt)的基本语法。

```{note}
在某些情况下（例如对基座的测试），我们并不需要在正常对话中注入任何的指令，此时我们可以将 meta template 置空。在这种情况下，模型接收到的 prompt 仅由数据集配置定义，是一个普通的字符串。若数据集配置使用的是对话式模板，不同角色的发言将会由 \n 拼接而成。
```

## 应用在语言模型上

下图展示了在 2-shot learning 的情况下，数据从数据集中经过 prompt template 和 meta template，最终构建出 prompt 的几种情况。读者可以该图为参考，方便理解后续的章节。

![](https://user-images.githubusercontent.com/22607038/251195073-85808807-6359-44df-8a19-9f5d00c591ec.png)

我们将会结合几个例子讲解 meta template 的定义方式。

假设根据数据集的对话式模板，产生了下面的 PromptList：

```python
PromptList([
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='2'),
    dict(role='HUMAN', prompt='2+2=?'),
    dict(role='BOT', prompt='4'),
])
```

我们希望把这段对话传到一个已经经过 SFT 的模型。模型约定的对话中不同的角色的发言以`<角色名>:`开头，并固定以一个特殊 token 和 \\n 结尾。以下是模型期望接收到的完整字符串：

```Plain
<HUMAN>: 1+1=?<eoh>
<BOT>: 2<eob>
<HUMAN>: 2+2=?<eoh>
<BOT>: 4<eob>
```

在 meta template 中，我们只需要把每轮对话的格式抽象为如下配置即可：

```Python
# model meta template
meta_template = dict(
    round=[
          dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n'),
          dict(role='BOT', begin='<BOT>: ', end='<eob>\n'),
    ],
 )
```

______________________________________________________________________

有的数据集中可能会引入 SYSTEM 级别的角色：

```python
PromptList([
    dict(role='SYSTEM', fallback_role='HUMAN', prompt='Solve the following math questions'),
    dict(role='HUMAN', prompt='1+1=?'),
    dict(role='BOT', prompt='2'),
    dict(role='HUMAN', prompt='2+2=?'),
    dict(role='BOT', prompt='4'),
])
```

假设模型同样接受 SYSTEM 这个角色，且期望输入为：

```Bash
<SYSTEM>: Solve the following math questions<eosys>\n
<HUMAN>: 1+1=?<eoh>\n
<BOT>: 2<eob>\n
<HUMAN>: 2+2=?<eoh>\n
<BOT>: 4<eob>\n
end of conversation
```

我们就可以把 SYSTEM 角色的定义放进 `reserved_roles` 中。`reserved_roles` 中的角色不会在常规对话中出现，但允许数据集配置的对话式模板在 `begin` 或者 `end` 中调用。

```Python
# model meta template
meta_template = dict(
    round=[
          dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n'),
          dict(role='BOT', begin='<BOT>: ', end='<eob>\n'),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<SYSTEM>: ', end='<eosys>\n'),],
 ),
```

若模型并不接受 SYSTEM 角色，则**不需要**配置此项，也能正常运行。这种情况下，模型会接收到的字符串变成了：

```Python
<HUMAN>: Solve the following math questions<eoh>\n
<HUMAN>: 1+1=?<eoh>\n
<BOT>: 2<eob>\n
<HUMAN>: 2+2=?<eoh>\n
<BOT>: 4<eob>\n
end of conversation
```

这是因为在 OpenCompass 预定义的数据集中，每个 `SYSTEM` 发言都会有一个 `fallback_role='HUMAN'`，即若 meta template 中的 `SYSTEM` 角色不存在，发言者会被切换至 `HUMAN` 角色。

______________________________________________________________________

有的模型还可能需要考虑在对话开始或结束时嵌入其它字符串，如系统指令：

```Bash
Meta instruction: You are now a helpful and harmless AI assistant.
<SYSTEM>: Solve the following math questions<eosys>\n
<HUMAN>: 1+1=?<eoh>\n
<BOT>: 2<eob>\n
<HUMAN>: 2+2=?<eoh>\n
<BOT>: 4<eob>\n
end of conversation
```

此时，我们可以通过指定 `begin` 和 `end` 参数指定这些字符串。

```Python
meta_template = dict(
    round=[
          dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n'),
          dict(role='BOT', begin='<BOT>: ', end='<eob>\n'),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<SYSTEM>: ', end='<eosys>\n'),],
    begin="Meta instruction: You are now a helpful and harmless AI assistant.",
    end="end of conversion",
 ),
```

______________________________________________________________________

在**生成式**的任务评测中，我们也不会将答案直接输入模型，而是通过截断 prompt，在保留上文的同时，把模型输出的答案留空。

```Bash
Meta instruction: You are now a helpful and harmless AI assistant.
<SYSTEM>: Solve the following math questions<eosys>\n
<HUMAN>: 1+1=?<eoh>\n
<BOT>: 2<eob>\n
<HUMAN>: 2+2=?<eoh>\n
<BOT>:
```

我们只需要把 BOT 的配置中把 `generate` 字段置为 True ，OpenCompass 即会将 BOT 的最后一句话留给模型生成：

```Python
meta_template = dict(
    round=[
          dict(role='HUMAN', begin='<HUMAN>: ', end='<eoh>\n'),
          dict(role='BOT', begin='<BOT>: ', end='<eob>\n', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<SYSTEM>: ', end='<eosys>\n'),],
    begin="Meta instruction: You are now a helpful and harmless AI assistant.",
    end="end of conversion",
 ),
```

需要注意的是，`generate` 仅影响生成式推理。在进行判别式推理时，模型接受到的 prompt 仍然是完整的。

### 全量字段介绍

```Bash
models = [
    dict(meta_template = dict(
            begin="Meta instruction: You are now a helpful and harmless AI assistant.",
            round=[
                    dict(role='HUMAN', begin='HUMAN: ', end='<eoh>\n'),  # begin and end can be a list of strings or integers.
                    dict(role='THOUGHTS', begin='THOUGHTS: ', end='<eot>\n', prompt='None'), # Here we can set the default prompt, which may be overridden by the speicfic dataset
                    dict(role='BOT', begin='BOT: ', generate=True, end='<eob>\n'),
            ],
            end="end of conversion",
            reserved_roles=[dict(role='SYSTEM', begin='SYSTEM: ', end='\n'),],
            eos_token_id=10000,
         ),
     )
]
```

meta_template 是一个字典，该字典可以包含以下数个字段：

- `begin`，`end` ：(str，可选) prompt 的开头和结尾，通常是一些系统级别的指令。

- `round`：(list) 每一轮对话的模板格式。每轮对话的 prompt 内容由数据集配置的对话式模板控制。

- `reserved_roles`:（list，可选）指定 `round` 中并未出现，但有可能在数据集配置中用到的的预留角色，例如 `SYSTEM` 角色。

- `eos_token_id`:（int, 可选）：指定了该模型的 eos token 的 id。如果不设置，则默认为 tokenizer 中的 eos token id。它的主要作用是在生成式任务中，截取模型的输出结果，因此一般应该被设置为 generate=True 的项所对应的 end 的第一个 token id。

meta_template 的 `round` 指定了一轮对话中每个角色说话的格式，接受一个字典组成的列表，每个字典的关键字如下：

- `role`（str）: 参与对话的角色名，该字符串并不影响实际的 prompt。

- `begin`, `end` (str): 指定该角色在说话时的固定开头或结尾。

- `prompt` (str)：角色的 prompt。在 meta template 中允许留空，但此时必须在数据集配置的 prompt 中指定。

- `generate` (bool): 指定为 True 时，该角色即为模型扮演的角色。在生成任务中，模型接收到的 prompt 会截止到该角色的 `begin` 处，剩下的内容由模型补全。

## 应用在 API 模型上

API 模型的 meta template 与普通模型的 meta template 类似，但配置更为简单。用户可以根据情况，直接使用下面的两种配置之一，即可以多轮对话的方式评测 API 模型：

```Bash
# 若 API 模型不支持 system 指令
meta_template=dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
)

# 若 API 模型支持 system 指令
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

### 原理

尽管不同 API 模型接受的数据结构不一，但总体上不乏共通之处。接受对话历史的接口里通常允许用户传入以下三个角色的 prompt：

- 用户

- 机器人

- 系统 （可选）

据此 OpenCompass 为 API 模型预设了三个 `api_role`：`HUMAN`, `BOT`, `SYSTEM`，同时约定 API 模型接受的输入除了普通字符串外，还有一种以 `PromptList` 结构表示对话的中间格式。API 模型会将对话重新以多轮对话格式打包，发送至后端。但要激活此功能，需要用户使用上面的 meta template 中把数据集 prompt 模板中的角色 `role` 映射到对应的 `api_role` 中。下图展示了 API 模型接受的输入与 Prompt Template 、Meta Template 之间的关系。

![](https://user-images.githubusercontent.com/22607038/251195872-63aa7d30-045a-4837-84b5-11b09f07fb18.png)

## 调试

如果需要调试 prompt，建议在准备好配置文件后，使用 `tools/prompt_viewer.py` 脚本预览模型实际接收到的 prompt。阅读[这里](../tools.md#prompt-viewer)了解更多。
