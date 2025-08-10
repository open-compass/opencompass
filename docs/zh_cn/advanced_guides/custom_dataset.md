# 自定义数据集

本教程仅供临时性的、非正式的数据集使用，如果所用数据集需要长期使用，或者存在定制化读取 / 推理 / 评测需求的，强烈建议按照 [new_dataset.md](./new_dataset.md) 中介绍的方法进行实现。

在本教程中，我们将会介绍如何在不实现 config，不修改 OpenCompass 源码的情况下，对一新增数据集进行测试的方法。我们支持的任务类型包括选择 (`mcq`) 和问答 (`qa`) 两种，其中 `mcq` 支持 `ppl` 推理和 `gen` 推理；`qa` 支持 `gen` 推理。

## 数据集格式

我们支持 `.jsonl` 和 `.csv` 两种格式的数据集。

### 选择题 (`mcq`)

对于选择 (`mcq`) 类型的数据，默认的字段如下：

- `question`: 表示选择题的题干
- `A`, `B`, `C`, ...: 使用单个大写字母表示选项，个数不限定。默认只会从 `A` 开始，解析连续的字母作为选项。
- `answer`: 表示选择题的正确答案，其值必须是上述所选用的选项之一，如 `A`, `B`, `C` 等。

对于非默认字段，我们都会进行读入，但默认不会使用。如需使用，则需要在 `.meta.json` 文件中进行指定。

`.jsonl` 格式样例如下：

```jsonl
{"question": "165+833+650+615=", "A": "2258", "B": "2263", "C": "2281", "answer": "B"}
{"question": "368+959+918+653+978=", "A": "3876", "B": "3878", "C": "3880", "answer": "A"}
{"question": "776+208+589+882+571+996+515+726=", "A": "5213", "B": "5263", "C": "5383", "answer": "B"}
{"question": "803+862+815+100+409+758+262+169=", "A": "4098", "B": "4128", "C": "4178", "answer": "C"}
```

`.csv` 格式样例如下:

```csv
question,A,B,C,answer
127+545+588+620+556+199=,2632,2635,2645,B
735+603+102+335+605=,2376,2380,2410,B
506+346+920+451+910+142+659+850=,4766,4774,4784,C
504+811+870+445=,2615,2630,2750,B
```

### 问答题 (`qa`)

对于问答 (`qa`) 类型的数据，默认的字段如下：

- `question`: 表示问答题的题干
- `answer`: 表示问答题的正确答案。可缺失，表示该数据集无正确答案。

对于非默认字段，我们都会进行读入，但默认不会使用。如需使用，则需要在 `.meta.json` 文件中进行指定。

`.jsonl` 格式样例如下：

```jsonl
{"question": "752+361+181+933+235+986=", "answer": "3448"}
{"question": "712+165+223+711=", "answer": "1811"}
{"question": "921+975+888+539=", "answer": "3323"}
{"question": "752+321+388+643+568+982+468+397=", "answer": "4519"}
```

`.csv` 格式样例如下：

```csv
question,answer
123+147+874+850+915+163+291+604=,3967
149+646+241+898+822+386=,3142
332+424+582+962+735+798+653+214=,4700
649+215+412+495+220+738+989+452=,4170
```

## 命令行列表

自定义数据集可直接通过命令行来调用开始评测。

```bash
python run.py \
    --models hf_llama2_7b \
    --custom-dataset-path xxx/test_mcq.csv \
    --custom-dataset-data-type mcq \
    --custom-dataset-infer-method ppl
```

```bash
python run.py \
    --models hf_llama2_7b \
    --custom-dataset-path xxx/test_qa.jsonl \
    --custom-dataset-data-type qa \
    --custom-dataset-infer-method gen
```

在绝大多数情况下，`--custom-dataset-data-type` 和 `--custom-dataset-infer-method` 可以省略，OpenCompass 会根据以下逻辑进行设置：

- 如果从数据集文件中可以解析出选项，如 `A`, `B`, `C` 等，则认定该数据集为 `mcq`，否则认定为 `qa`。
- 默认 `infer_method` 为 `gen`。

## 配置文件

在原配置文件中，直接向 `datasets` 变量中添加新的项即可即可。自定义数据集亦可与普通数据集混用。

```python
datasets = [
    {"path": "xxx/test_mcq.csv", "data_type": "mcq", "infer_method": "ppl"},
    {"path": "xxx/test_qa.jsonl", "data_type": "qa", "infer_method": "gen"},
]
```

## 数据集补充信息 `.meta.json`

OpenCompass 会默认尝试对输入的数据集文件进行解析，因此在绝大多数情况下，`.meta.json` 文件都是 **不需要** 的。但是，如果数据集的字段名不是默认的字段名，或者需要自定义提示词，则需要在 `.meta.json` 文件中进行指定。

我们会在数据集同级目录下，以文件名+`.meta.json` 的形式放置一个表征数据集使用方法的文件，样例文件结构如下：

```tree
.
├── test_mcq.csv
├── test_mcq.csv.meta.json
├── test_qa.jsonl
└── test_qa.jsonl.meta.json
```

该文件可能字段如下：

- `abbr` (str): 数据集缩写，作为该数据集的 ID。
- `data_type` (str): 数据集类型，可选值为 `mcq` 和 `qa`.
- `infer_method` (str): 推理方法，可选值为 `ppl` 和 `gen`.
- `human_prompt` (str): 用户提示词模板，用于生成提示词。模板中的变量使用 `{}` 包裹，如 `{question}`，`{opt1}` 等。如存在 `template`，则该字段会被忽略。
- `bot_prompt` (str): 机器人提示词模板，用于生成提示词。模板中的变量使用 `{}` 包裹，如 `{answer}` 等。如存在 `template`，则该字段会被忽略。
- `template` (str or dict): 问题模板，用于生成提示词。模板中的变量使用 `{}` 包裹，如 `{question}`，`{opt1}` 等。相关语法见[此处](../prompt/prompt_template.md) 关于 `infer_cfg['prompt_template']['template']` 的内容。
- `input_columns` (list): 输入字段列表，用于读入数据。
- `output_column` (str): 输出字段，用于读入数据。
- `options` (list): 选项列表，用于读入数据，仅在 `data_type` 为 `mcq` 时有效。

样例如下：

```json
{
    "human_prompt": "Question: 127 + 545 + 588 + 620 + 556 + 199 =\nA. 2632\nB. 2635\nC. 2645\nAnswer: Let's think step by step, 127 + 545 + 588 + 620 + 556 + 199 = 672 + 588 + 620 + 556 + 199 = 1260 + 620 + 556 + 199 = 1880 + 556 + 199 = 2436 + 199 = 2635. So the answer is B.\nQuestion: {question}\nA. {A}\nB. {B}\nC. {C}\nAnswer: ",
    "bot_prompt": "{answer}"
}
```

或者

```json
{
    "template": "Question: {my_question}\nX. {X}\nY. {Y}\nZ. {Z}\nW. {W}\nAnswer:",
    "input_columns": ["my_question", "X", "Y", "Z", "W"],
    "output_column": "my_answer",
}
```
