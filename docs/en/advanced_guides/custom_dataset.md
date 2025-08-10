# Custom Dataset Tutorial

This tutorial is intended for temporary and informal use of datasets. If the dataset requires long-term use or has specific needs for custom reading/inference/evaluation, it is strongly recommended to implement it according to the methods described in [new_dataset.md](./new_dataset.md).

In this tutorial, we will introduce how to test a new dataset without implementing a config or modifying the OpenCompass source code. We support two types of tasks: multiple choice (`mcq`) and question & answer (`qa`). For `mcq`, both ppl and gen inferences are supported; for `qa`, gen inference is supported.

## Dataset Format

We support datasets in both `.jsonl` and `.csv` formats.

### Multiple Choice (`mcq`)

For `mcq` datasets, the default fields are as follows:

- `question`: The stem of the multiple-choice question.
- `A`, `B`, `C`, ...: Single uppercase letters representing the options, with no limit on the number. Defaults to parsing consecutive letters strating from `A` as options.
- `answer`: The correct answer to the multiple-choice question, which must be one of the options used above, such as `A`, `B`, `C`, etc.

Non-default fields will be read in but are not used by default. To use them, specify in the `.meta.json` file.

An example of the `.jsonl` format:

```jsonl
{"question": "165+833+650+615=", "A": "2258", "B": "2263", "C": "2281", "answer": "B"}
{"question": "368+959+918+653+978=", "A": "3876", "B": "3878", "C": "3880", "answer": "A"}
{"question": "776+208+589+882+571+996+515+726=", "A": "5213", "B": "5263", "C": "5383", "answer": "B"}
{"question": "803+862+815+100+409+758+262+169=", "A": "4098", "B": "4128", "C": "4178", "answer": "C"}
```

An example of the `.csv` format:

```csv
question,A,B,C,answer
127+545+588+620+556+199=,2632,2635,2645,B
735+603+102+335+605=,2376,2380,2410,B
506+346+920+451+910+142+659+850=,4766,4774,4784,C
504+811+870+445=,2615,2630,2750,B
```

### Question & Answer (`qa`)

For `qa` datasets, the default fields are as follows:

- `question`: The stem of the question & answer question.
- `answer`: The correct answer to the question & answer question. It can be missing, indicating the dataset has no correct answer.

Non-default fields will be read in but are not used by default. To use them, specify in the `.meta.json` file.

An example of the `.jsonl` format:

```jsonl
{"question": "752+361+181+933+235+986=", "answer": "3448"}
{"question": "712+165+223+711=", "answer": "1811"}
{"question": "921+975+888+539=", "answer": "3323"}
{"question": "752+321+388+643+568+982+468+397=", "answer": "4519"}
```

An example of the `.csv` format:

```csv
question,answer
123+147+874+850+915+163+291+604=,3967
149+646+241+898+822+386=,3142
332+424+582+962+735+798+653+214=,4700
649+215+412+495+220+738+989+452=,4170
```

## Command Line List

Custom datasets can be directly called for evaluation through the command line.

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

In most cases, `--custom-dataset-data-type` and `--custom-dataset-infer-method` can be omitted. OpenCompass will

set them based on the following logic:

- If options like `A`, `B`, `C`, etc., can be parsed from the dataset file, it is considered an `mcq` dataset; otherwise, it is considered a `qa` dataset.
- The default `infer_method` is `gen`.

## Configuration File

In the original configuration file, simply add a new item to the `datasets` variable. Custom datasets can be mixed with regular datasets.

```python
datasets = [
    {"path": "xxx/test_mcq.csv", "data_type": "mcq", "infer_method": "ppl"},
    {"path": "xxx/test_qa.jsonl", "data_type": "qa", "infer_method": "gen"},
]
```

## Supplemental Information for Dataset `.meta.json`

OpenCompass will try to parse the input dataset file by default, so in most cases, the `.meta.json` file is **not necessary**. However, if the dataset field names are not the default ones, or custom prompt words are required, it should be specified in the `.meta.json` file.

The file is placed in the same directory as the dataset, with the filename followed by `.meta.json`. An example file structure is as follows:

```tree
.
├── test_mcq.csv
├── test_mcq.csv.meta.json
├── test_qa.jsonl
└── test_qa.jsonl.meta.json
```

Possible fields in this file include:

- `abbr` (str): Abbreviation of the dataset, serving as its ID.
- `data_type` (str): Type of dataset, options are `mcq` and `qa`.
- `infer_method` (str): Inference method, options are `ppl` and `gen`.
- `human_prompt` (str): User prompt template for generating prompts. Variables in the template are enclosed in `{}`, like `{question}`, `{opt1}`, etc. If `template` exists, this field will be ignored.
- `bot_prompt` (str): Bot prompt template for generating prompts. Variables in the template are enclosed in `{}`, like `{answer}`, etc. If `template` exists, this field will be ignored.
- `template` (str or dict): Question template for generating prompts. Variables in the template are enclosed in `{}`, like `{question}`, `{opt1}`, etc. The relevant syntax is in [here](../prompt/prompt_template.md) regarding `infer_cfg['prompt_template']['template']`.
- `input_columns` (list): List of input fields for reading data.
- `output_column` (str): Output field for reading data.
- `options` (list): List of options for reading data, valid only when `data_type` is `mcq`.

For example:

```json
{
    "human_prompt": "Question: 127 + 545 + 588 + 620 + 556 + 199 =\nA. 2632\nB. 2635\nC. 2645\nAnswer: Let's think step by step, 127 + 545 + 588 + 620 + 556 + 199 = 672 + 588 + 620 + 556 + 199 = 1260 + 620 + 556 + 199 = 1880 + 556 + 199 = 2436 + 199 = 2635. So the answer is B.\nQuestion: {question}\nA. {A}\nB. {B}\nC. {C}\nAnswer: ",
    "bot_prompt": "{answer}"
}
```

or

```json
{
    "template": "Question: {my_question}\nX. {X}\nY. {Y}\nZ. {Z}\nW. {W}\nAnswer:",
    "input_columns": ["my_question", "X", "Y", "Z", "W"],
    "output_column": "my_answer",
}
```
