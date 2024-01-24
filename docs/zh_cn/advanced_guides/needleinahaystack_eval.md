# 大海捞针(Needle In A Haystack)实验评估

## 大海捞针测试简介

大海捞针测试（灵感来自 [NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/LLMNeedleHaystackTester.py)）是指通过将关键信息随机插入一段长文本的不同位置，形成大语言模型 (LLM) 的Prompt，通过测试大模型是否能从长文本中提取出关键信息，从而测试大模型的长文本信息提取能力的一种方法，可反映LLM长文本理解的基本能力。

## 数据集介绍

`Skywork/ChineseDomainModelingEval` 数据集收录了 2023 年 9 月至 10 月期间发布的高质量中文文章，涵盖了多个领域。这些文章确保了公平且具有挑战性的基准测试。

## 文件介绍

该数据集包括特定领域的文件：

- `zh_finance.jsonl` - 金融
- `zh_game.jsonl` - 游戏
- `zh_government.jsonl` - 政务
- `zh_movie.jsonl` - 电影
- `zh_tech.jsonl` - 技术
- `zh_general.jsonl` - 综合

这些文件用于评估LLM对不同特定领域的理解能力。

### 评估步骤

1. 从 [Skywork/ChineseDomainModelingEval](https://huggingface.co/datasets/Skywork/ChineseDomainModelingEval/tree/main) 下载数据集。

2. 将下载的文件放置在 `opencompass/data/CDME/` 下。`CDME` 目录中的预期文件结构如下：

   ```
   opencompass/
   ├── configs
   ├── docs
   ├── data
   │   └── CDME
   │       ├── processed
   │       ├── README.md
   │       ├── zh_finance.jsonl
   │       ├── zh_game.jsonl
   │       ├── zh_general.jsonl
   │       ├── zh_government.jsonl
   │       ├── zh_movie.jsonl
   │       └── zh_tech.jsonl
   ├── LICENSE
   ├── opencompass
   ├── outputs
   ├── run.py
   ├── more...
   ```

### 环境配置

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

### 配置数据集

在最新版本中，数据集不再通过运行脚本手动生成，而是通过在配置文件中动态定义和加载。用户需要根据自己的需求，在配置文件中指定数据集的参数。这种方法提供了更大的灵活性和定制化选项。

#### 数据集配置示例

以下是一个数据集配置的示例，展示了如何在配置文件 `configs/datasets/cdme/cdme8k.py` 中定义一个数据集。这个示例展示了一个 8000 tokens 长度的中文数据集配置：

```python
for original_context_length in context_lengths:
    for depth_percent in generate_depth_percents(
            document_depth_percent_intervals,
            document_depth_percent_interval_type):
        dataset_dict = {
            'abbr': f'CDME_Length{original_context_length}Depth{int(depth_percent)}',
            'type': CDMEDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 200,
            'guide': True,
            'language': 'Chinese',
            'needle': '\n小明最喜欢的实习的地点就是上海人工智能实验室。\n',
            'retrieval_question': '小明最喜欢的实习地点是哪里？请按照“小明最喜欢的实习地点就是________。”的格式回答。',
            'reader_cfg': cdme_reader_cfg,
            'infer_cfg': cdme_infer_cfg,
            'eval_cfg': cdme_eval_cfg
        }
        cdme_datasets.append(dataset_dict)
```

在这个配置中，主要参数包括：

- `abbr`: 数据集的简称。
- `type`: 数据集类型。
- `path`: 数据集文件的路径。
- `length`: 上下文长度（以token为单位）。
- `depth`: 文档深度百分比。
- `tokenizer_model`: 使用的tokenizer 模型。
- `file_list`: 数据源文件列表。
- `num_repeats_per_file`: 每个文件重复的次数。
- `length_buffer`: 长度缓冲区。
- `guide`: 是否为引导式数据集。
- `language`: 数据集的语言。
- `needle`: 在数据集中要查找的特定文本（针）。
- `retrieval_question`: 用于提示模型检索的问题。
- `reader_cfg`, `infer_cfg`, `eval_cfg`: 分别对应读取、推理和评估的配置。

通过在配置文件中定义这些参数，您可以灵活地创建适合您需求的数据集。配置文件提供了一种高度可定制和扩展的方式来管理数据集的生成和使用。

### 多根针大海捞针测试

最新版本中引入了多根针大海捞针测试，允许在同一个数据集中插入多个不同的针（文本片段）。这些针将根据给定的深度参数依次插入数据集中。相对于单针测试，多针测试提供了更复杂的数据处理场景。

#### 多针数据集配置示例

以下是一个配置多针数据集的示例，展示了如何在配置文件 `configs/datasets/cdme/multi_needle/cdme8k_cot3_italy.py` 中定义多针数据集。这个示例展示了一个包含三根针的数据集配置：

```python
# 数据集基础配置
base_path = './data/CDME'
file_list = ['zh_finance.jsonl']

# 针（Needles）定义
needles = [
    '\n意大利的佛罗伦萨有一家名为"La Giostra"的餐馆，是整个佛罗伦萨中排行第一的餐馆。\n',
    '"La Giostra"餐馆的特色菜肴是松露奶酪通心粉。',
    '松露奶酪通心粉是该家餐馆的有着意大利皇室烹饪血统的大厨Jack制作'
]

# 配置参数
retrieval_question = ("制作佛罗伦萨中排行第一的餐馆的特色菜肴的人叫什么？"
                      "请按照'制作佛罗伦萨中排行第一的餐馆的特色菜肴的人叫______。'的格式回答。")
answer = "制作佛罗伦萨中排行第一的餐馆的特色菜肴的人叫Jack"
keyword = "Jack"
diff = 25

# 数据集生成循环
for original_context_length in context_lengths:
    for depth_percent in generate_depth_percents(
            document_depth_percent_intervals,
            document_depth_percent_interval_type):
        dataset_dict = {
            # 其他配置项...
            'needles': needles,
            'diff': diff,
            'keyword': keyword,
            # 其他配置项...
        }
        cdme_datasets.append(dataset_dict)
```

在这个配置中，除了标准的参数之外，主要新增了以下几个关键参数：

- `needles`: 一个包含多个字符串的列表，每个字符串代表一个要插入的针。
- `diff`: 定义后续针相对于第一根针的插入深度增量。
- `keyword`: 用于在评分过程中对答案进行校正的关键词。

#### 评分机制的改变

在 `opencompass/datasets/cdme/cdme_multi.py` 的源代码中，对于多根针的数据集，评分机制有所不同。新增了以下代码段，用于基于 `keyword` 对预测的答案进行评分校正：

```python
if keyword in prediction:
    print(f'{keyword} is in {prediction}')
    score = 100
else:
    print(f'{keyword} is not in {prediction}')
    score = 0.2 * score
```

这段代码意味着如果预测的答案中包含了 `keyword`，则会给予高分（如100分）。如果不包含，则分数会被大幅度降低（原分数的20%）。这种评分机制更加注重关键词的准确性，是对传统评分方法的一个重要补充。

### 评估

#### 使用 `internlm` 模型进行评估

例如，使用 `internlm` 模型进行评估，可以使用以下命令：

```bash
python run.py configs/eval_needleinahaystack.py --slurm -p partition_name -q auto --max-num-workers 32
```

这个命令将启动评估流程，其中模型将试图在生成的数据集中找到指定的“针”。参数 `-p partition_name -q auto` 和 `--max-num-workers 32` 用于指定 Slurm 队列和最大工作进程数。

#### 使用 `LMDeploy` 进行大规模文本评估

当评估特别长的文本（例如 200k tokens）时，常规方法可能会导致显存不足。在这种情况下，可以使用量化模型进行评估。这可以通过使用 `LMDeploy` 工具（[LMDeploy](https://github.com/InternLM/lmdeploy)）完成。

安装和配置 `LMDeploy` 的详细信息可以在其 GitHub 页面上找到。安装完成后，可以使用 `configs/eval_needleinahaystack_turbomind.py` 配置文件中定义的 `TurboMindModel` 模型进行评估。

以下是 `configs/eval_needleinahaystack_turbomind.py` 文件的示例配置：

```python
from opencompass.models.turbomind import TurboMindModel
from mmengine.config import read_base

with read_base():
    from .datasets.cdme.cdme200k import cdme_datasets

datasets = [*cdme_datasets]

internlm_meta_template = dict(round=[
    dict(role='HUMAN', begin=':', end='\n'),
    dict(role='BOT', begin=':', end='<eoa>\n', generate=True),
],
                              eos_token_id=103028)

models = [
    dict(
        type=TurboMindModel,
        abbr='internlm-chat-20b-turbomind',
        path='./turbomind',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        concurrency=8,
        meta_template=internlm_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
```

在这个配置中，`TurboMindModel` 模型结合了 `LMDeploy` 的功能，适用于处理大规模文本数据集，有效减少显存的占用。

### Score计算方法

在 `CDMEEvaluator` 类中，我们使用两个主要方法来计算得分：`levenshtein_distance` 和 `score`。下面是这些方法的详细介绍和实现。

#### Levenshtein Distance

Levenshtein 距离是一种衡量两个字符串差异的方法。它表示将一个字符串转换为另一个所需的最少单字符编辑（插入、删除或替换）的数量。

```python
def levenshtein_distance(self, s1, s2):
    if len(s1) < len(s2):
        return self.levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
```

#### Score Calculation

得分计算方法 `score` 接受预测值和参考值两个列表，并计算每对预测值和参考值的编辑距离和得分。

```python
def score(self, predictions, references):
    if len(predictions) != len(references):
        return {"error": "predictions and references have different lengths"}

    total_score = 0
    details = []
    for prediction, reference in zip(predictions, references):
        prediction = re.sub(r'\s+', '', prediction)
        reference = re.sub(r'\s+', '', reference)
        edit_distance = self.levenshtein_distance(prediction, reference)
        max_len = max(len(prediction), len(reference))
        score = 100 * (1 - edit_distance / max_len) if max_len != 0 else 100

        detail = {
            "pred": prediction,
            "answer": reference,
            "edit_distance": edit_distance,
            "score": score
        }
        total_score += score
        details.append(detail)

    average_score = total_score / len(predictions) if predictions else 0
    result = {"score": average_score, "details": details}
    return result
```

该方法首先去除预测值和参考值中的所有空白字符，然后计算它们之间的 Levenshtein 距离。得分计算为 100 减去基于编辑距离的百分比损失。最后，返回每个预测值的详细得分和平均得分。

### 可视化

可以使用 `tools_needleinahaystack.py` 脚本来对 CSV 文件进行可视化绘图。这个脚本支持通过 `--path` 参数指定一个或多个 CSV 文件的路径，并且可以使用 `--dataset_length` 参数来指定数据集的长度。

#### 使用示例

绘制单个 CSV 文件的可视化：

```bash
python tools/tools_needleinahaystack.py --path 'outputs/default/20231216_161457/summary/summary_20231216_161457.csv'
```

绘制多个 CSV 文件的可视化：

```bash
python tools/tools_needleinahaystack.py --path 'path_to_first_csv.csv' 'path_to_second_csv.csv'
```

指定数据集长度进行可视化,此参数用于生成可视化图中的图表标题：

```bash
python tools/tools_needleinahaystack.py --path 'path_to_csv.csv' --dataset_length 200K
```

目前该方案仅支持 CDME 数据集，我们欢迎社区贡献更多的数据集。

如果使用了该方法，请添加引用:

```bibtex

@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished={\url{https://github.com/open-compass/opencompass}},
    year={2023}
}

@misc{LLMTest_NeedleInAHaystack,
  title={LLMTest Needle In A Haystack - Pressure Testing LLMs},
  author={gkamradt},
  year={2023},
  howpublished={\url{https://github.com/gkamradt/LLMTest_NeedleInAHaystack}}
}

@misc{wei2023skywork,
      title={Skywork: A More Open Bilingual Foundation Model},
      author={Tianwen Wei and Liang Zhao and Lichang Zhang and Bo Zhu and Lijie Wang and Haihua Yang and Biye Li and Cheng Cheng and Weiwei Lü and Rui Hu and Chenxia Li and Liu Yang and Xilin Luo and Xuejie Wu and Lunan Liu and Wenjun Cheng and Peng Cheng and Jianhao Zhang and Xiaoyu Zhang and Lei Lin and Xiaokun Wang and Yutuan Ma and Chuanhai Dong and Yanqi Sun and Yifu Chen and Yongyi Peng and Xiaojuan Liang and Shuicheng Yan and Han Fang and Yahui Zhou},
      year={2023},
      eprint={2310.19341},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```
