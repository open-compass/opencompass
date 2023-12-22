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

### 生成数据集

运行以下命令以生成数据集：

```bash
python tools/tools_needleinahaystack.py \
  --processed_datasets_path './data/CDME/processed' \
  --data_path './data/CDME' \
  --tokenizer_model 'gpt-4' \
  --num_records_per_file 10 \
  --length_buffer 200 \
  --guided True \
  --file_list 'zh_finance.jsonl' \
  --context_lengths 1000 2000 3000 4000 5000 6000 7000 8000 \
  --needle '\n小明最喜欢的实习的地点就是上海人工智能实验室。\n' \
  --retrieval_question '小明最喜欢的实习地点是哪里？你的回答格式应该为“小明最喜欢的实习地点就是________。”' \
  --document_depth_percent_intervals 35 \
```

您可以在启动 `tools/tools_needleinahaystack.py` 时设置特定参数，以选择任务所需的数据集。主要参数包括：

- `needle`: 要在数据集中查找的指定文本（针）。
- `retrieval_question`: 用于提示模型检索的问题。
- `context_lengths`: 指定不同测试场景的上下文长度（以token为单位）。
- `document_depth_percent_intervals`: 文档深度的划分间隔数量，用于确定在何处插入“针”。

### 评估

例如，使用 `internlm` 模型进行评估，可以使用以下命令：

```bash
python run.py configs/eval_hf_internlm_chat_20b_cdme.py --slurm -p partition_name-q auto --max-num-workers 32
```

这个命令将启动评估流程，其中模型将试图在生成的数据集中找到指定的“针”。参数 `-p partition_name-q auto` 和 `--max-num-workers 32` 用于指定Slurm队列和最大工作进程数。

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

可以使用 `tools_needleinahaystack.py` 脚本，将 `outputs` 文件夹中的 CSV 文件进行可视化绘图。例如

```bash
python tools/tools_needleinahaystack.py \
  --plot \
  --csv_file_paths 'outputs/default/20231216_161457/summary/summary_20231216_161457.csv' 'outputs/default/20231217_022310/summary/summary_20231217_022310.csv'
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
