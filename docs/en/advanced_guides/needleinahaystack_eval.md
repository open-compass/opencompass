# Needle In A Haystack Experiment Evaluation

## Introduction to the Needle In A Haystack Test

The Needle In A Haystack test, inspired by [NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/LLMNeedleHaystackTester.py), involves embedding key information randomly in different parts of a long text to form prompts for large language models (LLMs). This test evaluates an LLM's ability to extract crucial information from lengthy texts, reflecting its fundamental capability in understanding long-form content.

## Dataset Overview

The `Skywork/ChineseDomainModelingEval` dataset includes high-quality Chinese articles published between September and October 2023, covering multiple domains. These articles ensure a fair and challenging benchmark.

## File Descriptions

The dataset includes files specific to different domains:

- `zh_finance.jsonl` - Finance
- `zh_game.jsonl` - Gaming
- `zh_government.jsonl` - Government Affairs
- `zh_movie.jsonl` - Movies
- `zh_tech.jsonl` - Technology
- `zh_general.jsonl` - General

These files are used to assess the LLM’s understanding in various specific fields.

### Evaluation Steps

1. Download the dataset from [Skywork/ChineseDomainModelingEval](https://huggingface.co/datasets/Skywork/ChineseDomainModelingEval/tree/main).

2. Place the downloaded files in `opencompass/data/CDME/`. The expected file structure in the `CDME` directory:

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
   └── more...
   ```

### Environment Setup

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

### Generating the Dataset

Run the following command to generate the dataset:

```bash
python tools/gen_needleinahaystack.py
```

You can set specific parameters in `tools/gen_needleinahaystack.py` to choose the dataset needed for the task. The main parameters include:

- `needle`: The specific text (needle) to be found in the dataset.
- `retrieval_question`: The question to prompt the model for retrieval.
- `context_lengths`: Specifies the context lengths (in tokens) for different testing scenarios.
- `document_depth_percent_intervals`: The intervals for dividing the document depth to determine where to insert the “needle”.

### Evaluation

For example, to evaluate using the `internlm` model, you can use the following command:

```bash
python run.py configs/eval_hf_internlm_chat_20b_cdme.py --slurm -p partition_name-q auto --max-num-workers 32
```

This command initiates the evaluation process where the model will attempt to find the specified “needle” in the generated dataset. The parameters `-p partition_name-q auto` and `--max-num-workers 32` specify the Slurm queue and the maximum number of work processes.

### Score Calculation Method

In the `CDMEEvaluator` class, we use two main methods to calculate the score: `levenshtein_distance` and `score`. Here are detailed introductions and implementations of these methods.

#### Levenshtein Distance

Levenshtein distance is a method for measuring the difference between two strings. It represents the minimum number of single-character edits (insertions, deletions, or substitutions) needed to change one string into another.

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

The `score` calculation method accepts lists of predictions and references and calculates the edit distance and score for each pair of prediction and reference.

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

This method first removes all whitespace characters in predictions and references and then calculates their Levenshtein distance. The score is calculated as 100 minus the percentage loss based on edit distance. Finally, it returns detailed scores and the average score for each prediction.

### Visualization

Use the `tools/viz_needleinahaystack.py` script to visualize the CSV files in the \`

outputs\` folder.

Currently, this scheme only supports the CDME dataset, and we welcome community contributions for more datasets.

If you use this method, please add a citation:

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
