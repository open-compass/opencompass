# Needle In A Haystack Experiment Evaluation

## Introduction to the Needle In A Haystack Test

The Needle In A Haystack test (inspired by [NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/LLMNeedleHaystackTester.py)) involves embedding key information randomly within a long text to form prompts for large language models (LLMs). This test evaluates the LLM's ability to extract key information from extensive text, reflecting the fundamental capabilities of LLMs in understanding long texts.

## Dataset Overview

The `Skywork/ChineseDomainModelingEval` dataset includes high-quality Chinese articles published from September to October 2023, covering multiple domains. These articles ensure a fair and challenging benchmark test.

## File Description

The dataset includes files specific to certain domains:

- `zh_finance.jsonl` - Finance
- `zh_game.jsonl` - Gaming
- `zh_government.jsonl` - Government Affairs
- `zh_movie.jsonl` - Movies
- `zh_tech.jsonl` - Technology
- `zh_general.jsonl` - General

These files are used to evaluate the LLM's understanding capabilities in different specific areas.

### Evaluation Steps

1. Download the dataset from [Skywork/ChineseDomainModelingEval](https://huggingface.co/datasets/Skywork/ChineseDomainModelingEval/tree/main).

2. Place the downloaded files in `opencompass/data/CDME/`. The expected file structure in the `CDME` directory is as follows:

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

You can set specific parameters when launching `tools/tools_needleinahaystack.py` to select the datasets required for your task. Key parameters include:

- `needle`: The specific text (needle) to be located within the dataset.
- `retrieval_question`: The question used to prompt the model for retrieval.
- `context_lengths`: Specifies the context lengths (in tokens) for different test scenarios.
- `document_depth_percent_intervals`: The number of interval divisions for document depth to determine where to insert the "needle".

### Evaluation

For example, to evaluate using the `internlm` model, you can use the following command:

```bash
python run.py configs/eval_hf_internlm_chat_20b_cdme.py --slurm -p partition_name-q auto --max-num-workers 32
```

This command initiates the evaluation process, where the model will attempt to find the specified "needle" in the generated dataset. The parameters `-p partition_name-q auto` and `--max-num-workers 32` specify the Slurm queue and the maximum number of worker processes.

### Score Calculation Method

In the `CDMEEvaluator` class, we use two main methods to calculate scores: `levenshtein_distance` and `score`. Here is a detailed introduction and implementation

of these methods.

#### Levenshtein Distance

Levenshtein distance is a method for measuring the difference between two strings. It represents the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one string into the other.

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

The method first removes all whitespace characters from the predictions and references, then calculates the Levenshtein distance between them. The score is calculated as 100 minus the percentage loss based on the edit distance. Finally, it returns detailed scores for each prediction and the average score.

### Visualization

You can visualize the CSV files in the `outputs` folder using the `tools_needleinahaystack.py` script. For example:

```bash
python tools/tools_needleinahaystack.py \
  --plot \
  --csv_file_paths 'outputs/default/20231216_161457/summary/summary_20231216_161457.csv' 'outputs/default/20231217_022310/summary/summary_20231217_022310.csv'
```

Currently, this approach only supports the CDME dataset, and we welcome community contributions to more datasets.

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
      author={Tianwen Wei and others},
      year={2023},
      eprint={2310.19341},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```
