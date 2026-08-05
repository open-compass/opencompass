# ELBench

## Introduction

[ELBench](https://huggingface.co/datasets/ZeroLoss-Lab/ELBench) is a
multi-dimensional benchmark for **education-facing** large language models. This
directory ports ELBench to OpenCompass while keeping ELBench's **original data
files and naming** (Chinese filenames and directory layout) unchanged.

## Modules and subsets

| Module | Subset (original name) | Items | Type | Config |
|:-------|:-----------------------|:-----:|:-----|:-------|
| 安全可信 Safety | 安全回答 — benign answering only† | 250 | LLM-judge | `elbench_safety_judge.py` |
| 高阶育人 High-Level | 高阶育人-omni | 500 | Objective (MCQ) | `elbench_highlevel_omni_gen.py` |
| 高阶育人 High-Level | 高阶育人-edu | 500 | LLM-judge | `elbench_highlevel_edu_judge.py` |
| 通用 General | mmlu_pro / ceval | 196 / 208 | Objective (MCQ) | `elbench_general_gen.py` |
| 通用 General | math_500 / aime24 / aime25 / aime26 | 200 / 30 / 30 / 30 | Objective (math) | `elbench_general_gen.py` |
| 通用 General | ifeval | 200 | Rule-based* | data only |
| 基本教育 Basic Education | knowledge / question / cross / guided | — | Multi-turn** | data only |

† **The safety subset shipped here is an over-refusal check, not a safety
score.** ELBench's safety module has five task families. The public dataset
ships only `安全回答` (benign answering) and withholds the other four —
harmful-request refusal, safe guidance, teaching safety and adversarial
robustness — because their prompts contain harmful or jailbreak content. The
shipped family asks only whether a model answers *benign* questions without
over-refusing; every model reported in the ELBench paper scores 98.4–100 on it,
so it does not separate models, and a model with no safety guardrails at all
would score near 100 here as well. The four withheld families can be requested
from the ELBench authors (see the dataset card). Please do not report this
number as "ELBench safety".

\* `ifeval` uses ELBench's instruction-following rule engine; its data is shipped
but it is not wired into a single-turn evaluator here.

\** 基本教育 is a multi-turn teaching task that needs a multi-turn runtime; its
data is shipped under `基本教育/` but is not wired into OpenCompass's single-turn
pipeline.

## Data

Nothing is committed to OpenCompass. ELBench's original file names and layout are
kept; the public dataset hosts them at the repository root:

```text
<dataset root>/
├── 安全可信/通用-应回答/安全回答.jsonl
├── 高阶育人/{omni/高阶育人-omni.jsonl, edu/高阶育人-edu.jsonl}
├── 通用/{mmlu_pro_sampled, ceval_sampled, math_500_sampled, aime24_sampled, aime25, aime26, ifeval_sampled}.jsonl
└── 基本教育/{知识点讲解, 情景化出题, 跨学科教案生成, 引导式讲题}/*.yaml   # multi-turn, data only
```

The data root is resolved through `DATASET_SOURCE`, like other OpenCompass
datasets:

| `DATASET_SOURCE` | Behaviour |
|:-----------------|:----------|
| `HF` | download [`ZeroLoss-Lab/ELBench`](https://huggingface.co/datasets/ZeroLoss-Lab/ELBench) from HuggingFace |
| `ModelScope` | download the ModelScope mirror — **not published yet**, use `HF` |
| unset (default) | **local mode** — read `$COMPASS_DATA_CACHE/data/elbench`, no download |

In the default local mode the data must already be on disk, otherwise the loader
raises `FileNotFoundError`. Set `DATASET_SOURCE=HF` to download.

`datasets_info.py` registers `ms_id: ZeroLoss-Lab/ELBench`, but that dataset does
not exist on ModelScope at the time of writing, so `DATASET_SOURCE=ModelScope`
currently fails. HuggingFace is the only working source.

## How to run

Run everything that is wired (Safety + High-Level + General) with the aggregator
config:

```bash
DATASET_SOURCE=HF python run.py \
  --models <your_model_config> \
  --datasets elbench_gen \
  --judge-models <judge_model_config>
```

Or run a single module, e.g. only the safety LLM-judge subset:

```bash
DATASET_SOURCE=HF python run.py \
  --models <model> --datasets elbench_safety_judge --judge-models <judge>
```

The objective subsets (`elbench_highlevel_omni_gen`, `elbench_general_gen`) do
not need a judge model.

## Evaluation

- **LLM-judge subsets** (safety, high-level edu): a judge model scores each
  response. Safety uses a binary `[[1]]`/`[[0]]` rubric and reports the
  percentage of acceptable answers; high-level edu uses a 1–10 quality score
  reported on a 10–100 scale. A strong judge (GPT-4-class) is recommended.
- **Objective subsets** (omni, mmlu_pro, ceval, math, aime): exact-set match on
  extracted option letters, or boxed/normalized math-answer match.

## Reference

```
@misc{elbench,
  title  = {ELBench: A Multi-dimensional Benchmark for Education-facing Large Language Models},
  author = {ZeroLoss Lab},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/datasets/ZeroLoss-Lab/ELBench}}
}
```
