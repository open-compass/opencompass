# MedFailBench

## Introduction

MedFailBench is a clinician-authored medical AI safety benchmark. It checks
whether model answers preserve urgent triage language, avoid unsafe
reassurance, avoid remote dosing or treatment instructions, and keep
provenance/communication behaviour intact in high-risk clinical scenarios.

This adapter ships **two distinct data artefacts**. They are different files
with different schemas, different sizes, and different evaluation scopes. They
must not be treated as interchangeable, and neither one is a subset of the
other.

| Abbreviation | Dataset | Rows | Languages | Role |
| --- | --- | --- | --- | --- |
| `MedFailBench` | MedFailBench v0.2 prompt set | 222 | en (136), tr (86) | Backs the official MedFailBench leaderboard |
| `MedFailBench-TR-Preprint` | Turkish safety-layer evidence subset v0.1 | 44 | tr | Preprint evidence set; **not** the leaderboard data file |

Two clarifications that matter for how results should be read:

- **`MedFailBench` evaluates the leaderboard data file.** It loads the same
  `leaderboard/medfailbench_prompts_v0_2.jsonl` (222 rows) used by the LM
  Evaluation Harness task and the public leaderboard config. The data is the
  same; the OpenCompass judge is not the official scorer, so LLM-judge scores
  from this adapter are not itself an official leaderboard score.
- **`MedFailBench-TR-Preprint` uses a different file.** Its 44 rows come from
  `data/tr_medllm_synthetic_eval_set_v0_3.jsonl` and are exported for
  adapter-level transparency. The upstream README describes these cases as the
  preprint evidence set, which is cited separately from the broader benchmark
  assets.

Both datasets are synthetic only and are not intended for clinical use. Neither
contains patient data, and this adapter does not claim clinical validation,
model superiority, institutional endorsement, or deployment readiness.

Note on scoring: exact prompt text overlap between the two files is zero. Do not
report a `MedFailBench-TR-Preprint` score as a `MedFailBench` score, and do not
average the two.

## Official Links

- Project repository: https://github.com/goktugozkanmd/medical-ai-failure-atlas
- Method paper: https://arxiv.org/abs/2607.15166
- Software and data archive: https://doi.org/10.5281/zenodo.21205535
- OpenCompass tracking issue: https://github.com/open-compass/opencompass/issues/2516

## Data

Both files are pinned to upstream commit
`7c6a9939bf6db67e7abd95a383e5aec229c5770d`, which contains both artefacts.

### MedFailBench (v0.2, leaderboard)

- Default source: `leaderboard/medfailbench_prompts_v0_2.jsonl`
  (blob `6bf23969335aa1f7de06cf13ac7cdd07c34810d4`, 107,101 bytes)
- Size: 222 synthetic prompts (86 Turkish, 136 English)
- Schema: `id`, `prompt`, `language`, `domain`, `safety_focus`, `risk_axis`,
  `safe_answer_hint`, `category`; some rows also carry `drift_pair_id` and
  `drift_dimension`
- Categories: `turkish_safety` (36), `bilingual_drift` (100), `case_intake` (86)

### MedFailBench-TR-Preprint (v0.1)

- Default source: `adapters/opencompass/medfailbench_safety_layer_docs_v0_1.jsonl`
  (blob `a11eedacec07c10605ecc0ee94662d490af82082`, 58,613 bytes)
- Size: 44 synthetic prompts, all Turkish
- Schema: `id`, `question`, `target`, `clinical_domain`, `risk_axis`,
  `safety_gate`, `severity_1_to_5`, `metadata`

## Evaluation

Both configs send the dataset prompt directly to the model with
`RawPromptTemplate`, without adding extra clinical guidance around the source
prompt. The included OpenCompass evaluation uses an OpenCompass-native LLM judge
so the dataset can run inside the OpenCompass evaluation flow.

The official MedFailBench leaderboard scoring pipeline is rule-based: official
scores come from the scorer and rubric in the project repository, applied to
exported model responses. The judge included here is **not equivalent** to it,
so LLM-judge numbers from this adapter are not official leaderboard scores. To
report numbers that line up with the published leaderboard, score the exported
responses with that scorer instead of the LLM judge. The judged metric here also
is not an exact-match medical QA score.

Per dataset:

- `MedFailBench` compares the answer against `safe_answer_hint` and prompts with
  the row's own `prompt` field, which is already in that row's language
  (Turkish or English).
- `MedFailBench-TR-Preprint` compares the answer against `target` and prompts
  with the row's own `question` field, which is Turkish.

## Usage

```bash
# Leaderboard data file (222 rows, en + tr)
python run.py --datasets medfailbench_llmjudge_gen --models <your_model>

# 44-row Turkish preprint evidence subset
python run.py --datasets medfailbench_tr_preprint_llmjudge_gen --models <your_model>
```

The configuration keys are `MedFailBench` and `MedFailBench-TR-Preprint`
respectively.

## Citation

```bibtex
@misc{medfailbench2026,
  title = {MedFailBench: A Clinician-Built Open-Source Benchmark for Medical AI Safety Boundary Inspection},
  author = {Ozkan, Goktug},
  year = {2026},
  eprint = {2607.15166},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url = {https://arxiv.org/abs/2607.15166}
}
```
