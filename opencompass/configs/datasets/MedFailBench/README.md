# MedFailBench

## Introduction

MedFailBench is a small Turkish medical AI safety evaluation set. It contains
44 synthetic patient-facing prompts designed to check whether model answers
preserve urgent triage language, avoid unsafe reassurance, and avoid remote
dosing or treatment instructions in high-risk medical scenarios.

The dataset is synthetic only and is not intended for clinical use. It does not
contain patient data and this adapter does not claim clinical validation, model
superiority, institutional endorsement, or deployment readiness.

## Official Links

- Project repository: https://github.com/goktugozkanmd/medical-ai-failure-atlas
- OpenCompass tracking issue: https://github.com/open-compass/opencompass/issues/2516

## Data

- Language: Turkish
- Size: 44 synthetic prompts
- Schema: `id`, `question`, `target`, `clinical_domain`, `risk_axis`,
  `safety_gate`, `severity_1_to_5`, and `metadata`
- Default source:
  `adapters/opencompass/medfailbench_safety_layer_docs_v0_1.jsonl` in the
  MedFailBench project repository

## Evaluation

The OpenCompass config uses open-ended generation with an LLM judge. The judged
metric checks whether each model answer satisfies the target safety behavior for
the synthetic prompt; it is not an exact-match medical QA score.

## Usage

```bash
python run.py --datasets medfailbench_llmjudge_gen --models <your_model>
```

## Citation

```bibtex
@misc{medfailbench2026,
  title = {MedFailBench: Turkish Synthetic Medical AI Safety Failure Prompts},
  author = {Ozkan, Goktug},
  year = {2026},
  url = {https://github.com/goktugozkanmd/medical-ai-failure-atlas}
}
```
