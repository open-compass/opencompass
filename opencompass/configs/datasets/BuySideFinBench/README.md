# BuySideFinBench

> A bilingual benchmark evaluating LLMs on buy-side equity research and valuation tasks.

## Overview

BuySideFinBench evaluates large language models on the core analytical skills of a buy-side equity research analyst, organized as **6 subjects × 2 languages (Chinese & English) = 12 subsets**.

## Why Buy-Side?

Existing finance LLM benchmarks predominantly target sell-side / news-driven tasks (sentiment, summarization, headline interpretation) or surface-level knowledge (CFA-style multiple choice). BuySideFinBench targets the deeper analytical reasoning that distinguishes a buy-side analyst from a generalist financial reader.

| Subject | What it tests |
|---------|---------------|
| **Three-Statement Linkage** | Tracing cash impact across Income Statement / Balance Sheet / Cash Flow Statement |
| **DCF Valuation** | Discount rate logic, terminal value methodology, FCF projection |
| **Comparable Company Analysis** | Peer set construction, multiple selection, valuation reconciliation |
| **Financial Ratios** | Interpretation in industry context, not pure calculation |
| **Accounting Standards** | IFRS vs US GAAP distinctions (revenue recognition, leases, impairment) |
| **Sensitivity & Scenario Analysis** | Driver decomposition, two-way sensitivity tables, scenario weighting |

## Data Composition

- **Subjects**: 6 (listed above)
- **Languages**: Chinese (`zh`) and English (`en`)
- **Per subset**: 5 dev questions (for 5-shot in-context examples) + 10 test questions
- **Total**: 12 subsets, 180 evaluation instances

## Evaluation Protocol

Following the FinanceIQ pattern for direct comparability:

- **Prompting**: 5-shot using `FixKRetriever`
- **Inference**: `GenInferencer` (open-ended generation with parsed answer extraction)
- **Metric**: `AccEvaluator` (exact match after answer normalization)

## Data Source Methodology

All questions are derived from publicly available materials:
- Financial education textbooks and CFA / CICPA preparatory materials (paraphrased, not reproduced)
- Regulatory disclosure examples from SEC EDGAR, HKEXnews, and CSRC public filings
- Standard-setter publications (IFRS Foundation, FASB)
- Original analytical scenarios constructed from publicly known company financials

No proprietary research, paywalled databases, or licensed material is included. All financial figures used in scenarios are either from public filings or synthetically constructed.

## Usage

```bash
# Full benchmark (12 subsets)
python run.py --datasets BuySideFinBench_gen --models <your_model>

# Chinese subsets only
python run.py --datasets BuySideFinBench_zh_gen --models <your_model>

# Single subject across both languages (e.g., DCF Valuation)
python run.py --datasets BuySideFinBench_dcf_gen --models <your_model>
```

## File Structure

```
configs/datasets/BuySideFinBench/
├── README.md                         # This file
├── BuySideFinBench_gen.py            # Entry point (imports the sized variant)
└── BuySideFinBench_gen_5f8a1c.py     # Main config: 12 subsets, 5-shot, gen
```

Dataset hosting: [`huggingface.co/datasets/cindy90/BuySideFinBench`](https://huggingface.co/datasets/cindy90/BuySideFinBench)

## License

Released under the Apache License 2.0, consistent with the OpenCompass project.

## Citation

If you use BuySideFinBench in your research, please cite:

```bibtex
@misc{buysidefinbench2026,
  title  = {BuySideFinBench: A Bilingual Benchmark for Buy-Side Financial Analysis},
  author = {cindy90},
  year   = {2026},
  url    = {https://github.com/open-compass/opencompass}
}
```

## Acknowledgements

The dataset structure follows the convention established by FinanceIQ in OpenCompass. Thanks to the OpenCompass team for the evaluation infrastructure.
