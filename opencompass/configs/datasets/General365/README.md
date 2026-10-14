# General365

This configuration loads the official public split directly from
`meituan-longcat/General365_Public` with `datasets.load_dataset`.

- `General365-math` evaluates `number` answers with
  `MATHVerifyEvaluator`, then sends rule failures to `GenericLLMEvaluator`.
- `General365-text` evaluates text and choice answers directly with
  `GenericLLMEvaluator`.

The LLM judge prompt follows the official project. To reproduce the paper,
configure GPT-4.1 as the judge through `OC_JUDGE_MODEL`, `OC_JUDGE_API_KEY`,
and optionally `OC_JUDGE_API_BASE`. The official repository currently defaults
to GPT-4.1-mini, while the paper reports GPT-4.1.
