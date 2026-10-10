# Prompt Preview and Debugging

Do not run a complete benchmark immediately after changing a prompt. First inspect the input the model will receive, then validate generation and answer extraction on a small sample.

## Prompt Viewer

```bash
python tools/prompt_viewer.py my_eval.py -n -c 3
```

- `-n`: use non-interactive mode and select the first model and dataset;
- `-c`: set the number of samples to print;
- `-a`: inspect every model-dataset combination;
- `-p PATTERN`: select only datasets whose abbreviations match the pattern.

When given a complete experiment configuration, the tool constructs the tokenizer, displays input after model-template processing, and reports the token count. When given only a dataset configuration, it can normally inspect only the dataset-side prompt.

## Export Messages Only

If the evaluation backend or template is unsuitable for Prompt Viewer, use `--dump-only-message-path` to export the constructed inputs. This argument currently supports only dataset configurations that use `GenInferencer`. Specify `--mode infer` as well to avoid proceeding to evaluation:

```bash
opencompass my_eval.py \
    --mode infer \
    --dump-only-message-path /opencompass-messages \
    --debug
```

The export directory is organized by model and dataset `abbr`:

```text
/opencompass-messages/
└── gpt-6-astra-response/
    └── demo_gsm8k.jsonl
```

Each line of the JSONL file corresponds to one sample:

```json
{"message": [{"role": "system", "content": "Answer concisely."}, {"role": "user", "content": "What is 1 + 1?"}], "gold": "2"}
```

`message` is the result after the dataset template performs field substitution and few-shot composition and after the model `meta_template` is applied. `gold` is the unprocessed reference answer. Export occurs before `model.generate()`, so it does not apply the model's internal tokenizer chat template, tokenization, or API request conversion, and no model generation is performed. The task still initializes the model object and does not produce formal `predictions/` files that can be scored or reused.
