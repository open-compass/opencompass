# Analyzing Repetition and Response Length

Adding `--dump-res-length` during inference writes response-length information into predictions. Adding `--analysis-repeat` during summarization analyzes abnormal repetition patterns:

```bash
opencompass my_eval.py --dump-res-length --analysis-repeat
```

To analyze existing results, use:

```bash
python tools/analyze_repeat.py outputs/my_eval/<timestamp> \
    --model model-abbr \
    --tokenizer gpt-4o
```

Use `--think-tag` to analyze reasoning content and final replies separately, and `--out` to choose an output file. When a Hugging Face tokenizer is supplied, the tool may need network access to load it; in an offline environment, pass a local tokenizer path.

Response length helps reveal truncation, empty replies, and anomalous cost, but it is not the same as input-token statistics. For long-context input checks, see [Long-Context Evaluation](../faq/long_context.md).
