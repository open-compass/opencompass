# Long-Context Evaluation

This page summarizes considerations for long-context evaluation. Search `configs/datasets` for the specific configurations of NeedleBench, RULER, LongBench, and similar datasets, and review their README and Summarizer.

The key to long-context evaluation is not `32k` or `128k` in a configuration filename, but the number of tokens actually received by the model, its truncation behavior, and the available output budget. For new configurations, prefer variants containing `rawprompt`.

## Length Budget

```text
final context = system + few-shot + question body + message history + generation prompt
available input limit ≈ max_seq_len - max_out_len
```

Different tokenizers produce different token counts for the same text. Inspect the final messages with the tokenizer of the model under evaluation; character count, file size, and the result from another model's tokenizer are estimates only.

## Common Failures

- The tokenizer or backend silently truncates the input.
- The inference service has a lower real context limit than the model configuration.
- `max_out_len` consumes too much of the context budget.
- Images, tool messages, or a chat template add extra tokens.
- Very long samples cause out-of-memory errors or request timeouts.
- Old predictions are incorrectly reused after the sharding configuration changes.

First inspect token counts and truncation with [Prompt Viewer](../prompt/debugging.md), then run small trials grouped by length range. Reports should include the tokenizer, input-token distribution and maximum, number of truncated samples, output budget, and actual backend limit.
