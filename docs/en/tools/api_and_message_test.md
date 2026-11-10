# Testing APIs and Message Formats

## API Model Tester

```bash
python tools/test_api_model.py path/to/model_config.py -n
```

This tool builds an API model and sends its built-in test prompt to validate configuration parsing, authentication, message protocol, and basic generation. It does not replace inspection of the prompt from the actual Dataset, nor does it prove high-concurrency stability under rate limiting.

## ChatML Format Check

The repository's `tools/chatml_format_test.py` checks the relevant message format. Inspect its current arguments before running it:

```bash
python tools/chatml_format_test.py --help
```

## Recommended Test Order

1. Validate the endpoint, model name, and key with one request.
2. Inspect the actual Dataset input with [Prompt Viewer](../prompt/debugging.md).
3. Run inference and evaluation end to end on a small sample.
4. Gradually increase the internal `max_workers` and Runner concurrency.
5. Record 429 responses, timeouts, empty replies, retry counts, and actual cost.

Provide keys through environment variables or a controlled secrets file. Do not write them to test output or Git configuration.
