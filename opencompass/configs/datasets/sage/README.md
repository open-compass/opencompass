## SAGE: Scientific Advanced General Evaluation

### Usage

```python
with read_base():
    from opencompass.configs.datasets.sage.sage_gen import sage_datasets

# update your judge model information
sage_datasets[0]["eval_cfg"]["evaluator"]["judge_cfg"]["judgers"][0].update(dict(
    abbr="YOUR_ABBR",
    openai_api_base="YOUR_URL",
    path="YOUR_PATH",
    key="YOUR_API_KEY",
))
```

#### Test split

```python
with read_base():
    from opencompass.configs.datasets.sage.sage_gen import sage_datasets

sage_datasets[0]["abbr"] = "sage-test"
sage_datasets[0]["split"] = "test"
```

> The `test` split is only supported for infer, which means you should set `-m infer` for oc command.

### Performance

#### OpenAI o4-mini as Judge

| DeepSeek-R1-0528 | Gemini-2.5-Pro | Grok-4 |
| ----------- | ----------- |  ----------- |
| 25.8 | 34.9 | 32.9 |