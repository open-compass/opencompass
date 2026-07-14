## ATLAS: A High-Difficulty, Multidisciplinary Benchmark for Frontier Scientific Reasoning

### Usage

```python
with read_base():
    from opencompass.configs.datasets.atlas.atlas_gen import atlas_datasets

# update your judge model information
atlas_datasets[0]["eval_cfg"]["evaluator"]["judge_cfg"]["judgers"][0].update(dict(
    abbr="YOUR_ABBR",
    openai_api_base="YOUR_URL",
    path="YOUR_PATH",
    key="YOUR_API_KEY",
    # tokenizer_path="o3",  # Optional: update if using a different model
))
```

#### Test split

```python
with read_base():
    from opencompass.configs.datasets.atlas.atlas_gen import atlas_datasets

# default is val split, if you want to test on test split, uncomment following lines

# atlas_datasets[0]["abbr"] = "atlas-test" 
# atlas_datasets[0]["split"] = "test"
# atlas_datasets[0]["eval_cfg"]["evaluator"]["dataset_cfg"]["abbr"] = "atlas-test"
# atlas_datasets[0]["eval_cfg"]["evaluator"]["dataset_cfg"]["split"] = "test"

```

> The `test` split is only supported for infer, which means you should set `-m infer` for oc command.

### Performance

#### OpenAI o4-mini as Judge

| DeepSeek-R1-0528 | Gemini-2.5-Pro | Grok-4 |
| ----------- | ----------- |  ----------- |
| 25.8 | 34.9 | 32.9 |