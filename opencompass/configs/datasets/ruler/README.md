# Ruler
OpenCompass now supports the brand new long-context language model evaluation benchmark — [RULER](https://arxiv.org/pdf/2404.06654). RULER provides an evaluation of long-context including retrieval, multi-hop tracing, aggregation, and question answering through flexible configurations.

OpenCompass have providied two types of evaluation demo for using different tokenizers.

For using the same tokenizer (typicall GPT-4), you can follow the demo (examples/eval_ruler_fix_tokenizer.py) where most of the settings are already defined.


For evaluation using each model's own tokenizer, you have to build the settings when you run the demo (we do not know which model you are trying to evaluate!) you can create a new evaluation script following the example (examples/eval_ruler.py) and change the context window sizes or add models according to your settings.

```bash
python run.py examples/eval_ruler_fix_tokenizer.py # For evaluation with GPT-4 tokenizer
python run.py examples/eval_ruler.py # For evaluation with model's tokenizer
```
