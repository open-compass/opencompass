# Ruler

Ruler
We have providied two types of evaluation demo for using different tokenizers. For your own evaluation, you can create a new evaluation script following the example and change the context window sizes or add models according to your settings.

```bash
python run.py configs/eval_ruler_fix_tokenizer.py # For evaluation with GPT-4 tokenizer
python run.py configs/eval_ruler.py # For evaluation with model's tokenizer
```
