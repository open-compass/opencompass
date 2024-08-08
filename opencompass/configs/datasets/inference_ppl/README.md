# Inference-PPL Datasets

- **Description**: Compute the loss only on the labeled positions, especially used for reasoning corpus.
- **Datasets**: cn-reasoning-val.jsonl (example datasets, inference-ppl can be generalized to more corpus).

# PPL Computation

$$ \text{ppl} = - \frac{1}{n} \sum_{i=0}^n \sum_{c=0}^{vocab\_size} y_{i,c} \log p_{i,c} \tag{1} $$

where Eq. (1) is the normal mean ppl computation formula, for inference-ppl, we only compute the average score based on pre-labeled position.

# Quick Start

```shell
cd opencompass
python run.py configs/eval_inference_ppl.py
```

# Some results

| Model      | Result |
| ----------- | ----------- |
| Qwen1.5-7b    | 0.59       |
| Qwen1.5-14b   | 0.54        |
| Llama2-7b | 0.49 |
| Llama2-13b | 0.43 |
