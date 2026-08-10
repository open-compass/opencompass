# C4 Bench

C4 Bench evaluates multimodal cross-concept creativity through Chinese
chengyu. The primary configuration contains H0, H1, H4, and E0 (884 task
rows); E1 is available through the task-specific configurations and is not
included in the primary score.

The loader reads task rows and images from
[`sci-m-wang/C4-Eval`](https://huggingface.co/datasets/sci-m-wang/C4-Eval).
Use an image-capable API model backend, such as `LiteLLMAPI` configured for a
vision-language model. The dataset configuration does not override context or
output length settings.

```python
from opencompass.configs.datasets.c4_bench.c4_bench_gen import c4_bench_datasets

datasets = [*c4_bench_datasets]
```

## Citation

```bibtex
@misc{wang2026mllmsdecodecreativeleap,
      title={Can MLLMs Decode the Creative Leap? Introducing C4 for Cross-Concept Understanding},
      author={Ming Wang and Yuqing Zhang and Tingna Xie and Xiangju Li and
              Xiaocui Yang and Daling Wang and Shi Feng and Yifei Zhang},
      year={2026},
      eprint={2608.06501},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.06501},
}
```
