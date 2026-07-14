# SEED — Quick Start

**SEED (Scalable Expression Edit Distance)** is a extended version of previous metric [Expression Edit Distance (EED)](https://arxiv.org/pdf/2504.16074).

**SEED** can evaluate LaTeX math pairs (GT vs. prediction) with a **0–100** score and distance metrics. Supports Expression, Equation, Tuple, Interval, and Numeric (with unit conversion).

> Full examples: see `SEED/test.py`.

## Install

```bash
pip install sympy numpy latex2sympy2_extended timeout_decorator pint
```

## Minimal Use

```python
from SEED.SEED import SEED
gt   = r"2 m g + 4\frac{m v_0^2}{l}"
pred = r"2 m g + 2\frac{m v_0^2}{l}"
score, rel_dist, tree_size, dist = SEED(gt, pred, "Expression")
```

**Signature**

```python
SEED(answer_latex, test_latex, t, debug_mode=False)
-> (score, relative_distance, answer_tree_size, distance)
```

**Types**: `"Expression" | "Equation" | "Tuple" | "Interval" | "Numeric"`

**Returns**: For `"Numeric"`, only `score` is meaningful; others are `-1`.

## Run Examples

```bash
python -m SEED.test
```

## Notes

- `timeout_decorator` isn’t supported on Windows (use `threading`/`multiprocessing`).
- `\int` / `\sum` currently return 0 (not evaluated).

## Acknowledgments

**Scalable Expression Edit Distance (SEED)** is inspired by `Expression Edit Distance (EED)` metric from **[PHYBench](https://www.phybench.cn/)**, which introduced Edit Distance to evaluating symbolic reasoning in physics. We extend and modify this idea by proposing the , supporting more diverse answer types and providing finer-grained evaluation dedicated for the fields of Condensed Matter Physics.

We sincerely thank the PHYBench team for their open-source contribution. Their code is released under the [MIT license](https://github.com/phybench-official/phybench?tab=MIT-1-ov-file#readme) and is available at [https://github.com/phybench-official/phybench](https://github.com/phybench-official/phybench).
