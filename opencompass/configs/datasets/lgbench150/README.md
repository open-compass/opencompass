# LG-Bench 150

This configuration evaluates the bundled 150-question subset of LG-Bench.
It reports Acc, GCS, and KBS for the full induced graph and separately for
medicine, biology, and chemistry. The subset contains 50 questions per
domain. Its scores are not comparable to results on the complete 10,200-question
benchmark.

The graph is JSON Lines with one edge per line. Overall metrics include
cross-domain edges; each domain metric uses only edges whose endpoints both
belong to that domain. Edges with weight less than or equal to 0.4 are excluded.

To use the dataset, import `lgbench150_datasets` in an evaluation config and
pair it with the model configuration you want to evaluate. The model should be
prompted to return a single answer letter (A–D). Invalid or ambiguous outputs
are counted as incorrect.

Paper: Lu Sun et al., [LG-Bench: A Graph-Structured Evaluation Benchmark for
Life Sciences](https://arxiv.org/abs/2610.26787).

The included question and graph files are the 150-question subset and induced
graph prepared for this integration. Cite the paper when reporting results.
