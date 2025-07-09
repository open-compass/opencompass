
# Aider polyglot benchmark

## Prepare the dataset

We support the [Aider polyglot benchmark](https://aider.chat/docs/leaderboards/).   

You have to download our preprocessed dataset. The format of dir should be like:

```
aider
---Aider.json
```

The Aider.json is the preprocessed dataset used for score.

> **Note**: Currently, the supported version of Aider only supports **single-turn conversations**, meaning multi-turn dialogues are not yet supported. Additionally, it only supports the `whole` edit format and does not support incremental or diff-based formats.

## Run

We have provide the script for wildbench in `examples/eval_aider.py`.

## Acknowledgement

We greatly appreciate the authors of [Aider polyglot benchmark](https://github.com/Aider-AI/aider/tree/main).   If you find it is useful in your research, please consider cite them.