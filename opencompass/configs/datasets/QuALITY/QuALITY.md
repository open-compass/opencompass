# QuALITY
## Introduction
The following introduction comes from the description in [QuALITY Leaderboard](https://nyu-mll.github.io/quality/)

```
QuALITY is a multiple-choice question answering dataset with context passages in English that have an average length of about 5,000 tokens.
```

These questions were categorized into two levels: easy and hard.

## Official link

### Paper

[QuALITY: Question Answering with Long Input Texts, Yes!](https://arxiv.org/pdf/2112.08608.pdf)

### Repository

[nyu-mll/quality](https://github.com/nyu-mll/quality)


## Evaluation results

```
dataset    version    metric    mode      qwen1.5-7b-chat-hf    qwen1.5-14b-chat-hf    qwen1.5-72b-chat-hf
---------  ---------  --------  ------  --------------------  ---------------------  ---------------------
QuALITY    ed2404     easy_acc  gen                    62.39                  68.17                  76.69
QuALITY    ed2404     hard_acc  gen                    49.27                  56.22                  63.96
QuALITY    ed2404     all_acc   gen                    54.65                  60.88                  68.84
```

## Reference
```
@inproceedings{pang-etal-2022-quality,
    title = "{Q}u{ALITY}: Question Answering with Long Input Texts, Yes!",
    author = "Pang, Richard Yuanzhe  and
      Parrish, Alicia  and
      Joshi, Nitish  and
      Nangia, Nikita  and
      Phang, Jason  and
      Chen, Angelica  and
      Padmakumar, Vishakh  and
      Ma, Johnny  and
      Thompson, Jana  and
      He, He  and
      Bowman, Samuel",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.391",
    pages = "5336--5358",
    abstract = "To enable building and testing models on long-document comprehension, we introduce QuALITY, a multiple-choice QA dataset with context passages in English that have an average length of about 5,000 tokens, much longer than typical current models can process. Unlike in prior work with passages, our questions are written and validated by contributors who have read the entire passage, rather than relying on summaries or excerpts. In addition, only half of the questions are answerable by annotators working under tight time constraints, indicating that skimming and simple search are not enough to consistently perform well. Our baseline models perform poorly on this task (55.4{\%}) and significantly lag behind human performance (93.5{\%}).",
}
```
