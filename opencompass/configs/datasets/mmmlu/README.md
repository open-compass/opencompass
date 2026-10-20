# MMMLU
## Dataset Description
Multilingual Massive Multitask Language Understanding (MMMLU)
The MMLU is a widely recognized benchmark of general knowledge attained by AI models. It covers a broad range of topics from 57 different categories, covering elementary-level knowledge up to advanced professional subjects like law, physics, history, and computer science.

We translated the MMLU’s test set into 14 languages using professional human translators. Relying on human translators for this evaluation increases confidence in the accuracy of the translations, especially for low-resource languages like Yoruba. We are publishing the professional human translations and the code we use to run the evaluations.

This effort reflects our commitment to improving the multilingual capabilities of AI models, ensuring they perform accurately across languages, particularly for underrepresented communities. By prioritizing high-quality translations, we aim to make AI technology more inclusive and effective for users worldwide.
MMMLU contains the MMLU test set translated into the following locales:

- AR_XY (Arabic)
- BN_BD (Bengali)
- DE_DE (German)
- ES_LA (Spanish)
- FR_FR (French)
- HI_IN (Hindi)
- ID_ID (Indonesian)
- IT_IT (Italian)
- JA_JP (Japanese)
- KO_KR (Korean)
- PT_BR (Brazilian Portuguese)
- SW_KE (Swahili)
- YO_NG (Yoruba)
- ZH_CH (Simplied Chinese)


## How to Use
Download file from [link](https://hf-mirror.com/datasets/openai/MMMLU)

```python
from datasets import load_dataset
ds = load_dataset("openai/MMMLU", "default")
from datasets import load_dataset
ds = load_dataset("openai/MMMLU", "by_language")
```