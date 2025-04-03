import os
import pandas as pd
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset

@LOAD_DATASET.register_module()
class WMT19TranslationDataset(BaseDataset):
    @staticmethod
    def load(path: str, src_lang: str, tgt_lang: str):
        print(f"Attempting to load data from path: {path}")
        print(f"Source language: {src_lang}, Target language: {tgt_lang}")

        lang_pair_dir = os.path.join(path, f"{src_lang}-{tgt_lang}")
        if not os.path.exists(lang_pair_dir):
            lang_pair_dir = os.path.join(path, f"{tgt_lang}-{src_lang}")
            if not os.path.exists(lang_pair_dir):
                raise ValueError(f"Cannot find directory for language pair {src_lang}-{tgt_lang} or {tgt_lang}-{src_lang}")

        print(f"Loading data from directory: {lang_pair_dir}")

        val_file = os.path.join(lang_pair_dir, "validation-00000-of-00001.parquet")
        val_df = pd.read_parquet(val_file)

        def process_split(df):
            return Dataset.from_dict({
                'input': df['translation'].apply(lambda x: x[src_lang]).tolist(),
                'target': df['translation'].apply(lambda x: x[tgt_lang]).tolist()
            })

        return DatasetDict({
            'validation': process_split(val_df)
        })

    @classmethod
    def get_dataset(cls, path, src_lang, tgt_lang):
        return cls.load(path, src_lang, tgt_lang)

