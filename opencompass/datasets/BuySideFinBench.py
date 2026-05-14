#!/usr/bin/env python
"""Upload BuySideFinBench CSV data to HuggingFace Hub.

Prerequisites
-------------
    pip install huggingface_hub datasets pandas
    huggingface-cli login   # paste your HF write token when prompted

Run this script from the directory that contains
    data/BuySideFinBench/dev/*.csv
    data/BuySideFinBench/test/*.csv

Adjust LOCAL_DATA_DIR below if your path differs.
"""
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import create_repo

# ============ CONFIGURATION ============
HF_USERNAME = "cindy90"        # change if your HF username differs
DATASET_NAME = "BuySideFinBench"
LOCAL_DATA_DIR = Path("data/BuySideFinBench")  # adjust if needed

# The 12 subset names — must match CSV filenames (without .csv) and
# the subject mapping in BuySideFinBench_gen_5f8a1c.py
SUBSETS = [
    "three_statements_zh", "three_statements_en",
    "dcf_valuation_zh", "dcf_valuation_en",
    "comps_analysis_zh", "comps_analysis_en",
    "financial_ratios_zh", "financial_ratios_en",
    "accounting_standards_zh", "accounting_standards_en",
    "sensitivity_scenario_zh", "sensitivity_scenario_en",
]
# =======================================


def load_csv_as_dataset(csv_path: Path) -> Dataset:
    """Read a CSV and convert to a HuggingFace Dataset.

    Matches the loader logic:
      - skip the first column (id/index)
      - keep columns 1-6 as: question, A, B, C, D, answer
    """
    df = pd.read_csv(csv_path, header=0)
    if df.shape[1] != 7:
        raise ValueError(
            f"Expected 7 columns in {csv_path}, got {df.shape[1]}"
        )

    df_clean = df.iloc[:, 1:7].copy()
    df_clean.columns = ["question", "A", "B", "C", "D", "answer"]
    return Dataset.from_pandas(df_clean, preserve_index=False)


def main() -> int:
    repo_id = f"{HF_USERNAME}/{DATASET_NAME}"

    if not LOCAL_DATA_DIR.exists():
        print(f"ERROR: LOCAL_DATA_DIR does not exist: {LOCAL_DATA_DIR}")
        print("Edit LOCAL_DATA_DIR at the top of this script.")
        return 1

    print(f"Target HF repo: {repo_id}")
    print(f"Source data dir: {LOCAL_DATA_DIR.resolve()}")
    print()

    # 1. Create the dataset repo (idempotent)
    print(f"[1/2] Creating dataset repo (if not exists)...")
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    print(f"      OK: https://huggingface.co/datasets/{repo_id}")
    print()

    # 2. Push each subset as a separate config
    print(f"[2/2] Uploading {len(SUBSETS)} subsets as configs...")
    succeeded, skipped = 0, 0
    for subset in SUBSETS:
        print(f"  - {subset}")
        dataset_dict = DatasetDict()
        for split in ["dev", "test"]:
            csv_path = LOCAL_DATA_DIR / split / f"{subset}.csv"
            if not csv_path.exists():
                print(f"      WARNING: missing {csv_path}, skipping split")
                continue
            ds = load_csv_as_dataset(csv_path)
            dataset_dict[split] = ds
            print(f"      {split}: {len(ds)} rows")

        if len(dataset_dict) == 0:
            print(f"      SKIPPED: no splits loaded")
            skipped += 1
            continue

        dataset_dict.push_to_hub(repo_id, config_name=subset)
        succeeded += 1
        print(f"      pushed")

    print()
    print("=" * 50)
    print(f"Done. Pushed: {succeeded} / {len(SUBSETS)}  Skipped: {skipped}")
    print(f"Verify at: https://huggingface.co/datasets/{repo_id}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
