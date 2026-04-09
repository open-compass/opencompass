"""
Molecule pool loader for MolecularIQ.

Provides access to molecule pools for training and development.
Validation pools are intentionally hidden to prevent test data leakage.
"""

from typing import List, Optional

# HuggingFace dataset identifier
_HF_DATASET_ID = "ml-jku/moleculariq-trainPool"

# Pools that are available for loading
_AVAILABLE_POOLS = {"train"}

# Pools that are hidden to prevent data leakage
_HIDDEN_POOLS = {"val_easy", "val_hard", "test", "validation"}


class MoleculePoolHiddenError(Exception):
    """Raised when attempting to access a hidden molecule pool."""
    pass


def load_molecule_pool(
    pool_name: str,
    cache_dir: Optional[str] = None
) -> List[str]:
    """
    Load a molecule pool by name.

    This function provides access to molecule pools for training and
    development. Validation and test pools are intentionally hidden
    to prevent data leakage during model development.

    Args:
        pool_name: Name of the pool to load. Currently available: "train"
        cache_dir: Optional directory to cache downloaded data

    Returns:
        List of SMILES strings from the requested pool

    Raises:
        MoleculePoolHiddenError: If attempting to access a validation/test pool
        ValueError: If pool_name is not recognized

    Example:
        >>> from moleculariq_core import load_molecule_pool
        >>>
        >>> # Load training molecules
        >>> train_smiles = load_molecule_pool("train")
        >>> print(f"Loaded {len(train_smiles)} training molecules")
        >>>
        >>> # Validation pools are hidden
        >>> load_molecule_pool("val_hard")  # Raises MoleculePoolHiddenError
    """
    pool_name_lower = pool_name.lower().strip()

    if pool_name_lower in _HIDDEN_POOLS:
        raise MoleculePoolHiddenError(
            f"Molecule pool '{pool_name}' has been hidden to prevent test data leakage. "
            f"Only the training pool is available for development. "
            f"Use load_molecule_pool('train') instead."
        )

    if pool_name_lower not in _AVAILABLE_POOLS:
        available = ", ".join(sorted(_AVAILABLE_POOLS))
        raise ValueError(
            f"Unknown molecule pool '{pool_name}'. "
            f"Available pools: {available}"
        )

    return _load_from_huggingface(pool_name_lower, cache_dir)


def _load_from_huggingface(pool_name: str, cache_dir: Optional[str]) -> List[str]:
    """Load a pool from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required to load molecule pools. "
            "Install it with: pip install datasets"
        )

    # The train pool is uploaded as a single dataset (default "train" split)
    dataset = load_dataset(
        _HF_DATASET_ID,
        split="train",
        cache_dir=cache_dir
    )

    return list(dataset["smiles"])


def get_available_pools() -> List[str]:
    """
    Get list of available molecule pools.

    Returns:
        List of pool names that can be loaded
    """
    return sorted(_AVAILABLE_POOLS)
