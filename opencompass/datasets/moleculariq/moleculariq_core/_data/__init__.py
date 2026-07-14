"""
Data files for moleculariq_core.

This module provides paths to data files used by the solver and NLP modules.

Files:
- smarts_functional_groups.txt: SMARTS patterns for functional group detection
- smarts_renamed.txt: Human-readable functional group names
- reaction_templates.txt: Reaction SMIRKS patterns and descriptions
"""

from pathlib import Path

# Directory containing data files
DATA_DIR = Path(__file__).parent


def get_data_path(filename: str) -> Path:
    """
    Get the full path to a data file.

    Args:
        filename: Name of the data file

    Returns:
        Path to the data file

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path


# Convenience constants for common files
SMARTS_FUNCTIONAL_GROUPS = DATA_DIR / "smarts_functional_groups.txt"
SMARTS_RENAMED = DATA_DIR / "smarts_renamed.txt"
REACTION_TEMPLATES = DATA_DIR / "reaction_templates.txt"
