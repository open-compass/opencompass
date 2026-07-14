"""
MolecularIQ Core
================

Core library for molecular reasoning: property computation, question generation,
and answer evaluation.

Quick Start
-----------

The easiest way to use MolecularIQ is through the MolecularIQD class::

    from moleculariq_core import MolecularIQD

    mqd = MolecularIQD(seed=42)

    # Generate a question
    question, answer, metadata = mqd.generate_count_question(
        smiles="CCO",
        count_properties="ring_count"
    )

    # Validate a prediction
    score = mqd.validate_count_answer("CCO", {"ring_count": 0})

Load molecule pools for training::

    from moleculariq_core import load_molecule_pool

    train_smiles = load_molecule_pool("train")

For lower-level access, use the primitives directly::

    from moleculariq_core import SymbolicSolver, evaluate_answer

    solver = SymbolicSolver()
    rings = solver.get_ring_count("CCO")

    score = evaluate_answer(
        task_type="single_count",
        predicted={"ring_count": 0},
        target={"ring_count": 0}
    )

Modules
-------
- MolecularIQD: High-level API for question generation and evaluation
- load_molecule_pool: Access training molecule pools
- SymbolicSolver: Low-level molecular property computation
- NaturalLanguageFormatter: Question/answer formatting
- evaluate_answer: Answer validation and scoring
"""

__version__ = "0.1.0"

# =============================================================================
# Primary API - What most users need
# =============================================================================

# High-level convenience class (recommended for most users)
from ._dynamic import MolecularIQD

# Molecule pool loading
from ._pools import load_molecule_pool, get_available_pools, MoleculePoolHiddenError

# Low-level primitives
from .solver import SymbolicSolver
from ._nlp import NaturalLanguageFormatter
from .questions import TASKS, SYSTEM_PROMPTS
from .rewards import chemical_reward, valid_smiles

# Friendly alias for chemical_reward
evaluate_answer = chemical_reward

# =============================================================================
# Secondary API - For advanced usage
# =============================================================================

# Additional solver classes (rarely needed directly)
from .solver import FunctionalGroupSolver, TemplateBasedReactionSolver

# NLP utilities
from ._nlp import (
    COUNT_MAPPINGS,
    INDEX_MAPPINGS,
    get_natural_language,
    parse_natural_language,
)

# Additional reward functions (use evaluate_answer for most cases)
from .rewards import (
    multi_count_dict_reward,
    single_count_reward,
    multi_index_identification_reward,
    single_index_reward,
    multi_constraint_generation_reward,
    constraint_reward,
    is_reasonable_molecule,
)

# Property definitions
from .properties import (
    COUNT_MAP,
    INDEX_MAP,
    CONSTRAINT_MAP,
    COUNT_TO_INDEX_MAP,
    KEY_ALIAS_MAP,
    get_alias,
    canonicalize_property_name,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Primary API - High-level
    'MolecularIQD',
    'load_molecule_pool',
    'get_available_pools',
    'MoleculePoolHiddenError',

    # Primary API - Low-level primitives
    'SymbolicSolver',
    'NaturalLanguageFormatter',
    'TASKS',
    'SYSTEM_PROMPTS',
    'evaluate_answer',
    'valid_smiles',

    # Secondary API - Solver
    'FunctionalGroupSolver',
    'TemplateBasedReactionSolver',

    # Secondary API - NLP
    'COUNT_MAPPINGS',
    'INDEX_MAPPINGS',
    'get_natural_language',
    'parse_natural_language',

    # Secondary API - Rewards
    'chemical_reward',
    'multi_count_dict_reward',
    'single_count_reward',
    'multi_index_identification_reward',
    'single_index_reward',
    'multi_constraint_generation_reward',
    'constraint_reward',
    'is_reasonable_molecule',

    # Secondary API - Properties
    'COUNT_MAP',
    'INDEX_MAP',
    'CONSTRAINT_MAP',
    'COUNT_TO_INDEX_MAP',
    'KEY_ALIAS_MAP',
    'get_alias',
    'canonicalize_property_name',
]
