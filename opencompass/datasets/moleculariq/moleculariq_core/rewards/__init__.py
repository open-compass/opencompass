"""
Streamlined reward system for chemistry tasks.

This module provides reward functions for:
- Count tasks (single and multi)
- Index identification tasks (single and multi)
- Constraint generation tasks

Optionally, count tasks can return detailed per-property feedback by supplying
``return_details=True`` to ``chemical_reward`` (or the underlying count helpers).
"""

from typing import Any, Dict, Union

# Import main reward functions
from .count_reward import (
    multi_count_dict_reward,
    single_count_reward
)

from .index_reward import (
    multi_index_identification_reward,
    single_index_reward
)

from .constraint_reward import (
    multi_constraint_generation_reward,
    constraint_reward
)

# Import utilities
from .utils import (
    valid_smiles,
    is_reasonable_molecule,
    evaluate_numeric_constraint,
    parse_natural_language_property
)

def chemical_reward(
    task_type: str,
    predicted,
    target=None,
    constraints=None,
    *,
    return_details: bool = False,
    **kwargs
) -> Union[float, Dict[str, Any]]:
    """
    Unified dispatcher for chemical rewards based on task type.

    Args:
        task_type: Type of task - one of:
            - 'single_count': Single property counting
            - 'multi_count': Multiple property counting
            - 'single_index': Single index identification
            - 'multi_index': Multiple index identification
            - 'constraint_generation': Constraint-based molecule generation
            - 'multi_count_dict': Alias for multi_count
            - 'multi_index_identification': Alias for multi_index
            - 'multi_constraint_generation': Alias for constraint_generation
        predicted: Model prediction (format depends on task type)
        target: Ground truth (for count/index tasks)
        constraints: Constraint list (for generation tasks)
        return_details: When ``True`` the dispatcher forwards the flag to the
            underlying helper so callers can request detailed diagnostics.
            Count and index helpers expose per-property match reports, while
            constraint helpers return per-constraint status information.
        **kwargs: Additional arguments passed through to underlying helpers.

    Returns:
        Union[float, Dict[str, Any]]: Reward value (float) or detailed dictionary
        for count tasks when ``return_details`` is ``True``.
    """
    # Normalize task type
    task_type = task_type.lower().replace('-', '_')

    # Route to appropriate reward function
    if task_type in ['single_count']:
        if target is None:
            raise ValueError("Target required for count tasks")
        return single_count_reward(
            predicted,
            target,
            return_details=return_details
        )

    elif task_type in ['multi_count', 'multi_count_dict', 'multiple_count', 'count']:
        if target is None:
            raise ValueError("Target required for multi-count tasks")
        return multi_count_dict_reward(
            predicted,
            target,
            return_details=return_details
        )

    elif task_type in ['single_index', 'single_index_identification']:
        if target is None:
            raise ValueError("Target required for index tasks")
        return single_index_reward(
            predicted,
            target,
            return_details=return_details
        )

    elif task_type in ['multi_index', 'multi_index_identification', 'multiple_index', 'index']:
        if target is None:
            raise ValueError("Target required for multi-index tasks")
        return multi_index_identification_reward(
            predicted,
            target,
            return_details=return_details
        )

    elif task_type in ['constraint_generation', 'constraint', 'generation',
                        'multi_constraint_generation', 'multi_constraint']:
        # For constraint tasks, constraints can be in target or constraints param
        constraint_list = constraints if constraints is not None else target
        if constraint_list is None:
            raise ValueError("Constraints required for generation tasks")
        return multi_constraint_generation_reward(
            predicted,
            constraint_list,
            return_details=return_details,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown task type: {task_type}. Supported types: "
                        "single_count, multi_count, single_index, multi_index, constraint_generation")


# Define what's exported
__all__ = [
    # Main dispatcher
    'chemical_reward',

    # Count rewards
    'multi_count_dict_reward',
    'single_count_reward',

    # Index rewards
    'multi_index_identification_reward',
    'single_index_reward',

    # Constraint rewards
    'multi_constraint_generation_reward',
    'constraint_reward',

    # Utilities
    'valid_smiles',
    'is_reasonable_molecule',
    'evaluate_numeric_constraint',
    'parse_natural_language_property'
]
