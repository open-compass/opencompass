"""
Natural Language Processing for Chemistry Tasks
===============================================

This module provides unified natural language formatting and parsing
for chemistry-related tasks including constraints, counts, and indices.
"""

from .formatter import NaturalLanguageFormatter
from .mappings import (
    COUNT_MAPPINGS,
    INDEX_MAPPINGS,
    FUNCTIONAL_GROUPS,
    REACTION_TEMPLATES,
    ALIASES,
    get_natural_language,
    parse_natural_language,
    get_all_properties
)

__all__ = [
    'NaturalLanguageFormatter',
    'COUNT_MAPPINGS',
    'INDEX_MAPPINGS',
    'FUNCTIONAL_GROUPS',
    'REACTION_TEMPLATES',
    'ALIASES',
    'get_natural_language',
    'parse_natural_language',
    'get_all_properties'
]
