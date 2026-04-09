"""
Unified Natural Language Formatter for Chemistry Tasks
======================================================

Provides bidirectional conversion between technical chemistry terms
and natural language for counts, indices, and constraints.
"""

import re
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from .mappings import (
    COUNT_MAPPINGS,
    INDEX_MAPPINGS,
    FUNCTIONAL_GROUPS,
    ALIASES,
    get_natural_language,
    parse_natural_language
)


class NaturalLanguageFormatter:
    """Unified formatter for all chemistry natural language processing."""

    _COUNT_SINGLE_HINTS = [
        " Return the result as a JSON object using the key {keys}.",
        " Respond with a JSON object keyed by {keys}.",
        " Provide a JSON mapping whose field is {keys}.",
        " Give the answer as JSON with key {keys}."
    ]

    _COUNT_MULTI_HINTS = [
        " Return the result as a JSON object using these keys: {keys}.",
        " Respond with a JSON object containing fields {keys}.",
        " Provide a JSON mapping keyed by {keys}.",
        " Give the answer as JSON with keys {keys}."
    ]

    _INDEX_SINGLE_HINTS = [
        " Return the result as a JSON object mapping {keys} to its indices.",
        " Respond with a JSON mapping using key {keys} for the index list.",
        " Provide a JSON object where {keys} holds the indices.",
        " Give the indices as JSON under the key {keys}."
    ]

    _INDEX_MULTI_HINTS = [
        " Return the result as a JSON object mapping each key to its indices using {keys}.",
        " Respond with a JSON mapping where the keys {keys} contain the index lists.",
        " Provide a JSON object with index lists keyed by {keys}.",
        " Give the indices as JSON with keys {keys}."
    ]

    _CONSTRAINT_HINTS = [
        " Return the result as a JSON object using the key `smiles`.",
        " Respond with a JSON mapping whose key is `smiles`.",
        " Provide your answer as JSON with `smiles` as the key.",
        " Give the generated molecule in a JSON object keyed by `smiles`."
    ]

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        *,
        seed: Optional[int] = None,
        enable_random_phrasing: bool = True
    ):
        """Initialize the formatter with mappings and phrasing controls."""
        if rng is not None and seed is not None:
            raise ValueError("Provide either an RNG instance or a seed, not both.")

        if rng is None:
            rng = random.Random(seed)

        self.count_mappings = COUNT_MAPPINGS
        self.index_mappings = INDEX_MAPPINGS
        self.functional_groups = FUNCTIONAL_GROUPS
        self.aliases = ALIASES
        self._rng = rng
        self._use_random_phrasing = enable_random_phrasing

    # ========================================================================
    # COUNT METHODS
    # ========================================================================

    def format_count_query(
        self,
        smiles: str = None,
        count_types: List[str] = None,
        template: Optional[str] = None,
        return_parts: bool = False,
        include_key_hint: bool = False,
        key_names: Optional[List[str]] = None
    ) -> Union[str, Dict[str, str]]:
        """
        Format a count task question or return formatted parts.

        Args:
            smiles: SMILES string of the molecule (optional if return_parts=True)
            count_types: List of count types to ask about
            template: Optional question template with {smiles} and {count_types} placeholders
            return_parts: If True, return dict with formatted parts instead of full string

        Returns:
            If return_parts=True: Dict with 'count_types' key
            Otherwise: Formatted question string
        """
        count_types_list = self._ensure_type_list(count_types, "count_types")
        count_types_natural = self._format_multiple_types(count_types_list, "count")

        if return_parts:
            return {"count_types": count_types_natural}

        if template is None:
            if len(count_types_list) == 1:
                template = "How many {count_types} are present in {smiles}?"
            else:
                template = "For the molecule {smiles}, count the following features: {count_types}."

        question = template.format(smiles=smiles, count_types=count_types_natural)

        if len(count_types_list) == 1:
            question = self._remove_redundant_count_word(question, count_types_natural)

        if include_key_hint and not return_parts:
            hint_keys = key_names if key_names is not None else count_types_list
            question = question.rstrip() + self._format_key_hint(hint_keys, "count")

        return question

    def format_count_answer(self, counts: Dict[str, Optional[int]]) -> str:
        """
        Format count results as natural language.

        Args:
            counts: Dictionary mapping count_type -> count value (or None if absent)

        Returns:
            Natural language answer string
        """
        if not counts:
            return "No counts available"

        answers = []
        for count_type, count in counts.items():
            natural_name = self.technical_to_natural(count_type, "count")
            if count is None:
                answers.append(f"{natural_name}: none")
            else:
                answers.append(f"{natural_name}: {count}")

        return "; ".join(answers)

    def parse_count_answer(self, answer_text: str) -> Dict[str, Optional[int]]:
        """
        Parse a count answer from natural language.

        Args:
            answer_text: Answer in format "type1: count1; type2: count2" or JSON

        Returns:
            Dictionary mapping technical count types to counts (None if absent)
        """
        # Try JSON first
        if answer_text.strip().startswith('{'):
            try:
                data = json.loads(answer_text)
                if not isinstance(data, dict):
                    raise TypeError("Expected a JSON object for count answers")

                normalized = {}
                for key, value in data.items():
                    technical_key = self.natural_to_technical(key, "count")
                    try:
                        normalized[technical_key] = self._parse_optional_int(value)
                    except ValueError:
                        continue
                return normalized
            except (json.JSONDecodeError, TypeError):
                pass

        # Parse "type1: count1, type2: count2" format
        result = {}
        parts = re.split('[;,]', answer_text)

        for part in parts:
            part = part.strip()
            if ':' in part:
                count_type, count_str = part.split(':', 1)
                count_type = count_type.strip()
                count_str = count_str.strip()

                technical_type = self.natural_to_technical(count_type, "count")

                try:
                    count = self._parse_optional_int(count_str)
                except ValueError:
                    continue
                result[technical_type] = count

        return result

    # ========================================================================
    # INDEX METHODS
    # ========================================================================

    def format_index_query(
        self,
        smiles: str = None,
        index_types: List[str] = None,
        template: Optional[str] = None,
        return_parts: bool = False,
        include_key_hint: bool = False,
        key_names: Optional[List[str]] = None
    ) -> Union[str, Dict[str, str]]:
        """
        Format an index identification question or return formatted parts.

        Args:
            smiles: SMILES string of the molecule (optional if return_parts=True)
            index_types: List of index types to identify
            template: Optional question template
            return_parts: If True, return dict with formatted parts instead of full string

        Returns:
            If return_parts=True: Dict with 'index_types' key
            Otherwise: Formatted question string
        """
        index_types_list = self._ensure_type_list(index_types, "index_types")
        index_types_natural = self._format_multiple_types(index_types_list, "index")

        if return_parts:
            return {"index_types": index_types_natural}

        if template is None:
            if len(index_types_list) == 1:
                template = "What are the indices of {index_types} in {smiles}?"
            else:
                template = "For the molecule {smiles}, identify the atom indices for: {index_types}."

        question = template.format(smiles=smiles, index_types=index_types_natural)

        if include_key_hint and not return_parts:
            hint_keys = key_names if key_names is not None else index_types_list
            question = question.rstrip() + self._format_key_hint(hint_keys, "index")

        return question

    def format_index_answer(self, indices: Dict[str, List[int]]) -> str:
        """
        Format index results as natural language.

        Args:
            indices: Dictionary mapping index_type -> list of indices

        Returns:
            Natural language answer string
        """
        if not indices:
            return "No indices were requested."

        answers = []
        for index_type, idx_list in indices.items():
            natural_name = self.technical_to_natural(index_type, "index")

            if not idx_list:
                indices_str = "none"
            else:
                indices_str = ",".join(map(str, idx_list))

            answers.append(f"{natural_name}: {indices_str}")

        return "; ".join(answers)

    def parse_index_answer(
        self,
        answer_text: str,
        expected_types: Optional[List[str]] = None
    ) -> Dict[str, List[int]]:
        """
        Extract indices from natural language answer.

        Args:
            answer_text: Natural language answer text
            expected_types: List of expected index types (optional)

        Returns:
            Dictionary mapping index_type -> list of indices
        """
        result = {}

        # Handle structured format (key: values)
        segments = answer_text.split(';')

        for segment in segments:
            if ':' not in segment:
                continue

            parts = segment.split(':', 1)
            if len(parts) != 2:
                continue

            nl_key = parts[0].strip()
            indices_str = parts[1].strip()

            technical_key = self.natural_to_technical(nl_key, "index")

            if indices_str.lower() in ['none', 'empty', '[]', 'no atoms', 'no indices']:
                result[technical_key] = []
            else:
                # Extract all numbers
                if '-' in indices_str and ',' not in indices_str:
                    # Handle range notation
                    range_match = re.match(r'(\d+)\s*-\s*(\d+)', indices_str)
                    if range_match:
                        start, end = int(range_match.group(1)), int(range_match.group(2))
                        result[technical_key] = list(range(start, end + 1))
                    else:
                        numbers = re.findall(r'\d+', indices_str)
                        result[technical_key] = sorted([int(n) for n in numbers])
                else:
                    numbers = re.findall(r'\d+', indices_str)
                    result[technical_key] = sorted([int(n) for n in numbers])

        # Handle expected types not found
        if expected_types:
            for index_type in expected_types:
                if index_type not in result:
                    # Try to find in unstructured text
                    description = self.technical_to_natural(index_type, "index")
                    if description.lower() in answer_text.lower():
                        # Look for "no/zero" patterns
                        no_patterns = [
                            f"no {description}",
                            f"zero {description}",
                            f"There are no {description}",
                        ]
                        if any(p.lower() in answer_text.lower() for p in no_patterns):
                            result[index_type] = []
                    else:
                        result[index_type] = []

        return result

    # ========================================================================
    # CONSTRAINT METHODS
    # ========================================================================

    def format_constraint(
        self,
        constraint: Dict[str, Any],
        use_varied_phrasing: bool = True,
        return_only_text: bool = False
    ) -> str:
        """
        Format a constraint dictionary into natural language.

        Args:
            constraint: Dictionary with keys 'type', 'operator', 'value', etc.
            use_varied_phrasing: If True, randomly varies the phrasing
            return_only_text: If True, disables varied phrasing for deterministic text

        Returns:
            Natural language description of the constraint
        """
        if return_only_text:
            use_varied_phrasing = False

        constraint_type = constraint.get('type', '')
        operator = constraint.get('operator', '=')
        value = constraint.get('value')
        min_value = constraint.get('min_value')
        max_value = constraint.get('max_value')
        functional_group = constraint.get('functional_group')

        # Handle functional group constraints
        if 'functional_group' in constraint_type or functional_group:
            # Extract functional group name, handling both _nbrInstances and _count suffixes
            if constraint_type.endswith('_nbrInstances'):
                fg_name = constraint_type.replace('functional_group_', '').replace('_nbrInstances', '')
            elif constraint_type.endswith('_count'):
                fg_name = constraint_type.replace('functional_group_', '').replace('_count', '')
            else:
                fg_name = functional_group or constraint_type.replace('functional_group_', '')
            return self._format_functional_group_constraint(
                fg_name, operator, value, min_value, max_value, use_varied_phrasing
            )

        # Handle molecular formula
        if constraint_type == 'molecular_formula':
            return f"molecular formula {value}" if value else "any molecular formula"

        # Handle range operator
        if operator == 'range' and min_value is not None and max_value is not None:
            desc = self.technical_to_natural(constraint_type, "constraint")
            return f"between {min_value} and {max_value} {desc}"

        # Get the natural language description
        desc = self.technical_to_natural(constraint_type, "constraint")

        # Format based on operator
        return self._format_with_operator(desc, operator, value, use_varied_phrasing)

    def format_constraints_list(
        self,
        constraints: List[Dict[str, Any]],
        use_varied_connectors: bool = True,
        return_only_text: bool = False
    ) -> str:
        """
        Format a list of constraints into natural language.

        Args:
            constraints: List of constraint dictionaries
            use_varied_connectors: If True, varies the connecting words
            return_only_text: If True, disable varied phrasing/connector randomness

        Returns:
            Natural language description of all constraints
        """
        if not constraints:
            return "no constraints"

        if len(constraints) == 1:
            return self.format_constraint(
                constraints[0],
                use_varied_phrasing=self._use_random_phrasing and not return_only_text,
                return_only_text=return_only_text
            )

        allow_random_constraint = self._use_random_phrasing and not return_only_text
        formatted = [
            self.format_constraint(
                c,
                use_varied_phrasing=allow_random_constraint,
                return_only_text=return_only_text
            )
            for c in constraints
        ]

        allow_random_connectors = (
            use_varied_connectors and allow_random_constraint
        )

        if len(formatted) == 2:
            connector = self._pick([" and ", " with ", " plus "], allow_random_connectors)
            return connector.join(formatted)

        if allow_random_connectors:
            return self._pick([
                ", ".join(formatted[:-1]) + ", and " + formatted[-1],
                ", ".join(formatted[:-1]) + " and " + formatted[-1],
            ], allow_random_connectors)

        return ", ".join(formatted[:-1]) + ", and " + formatted[-1]

    def parse_constraint(self, nl_text: str) -> Dict[str, Any]:
        """
        Parse natural language into a constraint dictionary.

        Args:
            nl_text: Natural language constraint description

        Returns:
            Constraint dictionary with type, operator, value
        """
        # This is a simplified implementation - could be enhanced with NLP
        constraint = {}

        # Parse operators
        if "exactly" in nl_text or "precisely" in nl_text:
            constraint['operator'] = '='
        elif "at least" in nl_text or "minimum" in nl_text:
            constraint['operator'] = '>='
        elif "at most" in nl_text or "maximum" in nl_text:
            constraint['operator'] = '<='
        elif "more than" in nl_text or "greater than" in nl_text:
            constraint['operator'] = '>'
        elif "fewer than" in nl_text or "less than" in nl_text:
            constraint['operator'] = '<'
        elif "between" in nl_text and "and" in nl_text:
            constraint['operator'] = 'range'
        else:
            constraint['operator'] = '='

        # Extract numbers
        numbers = re.findall(r'\d+', nl_text)
        if numbers:
            if constraint['operator'] == 'range' and len(numbers) >= 2:
                constraint['min_value'] = int(numbers[0])
                constraint['max_value'] = int(numbers[1])
            else:
                constraint['value'] = int(numbers[0])

        # Try to identify the constraint type
        nl_lower = nl_text.lower()
        for natural, technical in self.aliases.items():
            if natural in nl_lower:
                constraint['type'] = technical
                break

        # Check for functional groups
        for fg_name in self.functional_groups.keys():
            if fg_name.replace('_', ' ') in nl_lower:
                constraint['type'] = 'functional_group_count'
                constraint['functional_group'] = fg_name
                break

        return constraint

    # ========================================================================
    # SHARED UTILITIES
    # ========================================================================

    def technical_to_natural(self, technical_name: str, task_type: str = "all") -> str:
        """
        Convert technical name to natural language.

        Args:
            technical_name: Technical name (e.g., 'hba', 'stereocenter')
            task_type: "count", "index", "constraint", or "all"

        Returns:
            Natural language description
        """
        # Check if it's a functional group with specific suffixes
        if technical_name.startswith('functional_group_'):
            if technical_name.endswith('_nbrInstances'):
                # For constraints - number of functional group instances
                fg_name = technical_name.replace('functional_group_', '').replace('_nbrInstances', '')
                fg_display = fg_name.replace('_', ' ')
                return f"{fg_display} groups"
            elif technical_name.endswith('_count'):
                # For count tasks - atom count
                fg_name = technical_name.replace('functional_group_', '').replace('_count', '')
                fg_display = fg_name.replace('_', ' ')
                return f"atoms in {fg_display} groups"
            elif technical_name.endswith('_index'):
                # For index tasks - atom positions
                fg_name = technical_name.replace('functional_group_', '').replace('_index', '')
                fg_display = fg_name.replace('_', ' ')
                return f"{fg_display} atom positions"
            else:
                # Generic functional group
                fg_name = technical_name.replace('functional_group_', '')
                fg_display = fg_name.replace('_', ' ')
                if task_type == "index":
                    return f"atoms in {fg_display} groups"
                elif task_type == "constraint":
                    return f"{fg_display} groups"
                else:
                    return f"{fg_display} groups"

        # Use get_natural_language function from mappings
        natural_forms = get_natural_language(technical_name, task_type)
        if natural_forms and natural_forms[0] != technical_name:
            return natural_forms[0]

        # Check functional groups directly
        if technical_name in self.functional_groups:
            base = self.functional_groups[technical_name]
            if task_type == "index":
                return f"atoms in {base}"
            return base

        # Fallback: replace underscores with spaces
        return technical_name.replace('_', ' ')

    def natural_to_technical(self, natural_text: str, task_type: str = "all") -> str:
        """
        Convert natural language to technical name.

        Args:
            natural_text: Natural language description
            task_type: "count", "index", "constraint", or "all"

        Returns:
            Technical name
        """
        normalized = " ".join(natural_text.strip().lower().split())

        # Use the parse_natural_language function from mappings
        result = parse_natural_language(normalized)
        if result != normalized:
            return result

        # Handle "atoms in X" pattern for indices
        if task_type == "index" and "atoms in" in normalized:
            parts = normalized.split("atoms in", 1)
            if len(parts) == 2:
                fg_part = parts[1].strip()
                # Remove plural 's' if present
                if fg_part.endswith('s') and not fg_part.endswith('ss'):
                    fg_part = fg_part[:-1]
                # Check if it's a functional group
                result = parse_natural_language(fg_part)
                if result != fg_part:
                    return result
                return f'functional_group_{fg_part.replace(" ", "_")}'

        # Try removing common suffixes
        for suffix in [' groups', ' group', ' atoms', ' atom', ' rings', ' ring']:
            if normalized.endswith(suffix):
                cleaned = normalized[:-len(suffix)].strip()
                result = parse_natural_language(cleaned)
                if result != cleaned:
                    return result

        # Fallback: replace spaces with underscores
        return normalized.replace(' ', '_').replace('-', '_')

    def _format_multiple_types(self, types: List[str], task_type: str) -> str:
        """
        Format a list of types into natural language.

        Args:
            types: List of technical type names
            task_type: "count" or "index"

        Returns:
            Natural language formatted list
        """
        if not types:
            return ""

        natural_names = [self.technical_to_natural(t, task_type) for t in types]

        if len(natural_names) == 1:
            return natural_names[0]
        elif len(natural_names) == 2:
            return f"{natural_names[0]} and {natural_names[1]}"
        else:
            return ", ".join(natural_names[:-1]) + f", and {natural_names[-1]}"

    def _format_key_hint(self, keys: List[str], task_type: str) -> str:
        """Format a key hint sentence appended to questions."""
        if not keys:
            return ""

        formatted_keys = [f"`{k}`" for k in keys]
        if len(formatted_keys) == 1:
            key_text = formatted_keys[0]
        elif len(formatted_keys) == 2:
            key_text = f"{formatted_keys[0]} and {formatted_keys[1]}"
        else:
            key_text = ", ".join(formatted_keys[:-1]) + f", and {formatted_keys[-1]}"

        allow_random = self._use_random_phrasing

        if task_type == "count":
            templates = self._COUNT_MULTI_HINTS if len(keys) > 1 else self._COUNT_SINGLE_HINTS
        elif task_type == "index":
            templates = self._INDEX_MULTI_HINTS if len(keys) > 1 else self._INDEX_SINGLE_HINTS
        else:
            templates = [" Return the result using these keys: {keys}."]

        template = self._pick(templates, allow_random)
        return template.format(keys=key_text)

    def format_key_hint(self, keys: List[str], task_type: str) -> str:
        """Public helper to format key hints."""
        return self._format_key_hint(list(keys), task_type)

    def format_constraint_hint(self) -> str:
        """Return a hint string for constraint tasks."""
        template = self._pick(self._CONSTRAINT_HINTS, self._use_random_phrasing)
        return template

    @staticmethod
    def _remove_redundant_count_word(text: str, natural_phrase: str) -> str:
        """Remove duplicated 'count' phrases like 'rings count'."""
        if not natural_phrase:
            return text

        phrase = natural_phrase.strip()
        if not phrase or phrase.lower().endswith('count'):
            return text

        pattern = re.compile(rf"{re.escape(phrase)}\s+count\b", re.IGNORECASE)
        cleaned = pattern.sub(phrase, text)
        return re.sub(r"\s{2,}", " ", cleaned)

    def _format_functional_group_constraint(
        self,
        fg_name: str,
        operator: str,
        value,
        min_value=None,
        max_value=None,
        use_varied_phrasing: bool = True
    ) -> str:
        """Format functional group constraints."""
        # Use the clean name directly, don't look up in functional_groups
        fg_base = fg_name.replace('_', ' ')

        # For functional group constraints, we're counting instances/groups
        # Determine if we need plural form
        needs_plural = (
            (operator == '=' and value != 1) or
            operator in ['>=', '>', '<', '<=', '!='] or
            operator == 'range'
        )

        if needs_plural:
            # Add "groups" suffix in plural
            fg_desc = f"{fg_base} groups"
        else:
            # Add "group" suffix in singular
            fg_desc = f"{fg_base} group"

        if operator == 'range':
            return f"between {min_value} and {max_value} {fg_desc}"

        return self._format_with_operator(fg_desc, operator, value, use_varied_phrasing)

    def _format_with_operator(
        self,
        desc: str,
        operator: str,
        value,
        use_varied_phrasing: bool = True
    ) -> str:
        """Format a constraint with an operator."""
        allow_random = use_varied_phrasing and self._use_random_phrasing
        if operator == '=':
            if value == 0:
                phrasings = [f"no {desc}", f"zero {desc}", f"without any {desc}"]
                return self._pick(phrasings, allow_random)
            elif value == 1:
                singular_desc = self._singularize(desc)
                phrasings = [
                    f"exactly 1 {singular_desc}",
                    f"precisely 1 {singular_desc}",
                    f"1 {singular_desc}"
                ]
                return self._pick(phrasings, allow_random)
            else:
                phrasings = [
                    f"exactly {value} {desc}",
                    f"precisely {value} {desc}",
                    f"{value} {desc}"
                ]
                return self._pick(phrasings, allow_random)

        elif operator == '>=':
            if value == 1:
                singular_desc = self._singularize(desc)
                return f"at least 1 {singular_desc}"
            return f"at least {value} {desc}"
        elif operator == '<=':
            if value == 1:
                singular_desc = self._singularize(desc)
                return f"at most 1 {singular_desc}"
            return f"at most {value} {desc}"
        elif operator == '>':
            if value == 1:
                singular_desc = self._singularize(desc)
                return f"more than 1 {singular_desc}"
            return f"more than {value} {desc}"
        elif operator == '<':
            if value == 1:
                singular_desc = self._singularize(desc)
                return f"fewer than 1 {singular_desc}"
            return f"fewer than {value} {desc}"
        elif operator == '!=':
            return f"not equal to {value} {desc}"
        else:
            return f"{desc} {operator} {value}"

    def _pluralize(self, text: str) -> str:
        """Simple pluralization logic."""
        if text.endswith('y') and text[-2] not in 'aeiou':
            return text[:-1] + 'ies'
        elif text.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return text + 'es'
        elif text.endswith('group'):
            return text + 's'
        else:
            return text + 's'

    def _singularize(self, text: str) -> str:
        """Simple singularization logic."""
        if text.endswith('ies'):
            return text[:-3] + 'y'
        elif text.endswith('es'):
            if text[:-2].endswith(('s', 'x', 'z', 'ch', 'sh')):
                return text[:-2]
            return text[:-1]
        elif text.endswith('s') and not text.endswith('ss'):
            return text[:-1]
        return text

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _pick(self, options: Sequence[str], allow_random: bool) -> str:
        """Pick a phrase optionally using randomness."""
        if not options:
            raise ValueError("Options must not be empty.")
        if not allow_random or len(options) == 1:
            return options[0]
        return self._rng.choice(list(options))

    def _ensure_type_list(
        self,
        values: Union[str, Iterable[str], None],
        argument_name: str
    ) -> List[str]:
        """Ensure that an input is a non-empty iterable of strings."""
        if values is None:
            raise ValueError(f"{argument_name} must contain at least one entry.")

        if isinstance(values, str):
            normalized = [values]
        else:
            try:
                normalized = list(values)
            except TypeError as exc:  # pragma: no cover - defensive programming
                raise TypeError(f"{argument_name} must be an iterable of strings.") from exc

        if not normalized:
            raise ValueError(f"{argument_name} must contain at least one entry.")

        if not all(isinstance(item, str) for item in normalized):
            raise TypeError(f"All values in {argument_name} must be strings.")

        return normalized

    def _parse_optional_int(self, raw_value: Any) -> Optional[int]:
        """Parse an optional integer value, allowing textual 'none' markers."""
        if raw_value is None:
            return None

        if isinstance(raw_value, bool):
            raise ValueError("Boolean values are not valid counts.")

        if isinstance(raw_value, int):
            return raw_value

        if isinstance(raw_value, float):
            if raw_value.is_integer():
                return int(raw_value)
            raise ValueError("Non-integer floats are not valid counts.")

        text_value = str(raw_value).strip()

        if not text_value:
            return None

        lowered = text_value.lower()
        if lowered in {"none", "null", "n/a", "na", "nan"}:
            return None

        return int(text_value)

    # ========================================================================
    # SIMPLIFIED TEXT-ONLY METHODS FOR TEMPLATES
    # ========================================================================

    def get_count_text(self, count_types: List[str]) -> str:
        """
        Get just the natural language text for count types.
        This is what would go in the {count_types} placeholder.

        Args:
            count_types: List of technical count type names

        Returns:
            Natural language text for the count types
        """
        return self._format_multiple_types(count_types, "count")

    def get_index_text(self, index_types: List[str]) -> str:
        """
        Get just the natural language text for index types.
        This is what would go in the {index_types} placeholder.

        Args:
            index_types: List of technical index type names

        Returns:
            Natural language text for the index types
        """
        return self._format_multiple_types(index_types, "index")

    def get_constraint_text(self, constraint: Dict[str, Any]) -> str:
        """
        Get just the natural language text for a constraint.
        This is what would go in the {constraint} placeholder.

        Args:
            constraint: Constraint dictionary

        Returns:
            Natural language text for the constraint
        """
        return self.format_constraint(constraint, use_varied_phrasing=True)


# Convenience functions for backward compatibility and simple usage
def get_count_text(count_types: List[str]) -> str:
    """Get natural language text for count types."""
    formatter = NaturalLanguageFormatter()
    return formatter.get_count_text(count_types)


def get_index_text(index_types: List[str]) -> str:
    """Get natural language text for index types."""
    formatter = NaturalLanguageFormatter()
    return formatter.get_index_text(index_types)


def get_constraint_text(constraint: Dict[str, Any]) -> str:
    """Get natural language text for a constraint."""
    formatter = NaturalLanguageFormatter()
    return formatter.get_constraint_text(constraint)


def format_count_query(
    smiles: str,
    count_types: List[str],
    template: Optional[str] = None,
    *,
    include_key_hint: bool = False,
    key_names: Optional[List[str]] = None
) -> str:
    """Format a complete count task question."""
    formatter = NaturalLanguageFormatter()
    return formatter.format_count_query(
        smiles,
        count_types,
        template,
        include_key_hint=include_key_hint,
        key_names=key_names
    )


def format_index_query(
    smiles: str,
    index_types: List[str],
    template: Optional[str] = None,
    *,
    include_key_hint: bool = False,
    key_names: Optional[List[str]] = None
) -> str:
    """Format a complete index identification question."""
    formatter = NaturalLanguageFormatter()
    return formatter.format_index_query(
        smiles,
        index_types,
        template,
        include_key_hint=include_key_hint,
        key_names=key_names
    )


def format_constraint(constraint: Dict[str, Any], use_varied_phrasing: bool = True) -> str:
    """Format a constraint dictionary into natural language."""
    formatter = NaturalLanguageFormatter()
    return formatter.format_constraint(constraint, use_varied_phrasing)
