"""
MolecularIQD - Dynamic question generation and answer validation.

Provides a convenient high-level API for generating molecular reasoning
questions and validating answers on-the-fly for any SMILES string.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

from ..solver import SymbolicSolver
from .._nlp import NaturalLanguageFormatter
from ..questions import TASKS, SYSTEM_PROMPTS
from ..properties import (
    COUNT_MAP,
    INDEX_MAP,
    CONSTRAINT_MAP,
    COUNT_TO_INDEX_MAP,
    get_alias,
    canonicalize_property_name,
)
from ..rewards import (
    multi_count_dict_reward,
    multi_index_identification_reward,
    multi_constraint_generation_reward,
)


class MolecularIQD:
    """
    Dynamic molecular reasoning question generator and evaluator.

    Provides a convenient API for generating questions and validating answers
    on-the-fly for any SMILES string. This is the recommended entry point for
    most users.

    Args:
        seed: Random seed for reproducible question generation
        enable_random_phrasing: Whether to vary question templates randomly
        cache_properties: Whether to cache computed properties for performance
        system_prompt_style: Style of system prompt ("with_key_hints" or "concise")

    Example:
        >>> from moleculariq_core import MolecularIQD
        >>>
        >>> mqd = MolecularIQD(seed=42)
        >>>
        >>> # Generate a count question
        >>> question, answer, metadata = mqd.generate_count_question(
        ...     smiles="CCO",
        ...     count_properties="ring_count"
        ... )
        >>> print(question)
        How many rings are in CCO? Return the result as a JSON object using the key `ring_count`.
        >>> print(answer)
        {'ring_count': 0}
        >>>
        >>> # Validate a prediction
        >>> score = mqd.validate_count_answer("CCO", {"ring_count": 0})
        >>> print(score)
        1.0
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        enable_random_phrasing: bool = True,
        cache_properties: bool = True,
        system_prompt_style: str = "with_key_hints"
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.solver = SymbolicSolver()
        self.formatter = NaturalLanguageFormatter(
            rng=self.rng,
            enable_random_phrasing=enable_random_phrasing
        )
        self.cache_properties = cache_properties
        self.system_prompt = SYSTEM_PROMPTS.get(
            system_prompt_style, SYSTEM_PROMPTS["with_key_hints"]
        )
        self._property_cache: Dict[Tuple[str, str], Any] = {}

    def _get_solver_method(self, property_name: str):
        """Get the solver method for a given property name."""
        if property_name.startswith("functional_group_"):
            def get_functional_group_property(smiles: str) -> Any:
                all_props = self.solver.functional_group_solver.get_counts_and_indices(smiles)
                return all_props.get(
                    property_name,
                    0 if "_count" in property_name or "_nbrInstances" in property_name else []
                )
            return get_functional_group_property

        if property_name.startswith("template_based_reaction_prediction_"):
            reaction_name = property_name.replace("template_based_reaction_prediction_", "")
            if "_success" in reaction_name:
                reaction_name = reaction_name.replace("_success", "")
                return lambda smiles: self.solver.reaction_solver.predict_reaction_success(
                    smiles, reaction_name
                )
            else:
                return lambda smiles: self.solver.reaction_solver.predict_reaction(
                    smiles, reaction_name
                )

        method_name = f"get_{property_name}"
        if hasattr(self.solver, method_name):
            return getattr(self.solver, method_name)

        if property_name.endswith('_index'):
            method_name = f"get_{property_name}ices"
            if hasattr(self.solver, method_name):
                return getattr(self.solver, method_name)

            plural_prop = property_name.replace('_index', '_indices')
            method_name = f"get_{plural_prop}"
            if hasattr(self.solver, method_name):
                return getattr(self.solver, method_name)

        base_name = property_name.replace('_count', '').replace('_index', '')
        method_name = f"get_{base_name}"
        if hasattr(self.solver, method_name):
            return getattr(self.solver, method_name)

        return None

    def compute_property(self, smiles: str, property_name: str) -> Any:
        """
        Compute a molecular property for the given SMILES string.

        Args:
            smiles: SMILES string of the molecule
            property_name: Name of the property (e.g., "ring_count", "carbon_atom_index")

        Returns:
            Computed property value (int for counts, list for indices)
        """
        cache_key = (smiles, property_name)
        if self.cache_properties and cache_key in self._property_cache:
            return self._property_cache[cache_key]

        solver_method = self._get_solver_method(property_name)
        if solver_method is None:
            return 0 if "_count" in property_name else []

        result = solver_method(smiles)

        if self.cache_properties:
            self._property_cache[cache_key] = result

        return result

    def compute_properties(self, smiles: str, property_names: List[str]) -> Dict[str, Any]:
        """
        Compute multiple properties for a molecule.

        Args:
            smiles: SMILES string
            property_names: List of property names

        Returns:
            Dictionary mapping property names to their values
        """
        return {prop: self.compute_property(smiles, prop) for prop in property_names}

    def generate_count_question(
        self,
        smiles: str,
        count_properties: Union[str, List[str]],
        template: Optional[str] = None,
        include_key_hint: bool = True
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        """
        Generate a count question for the given molecule.

        Args:
            smiles: SMILES string of the molecule
            count_properties: Property name(s) to ask about (e.g., "ring_count")
            template: Optional custom question template
            include_key_hint: Whether to append JSON key hints to the question

        Returns:
            Tuple of (question_string, ground_truth_dict, metadata_dict)

        Example:
            >>> mqd = MolecularIQD(seed=42)
            >>> q, a, m = mqd.generate_count_question("c1ccccc1", "aromatic_ring_count")
            >>> print(a)
            {'aromatic_ring_count': 1}
        """
        if isinstance(count_properties, str):
            count_properties = [count_properties]

        key_names = [get_alias(prop) for prop in count_properties]

        if template is None:
            task_type = "single_count" if len(count_properties) == 1 else "multi_count"
            template = self.rng.choice(TASKS[task_type]["question_templates"])

        count_types_natural = [
            self.formatter.technical_to_natural(prop, "count")
            for prop in count_properties
        ]
        count_types_str = (
            count_types_natural[0] if len(count_types_natural) == 1
            else ", ".join(count_types_natural)
        )

        question = template.format(
            smiles=smiles,
            count_type=count_types_str,
            count_types=count_types_str
        )

        if include_key_hint:
            hint = self.formatter.format_key_hint(key_names, "count")
            question = question.rstrip() + hint

        ground_truth = {
            key_name: self.compute_property(smiles, prop)
            for prop, key_name in zip(count_properties, key_names)
        }

        metadata = {
            "task_type": "single_count" if len(count_properties) == 1 else "multi_count",
            "smiles": smiles,
            "properties": count_properties,
            "key_names": key_names,
            "system_prompt": self.system_prompt
        }

        return question, ground_truth, metadata

    def generate_index_question(
        self,
        smiles: str,
        index_properties: Union[str, List[str]],
        template: Optional[str] = None,
        include_key_hint: bool = True
    ) -> Tuple[str, Dict[str, List[int]], Dict[str, Any]]:
        """
        Generate an index identification question for the given molecule.

        Args:
            smiles: SMILES string of the molecule
            index_properties: Property name(s) to ask about (e.g., "ring_index")
            template: Optional custom question template
            include_key_hint: Whether to append JSON key hints to the question

        Returns:
            Tuple of (question_string, ground_truth_dict, metadata_dict)

        Example:
            >>> mqd = MolecularIQD(seed=42)
            >>> q, a, m = mqd.generate_index_question("CCO", "carbon_atom_index")
            >>> print(a)
            {'carbon_atom_index': [0, 1]}
        """
        if isinstance(index_properties, str):
            index_properties = [index_properties]

        key_names = [get_alias(prop) for prop in index_properties]

        if template is None:
            task_type = (
                "single_index_identification"
                if len(index_properties) == 1
                else "multi_index_identification"
            )
            template = self.rng.choice(TASKS[task_type]["question_templates"])

        index_types_natural = [
            self.formatter.technical_to_natural(prop, "index")
            for prop in index_properties
        ]
        index_types_str = (
            index_types_natural[0] if len(index_types_natural) == 1
            else ", ".join(index_types_natural)
        )

        question = template.format(
            smiles=smiles,
            index_type=index_types_str,
            index_types=index_types_str
        )

        if include_key_hint:
            hint = self.formatter.format_key_hint(key_names, "index")
            question = question.rstrip() + hint

        ground_truth = {}
        for prop, key_name in zip(index_properties, key_names):
            value = self.compute_property(smiles, prop)
            if not isinstance(value, list):
                value = [value] if value is not None else []
            ground_truth[key_name] = value

        metadata = {
            "task_type": (
                "single_index_identification"
                if len(index_properties) == 1
                else "multi_index_identification"
            ),
            "smiles": smiles,
            "properties": index_properties,
            "key_names": key_names,
            "system_prompt": self.system_prompt
        }

        return question, ground_truth, metadata

    def generate_constraint_question(
        self,
        constraints: List[Dict[str, Any]],
        template: Optional[str] = None,
        include_key_hint: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a constraint-based molecule generation question.

        Args:
            constraints: List of constraint dicts, each with keys:
                - property: Property name (e.g., "ring_count")
                - operator: Comparison operator ("=", ">=", "<=", ">", "<")
                - value: Target value
            template: Optional custom question template
            include_key_hint: Whether to append JSON key hints

        Returns:
            Tuple of (question_string, metadata_dict)

        Example:
            >>> mqd = MolecularIQD(seed=42)
            >>> q, m = mqd.generate_constraint_question([
            ...     {"property": "ring_count", "operator": ">=", "value": 2}
            ... ])
        """
        formatted_constraints = []
        for constraint in constraints:
            formatted_constraint = {
                "type": constraint.get("property", constraint.get("type", "")),
                "operator": constraint.get("operator", "="),
                "value": constraint.get("value")
            }
            if "min_value" in constraint:
                formatted_constraint["min_value"] = constraint["min_value"]
            if "max_value" in constraint:
                formatted_constraint["max_value"] = constraint["max_value"]
            formatted_constraints.append(formatted_constraint)

        constraint_nl = self.formatter.format_constraints_list(formatted_constraints)

        if template is None:
            template = self.rng.choice(TASKS["constraint_generation"]["question_templates"])

        question = template.format(constraint=constraint_nl)

        if include_key_hint:
            hint = self.formatter.format_constraint_hint()
            question = question.rstrip() + hint

        metadata = {
            "task_type": "constraint_generation",
            "constraints": constraints,
            "constraint_natural_language": constraint_nl,
            "system_prompt": self.system_prompt
        }

        return question, metadata

    def validate_count_answer(
        self,
        smiles: str,
        predicted_answer: Dict[str, int],
        ground_truth: Optional[Dict[str, int]] = None,
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Validate a count answer against ground truth.

        Args:
            smiles: SMILES string of the molecule
            predicted_answer: Model's predicted answer dict
            ground_truth: Optional ground truth (computed automatically if not provided)
            return_details: If True, return detailed breakdown

        Returns:
            Score (1.0 = correct, 0.0 = incorrect), or details dict if requested
        """
        if ground_truth is None:
            property_names = [canonicalize_property_name(k) for k in predicted_answer.keys()]
            ground_truth = self.compute_properties(smiles, property_names)
            ground_truth = {
                k: ground_truth[canonicalize_property_name(k)]
                for k in predicted_answer.keys()
            }

        return multi_count_dict_reward(
            predicted_answer, ground_truth, return_details=return_details
        )

    def validate_index_answer(
        self,
        smiles: str,
        predicted_answer: Dict[str, List[int]],
        ground_truth: Optional[Dict[str, List[int]]] = None,
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Validate an index identification answer against ground truth.

        Args:
            smiles: SMILES string of the molecule
            predicted_answer: Model's predicted answer dict
            ground_truth: Optional ground truth (computed automatically if not provided)
            return_details: If True, return detailed breakdown

        Returns:
            Score (1.0 = correct, 0.0 = incorrect), or details dict if requested
        """
        if ground_truth is None:
            property_names = [canonicalize_property_name(k) for k in predicted_answer.keys()]
            ground_truth = self.compute_properties(smiles, property_names)
            ground_truth = {
                k: ground_truth[canonicalize_property_name(k)]
                for k in predicted_answer.keys()
            }

        return multi_index_identification_reward(
            predicted_answer, ground_truth, return_details=return_details
        )

    def validate_constraint_answer(
        self,
        predicted_smiles: str,
        constraints: List[Dict[str, Any]],
        return_details: bool = False
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """
        Validate a generated molecule against constraints.

        Args:
            predicted_smiles: Model's generated SMILES string
            constraints: List of constraints the molecule should satisfy
            return_details: If True, return detailed breakdown

        Returns:
            Score (1.0 = all constraints satisfied, 0.0 = invalid/failed),
            or (score, details_dict) if return_details=True
        """
        return multi_constraint_generation_reward(
            predicted_smiles,
            constraints,
            return_details=return_details
        )

    def generate_paired_question(
        self,
        smiles: str,
        count_property: str,
        template_count: Optional[str] = None,
        template_index: Optional[str] = None,
        include_key_hint: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate paired count and index questions for the same property.

        Useful for creating matched evaluation pairs where the count and index
        questions test the same underlying molecular feature.

        Args:
            smiles: SMILES string
            count_property: Count property name (e.g., "ring_count")
            template_count: Optional count question template
            template_index: Optional index question template
            include_key_hint: Whether to include key hints

        Returns:
            Tuple of (count_task_dict, index_task_dict), each containing
            'question', 'answer', and 'metadata' keys
        """
        index_property = COUNT_TO_INDEX_MAP.get(count_property)
        if index_property is None:
            index_property = count_property.replace("_count", "_index")

        count_q, count_a, count_m = self.generate_count_question(
            smiles, count_property, template_count, include_key_hint
        )
        index_q, index_a, index_m = self.generate_index_question(
            smiles, index_property, template_index, include_key_hint
        )

        return (
            {"question": count_q, "answer": count_a, "metadata": count_m},
            {"question": index_q, "answer": index_a, "metadata": index_m}
        )

    def get_available_count_properties(self) -> List[str]:
        """Get list of all available count properties."""
        return [prop for props in COUNT_MAP.values() for prop in props]

    def get_available_index_properties(self) -> List[str]:
        """Get list of all available index properties."""
        return [prop for props in INDEX_MAP.values() for prop in props]

    def get_available_constraint_properties(self) -> List[str]:
        """Get list of all properties usable in constraints."""
        return [prop for props in CONSTRAINT_MAP.values() for prop in props]

    def clear_cache(self):
        """Clear the property computation cache."""
        self._property_cache.clear()
