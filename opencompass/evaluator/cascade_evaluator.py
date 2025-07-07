import os
from typing import Any, Callable, Dict, List, Optional

import mmengine
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils.logging import get_logger


@ICL_EVALUATORS.register_module()
class CascadeEvaluator(BaseEvaluator):
    """Cascade Evaluator.

    First uses a rule-based method to judge predictions.
    If a sample is marked as incorrect by the rule-based method,
    then it uses an LLM judge to re-evaluate it.

    Arguments:
        llm_evaluator (dict): Configuration for the LLM evaluator.
        rule_evaluator (Optional[dict]): Configuration for the
            rule-based evaluator.
        sample_score_fn (Optional[Callable]): A function to
            score individual samples. If provided without rule_evaluator,
            this function will be used directly.
        parallel (bool): Whether to run in parallel mode.
    """

    def __init__(
        self,
        llm_evaluator: Dict,
        rule_evaluator: Optional[Dict] = None,
        sample_score_fn: Optional[Callable] = None,
        parallel: bool = True,
    ) -> None:
        self.logger = get_logger()

        # Initialize the LLM evaluator
        llm_evaluator_type = llm_evaluator.pop('type')
        if isinstance(llm_evaluator_type, str):
            llm_evaluator_type = ICL_EVALUATORS.get(llm_evaluator_type)
        self.llm_evaluator = llm_evaluator_type(**llm_evaluator)

        # Initialize the rule evaluator if provided
        self.rule_evaluator = None
        if rule_evaluator:
            rule_evaluator_type = rule_evaluator.pop('type')
            if isinstance(rule_evaluator_type, str):
                rule_evaluator_type = ICL_EVALUATORS.get(rule_evaluator_type)
            self.rule_evaluator = rule_evaluator_type(**rule_evaluator)

        self.sample_score_fn = sample_score_fn
        self.parallel = parallel

        # At least one of rule_evaluator or sample_score_fn must be provided
        if not self.rule_evaluator and not self.sample_score_fn:
            raise ValueError(
                'Either rule_evaluator or sample_score_fn must be provided')

    def sample_score(self, prediction: str, reference: str) -> Dict[str, Any]:
        """Score a single sample using sample_score_fn or rule_evaluator.

        Args:
            prediction: The model's prediction.
            reference: The ground truth.

        Returns:
            Dict: A dictionary containing the score and other details.
        """
        if self.sample_score_fn:
            # Use user-provided function to evaluate a single sample
            result = self.sample_score_fn(prediction, reference)
            if not isinstance(result, dict):
                # Ensure result is a dictionary with at least 'correct' field
                result = {
                    'correct': bool(result),
                    'pred': prediction,
                    'answer': reference,
                }
            return result
        else:
            # Use rule_evaluator to evaluate a single sample by calling
            # the score method with single-element lists
            result = self.rule_evaluator.score([prediction], [reference])
            if 'details' in result and len(result['details']) > 0:
                return result['details'][0]
            else:
                # Fallback if rule_evaluator doesn't provide detailed results
                return {
                    'correct': result.get('accuracy', 0) > 0,
                    'pred': prediction,
                    'answer': reference,
                }

    def _get_llm_correctness(self, llm_detail):
        """Determine if the LLM judge considers the answer correct.

        Args:
            llm_detail: The evaluation details from the LLM judge.

        Returns:
            bool: Whether the answer is correct according to the LLM judge.
        """
        if 'prediction' in llm_detail:
            response = llm_detail['prediction'].strip().upper()
            return response == 'A' or response.startswith('CORRECT')
        elif 'correct' in llm_detail:
            return llm_detail['correct']
        elif 'score' in llm_detail:
            return llm_detail['score'] > 0.5
        return False

    def score(
        self,
        predictions: List[str],
        references: List[str],
        test_set: Optional[Dataset] = None,
    ) -> Dict[str, Any]:
        """Score predictions using cascade or parallel evaluation.

        Args:
            predictions: List of model predictions.
            references: List of ground truths.
            test_set: Huggingface Dataset containing original test samples.

        Returns:
            Dict: A dictionary containing the scores and details.
        """
        self.logger.info(
            f"Running {'parallel' if self.parallel else 'cascade'} evaluation")

        # Step 1: Evaluate each sample individually using rule-based evaluation
        details = []
        failed_predictions = []
        failed_references = []
        failed_indices = []

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            result = self.sample_score(pred, ref)
            result['evaluation_method'] = 'rule'
            details.append({'rule_evaluation': result})

            # If the sample failed rule-based evaluation or in parallel
            # mode, mark it for LLM evaluation
            if not result.get('correct', False) or self.parallel:
                failed_predictions.append(pred)
                failed_references.append(ref)
                failed_indices.append(i)

        # Calculate initial accuracy based on rule evaluation
        initial_correct = sum(
            1 for detail in details
            if detail['rule_evaluation'].get('correct', False))
        initial_accuracy = (100 * initial_correct /
                            len(predictions) if predictions else 0)

        self.logger.info(
            f'Rule-based evaluation: {initial_correct}/{len(predictions)} '
            f'correct ({initial_accuracy:.2f}%)')

        eval_mode = ('parallel (all samples)'
                     if self.parallel else 'cascade (only failed samples)')
        self.logger.info(f'Samples requiring LLM evaluation ({eval_mode}): '
                         f'{len(failed_indices)}')

        # Step 2: If there are samples for LLM evaluation
        if failed_predictions and test_set is not None:
            self.logger.info(f'Running LLM evaluation in {eval_mode} mode...')

            # Create a subset of the test_set for LLM evaluation
            failed_subset = test_set.select(failed_indices)

            # Add prediction and reference columns to the dataset
            failed_subset = failed_subset.add_column('prediction',
                                                     failed_predictions)
            failed_subset = failed_subset.add_column('reference',
                                                     failed_references)

            # Set a custom output path for LLM evaluation
            original_out_dir = getattr(self.llm_evaluator, '_out_dir', None)
            self.llm_evaluator._out_dir = f'{self._out_dir}_llm_judge'

            # Check if results already exist to avoid re-evaluation
            llm_results_path = f'{self.llm_evaluator._out_dir}.json'
            if os.path.exists(llm_results_path):
                self.logger.info(
                    f'Loading existing LLM evaluation results from '
                    f'{llm_results_path}')
                llm_results = mmengine.load(llm_results_path)

                # Extract details from loaded results
                if llm_results.get('details', []):
                    loaded_details = llm_results['details']
                else:
                    loaded_details = llm_results

                # Strictly verify that the loaded results match
                # the current evaluation needs
                if len(loaded_details) != len(failed_indices):
                    error_msg = (
                        f'Error: Loaded LLM results contain '
                        f'{len(loaded_details)} samples, but current '
                        f'evaluation requires {len(failed_indices)} samples. '
                        f"The cached results at {llm_results_path} don't match"
                        f'the current evaluation needs. '
                        f'Please remove the cache file or fix the mismatch.')
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

            else:
                # Use GenericLLMEvaluator to evaluate samples
                # unset dataset_cfg for GenericLLMEvaluator to
                # directly use test_set
                self.llm_evaluator.dataset_cfg = None
                llm_results = self.llm_evaluator.score(
                    predictions=failed_predictions,
                    references=failed_references,
                    test_set=failed_subset,
                )

            # Restore original output directory
            if original_out_dir:
                self.llm_evaluator._out_dir = original_out_dir

            if llm_results.get('details', []):
                llm_details = llm_results['details']
            else:
                llm_details = llm_results

            # Initialize counters for accuracy calculation
            final_correct = initial_correct if not self.parallel else 0
            llm_correct = 0
            llm_evaluated = 0

            # Update the details for samples that were evaluated by LLM
            for i, llm_detail in enumerate(llm_details.values()):
                original_index = failed_indices[i]
                # Store original rule-based evaluation result
                rule_result = details[original_index].copy()
                rule_correct = rule_result['rule_evaluation'].get(
                    'correct', False)

                # Add LLM evaluation details
                details[original_index]['llm_evaluation'] = llm_detail

                # Determine LLM correctness judgment and store it
                is_correct = self._get_llm_correctness(llm_detail)
                details[original_index]['llm_evaluation'][
                    'llm_correct'] = is_correct

                # Count LLM evaluation statistics
                llm_evaluated += 1
                if is_correct:
                    llm_correct += 1

                # Update final_correct counter based on evaluation mode
                if self.parallel:
                    # In parallel mode, either rule-based or LLM evaluations
                    # should be correct
                    if rule_correct or is_correct:
                        final_correct += 1
                else:
                    # In cascade mode, if rule was incorrect but LLM
                    # correct, increment
                    # (rule correct samples are already counted
                    # in initial_correct)
                    if not rule_correct and is_correct:
                        final_correct += 1

            # Calculate final accuracy
            final_accuracy = (100 * final_correct /
                              len(predictions) if predictions else 0)
            llm_accuracy = (100 * llm_correct /
                            llm_evaluated if llm_evaluated else 0)

            self.logger.info(
                f'Final evaluation: {final_correct}/{len(predictions)}'
                f'correct ({final_accuracy:.2f}%)')

            if llm_evaluated > 0:
                self.logger.info(
                    f'LLM evaluation: {llm_correct}/{llm_evaluated} '
                    f'correct ({llm_accuracy:.2f}%)')

            result = {
                'accuracy': final_accuracy,
                'cascade_stats': {
                    'total_samples': len(predictions),
                    'rule_correct': initial_correct,
                    'rule_accuracy': initial_accuracy,
                    'llm_evaluated': llm_evaluated,
                    'llm_correct': llm_correct,
                    'llm_accuracy': llm_accuracy,
                    'final_correct': final_correct,
                    'final_accuracy': final_accuracy,
                    'parallel_mode': self.parallel,
                },
                'details': details,
            }

            return result
