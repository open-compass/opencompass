# flake8: noqa

import os
from typing import Any, Callable, Dict, List, Optional

import mmengine
from datasets import Dataset
from nltk import accuracy

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils.logging import get_logger
from opencompass.utils.multi_box_extract import multi_box_extract_processor


@ICL_EVALUATORS.register_module()
class MultiCascadeEvaluator(BaseEvaluator):
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
        super().__init__()
        self.logger = get_logger(__name__)

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

    def sample_score(self,
                     prediction: str,
                     reference: str,
                     test_set=None) -> Dict[str, Any]:
        """Score a single sample using sample_score_fn or rule_evaluator.

        Args:
            prediction: The model's prediction.
            reference: The ground truth.

        Returns:
            Dict: A dictionary containing the score and other details.
        """
        if self.sample_score_fn:
            # Use user-provided function to evaluate a single sample
            result = self.sample_score_fn(prediction, reference, test_set)
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
            result = self.rule_evaluator.score([prediction], [reference],
                                               [test_set])
            return {'accuracy': 0.0, 'details': [{'correct': False}]}
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
        elif 'llm_judge' in llm_detail:
            response = llm_detail['llm_judge'].strip().upper()
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

        for i, (pred_origin,
                ref_list) in enumerate(zip(predictions, references)):
            if test_set is not None:
                test_item = test_set[i]
            else:
                test_item = None
            # Apply prediction postprocessing for each sample
            pred_list = multi_box_extract_processor(pred_origin,
                                                    num_answers=len(ref_list))

            this_details = []
            this_failed_predictions = []
            this_failed_references = []
            this_failed_indices = []
            for j, (pred, ref) in enumerate(zip(pred_list, ref_list)):
                result = self.sample_score(pred, ref, test_item)
                result['evaluation_method'] = 'rule'
                this_details.append({'rule_evaluation': result})

                # If the sample failed rule-based evaluation or in parallel
                # mode, mark it for LLM evaluation
                if not result.get('correct', False) or self.parallel:
                    this_failed_predictions.append(pred)
                    this_failed_references.append(ref)
                    this_failed_indices.append(i)

            details.append(this_details)
            failed_predictions.append(this_failed_predictions)
            failed_references.append(this_failed_references)
            failed_indices.append(this_failed_indices)

        # Calculate initial accuracy based on rule evaluation
        sum_score_list = []
        for i in range(len(details)):
            this_score_list = []
            for j in range(len(details[i])):
                this_score_list.append(
                    details[i][j]['rule_evaluation'].get('correct', False) *
                    1 / len(details[i]))
            sum_score_list.append(this_score_list)
        score_sum = sum([sum(i) for i in sum_score_list])

        self.logger.info(f'Rule-based evaluation: {sum_score_list} \n'
                         f'score ({score_sum})')

        eval_mode = ('parallel (all samples)'
                     if self.parallel else 'cascade (only failed samples)')
        self.logger.info(f'Samples requiring LLM evaluation ({eval_mode}): '
                         f'{len(failed_indices)}')

        # Step 2: If there are samples for LLM evaluation
        if failed_predictions and test_set is not None:
            self.logger.info(f'Running LLM evaluation in {eval_mode} mode...')

            # Create a subset of the test_set for LLM evaluation
            failed_subset = test_set.select([i[0] for i in failed_indices])

            # Add prediction and reference columns to the dataset
            failed_subset = failed_subset.add_column('prediction',
                                                     failed_predictions)
            failed_subset = failed_subset.add_column('reference',
                                                     failed_references)

            ###################################
            new_data = []
            same_origin_flag = 0
            for i in range(len(failed_subset)):
                item = failed_subset[i]

                attribute_predictions = item['prediction']
                attribute_references = item['reference']
                # attribute_points = item["extra_info"]["points"]

                n = len(attribute_predictions)
                for j in range(n):
                    new_item = {key: value for key, value in item.items()}

                    new_item['prediction'] = attribute_predictions[j]
                    new_item['answer'] = attribute_references[j]
                    new_item['reference'] = attribute_references[j]
                    new_item['points'] = 1 / n
                    new_item['same_origin_flag'] = same_origin_flag

                    new_data.append(new_item)
                same_origin_flag += 1

            failed_subset = Dataset.from_list(new_data)
            ###################################

            # Set a custom output path for LLM evaluation
            original_out_dir = getattr(self.llm_evaluator, '_out_dir', None)
            self.llm_evaluator._out_dir = f'{self._out_dir}_llm_judge'

            # Generate random hash suffix
            llm_results_path = f'{self.llm_evaluator._out_dir}_replica{self.dataset_replica_idx}.json'  # noqa
            self.logger.info(f'LLM evaluation results will be saved at '
                             f'{llm_results_path}')
            # Check if results already exist to avoid re-evaluation
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

                # if len(loaded_details) != len(failed_indices):
                #     error_msg = (
                #         f'Error: Loaded LLM results contain '
                #         f'{len(loaded_details)} samples, but current '
                #         f'evaluation requires {len(failed_indices)} samples. '
                #         f"The cached results at {llm_results_path} don't match"
                #         f'the current evaluation needs. '
                #         f'Please remove the cache file or fix the mismatch.')
                #     self.logger.error(error_msg)
                #     raise ValueError(error_msg)

            else:
                # Use GenericLLMEvaluator to evaluate samples
                # unset dataset_cfg for GenericLLMEvaluator to
                # directly use test_set
                # self.llm_evaluator.output_path = llm_results_path
                self.llm_evaluator._dataset_replica_idx = \
                    self._dataset_replica_idx
                self.llm_evaluator.dataset_cfg = None

                # Apply prediction postprocessing to for LLM evaluator
                failed_predictions = self.llm_evaluator.pred_postprocess(
                    failed_predictions)

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
            if isinstance(llm_details, dict):
                llm_details_iter = llm_details.values()
            else:
                llm_details_iter = llm_details

            # Initialize counters for accuracy calculation

            # final_correct = initial_correct if not self.parallel else 0

            # Update the details for samples that were evaluated by LLM

            correct_list = []
            for i, llm_detail in enumerate(llm_details_iter):

                # Add dataset replica index to LLM evaluation result
                is_correct = self._get_llm_correctness(llm_detail)
                correct_list.append(is_correct)
                score_sum += is_correct * failed_subset[i]['points']
                self.logger.info(
                    f'this question: \n{failed_subset[i]["prediction"]} \n{failed_subset[i]["reference"]} \n{failed_subset[i]["points"]} \n{is_correct}'
                )

                llm_detail['dataset_replica_idx'] = self.dataset_replica_idx

            self.logger.info(f'Final evaluation score: {score_sum}')

            accuracy_list = []
            group_start_index = 0
            current_index = failed_subset[0]['same_origin_flag']
            for i in range(len(failed_subset)):
                if failed_subset[i]['same_origin_flag'] != current_index:
                    group = correct_list[group_start_index:i]
                    print(group)
                    if True in group:
                        accuracy_list.append(True)
                    else:
                        accuracy_list.append(False)
                    group_start_index = i
                    current_index = failed_subset[i]['same_origin_flag']
            group = correct_list[group_start_index:]
            if True in group:
                accuracy_list.append(True)
            else:
                accuracy_list.append(False)

            accuracy = 0
            for i in accuracy_list:
                if i is True:
                    accuracy += 1

            accuracy /= len(predictions)
            result = {'score': score_sum / 4, 'accuracy': accuracy * 100}

            return result
