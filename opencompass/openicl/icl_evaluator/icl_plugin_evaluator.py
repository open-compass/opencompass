"""Plugin Evaluator."""

import json


class TEvalEvaluator:
    """This module contains the following evaluators for evaluating the
    capabilities of the various dimensions of the LLM.

    specifically, InstructEvaluator is used to evaluate the instruction
    following capability of LLM, i.e. the ability of the model to perform tool
    calls according to an predefined format. ReasoningEvaluator is used to
    evaluate the model's ability to reason about the next execution step based
    on historical observations. PlanningEvaluator is used to evaluate the
    model's ability to plan a solution or program based on a given task.
    APIRetrievalEvaluator is used to evaluate the model's ability to retrieve a
    subset of tools relevant to the given task from a large number of tools.
    ReviewEvaluator is used to evaluate the model's ability to review whether a
    task was successfully completed.
    """

    def __init__(self, subset) -> None:

        from opencompass.datasets.teval.evaluators import (
            InstructEvaluator, PlanningEvaluator,
            ReasonRetrieveUnderstandEvaluator, ReviewEvaluator)

        super().__init__()
        self.subset = subset
        if subset == 'instruct':
            self.evaluator = InstructEvaluator('')
        elif subset == 'plan':
            self.evaluator = PlanningEvaluator('')
        elif subset == 'review':
            self.evaluator = ReviewEvaluator('')
        elif subset == 'reason_retrieve_understand':
            self.evaluator = ReasonRetrieveUnderstandEvaluator('')
        elif subset == 'reason':
            self.evaluator = ReasonRetrieveUnderstandEvaluator(
                '', default_prompt_type='str', eval_type='reason')
        elif subset == 'retrieve':
            self.evaluator = ReasonRetrieveUnderstandEvaluator(
                '', default_prompt_type='str', eval_type='retrieve')
        elif subset == 'understand':
            self.evaluator = ReasonRetrieveUnderstandEvaluator(
                '', default_prompt_type='str', eval_type='understand')

        elif subset == 'instruct_zh':
            self.evaluator = InstructEvaluator('')
        elif subset == 'plan_zh':
            self.evaluator = PlanningEvaluator(
                '', bert_score_model='thenlper/gte-large-zh')
        elif subset == 'review_zh':
            self.evaluator = ReviewEvaluator('')
        elif subset == 'reason_retrieve_understand_zh':
            self.evaluator = ReasonRetrieveUnderstandEvaluator(
                '', bert_score_model='thenlper/gte-large-zh')
        elif subset == 'reason_zh':
            self.evaluator = ReasonRetrieveUnderstandEvaluator(
                '',
                default_prompt_type='str',
                eval_type='reason',
                bert_score_model='thenlper/gte-large-zh')
        elif subset == 'retrieve_zh':
            self.evaluator = ReasonRetrieveUnderstandEvaluator(
                '', default_prompt_type='str', eval_type='retrieve')
        elif subset == 'understand_zh':
            self.evaluator = ReasonRetrieveUnderstandEvaluator(
                '',
                default_prompt_type='str',
                eval_type='understand',
                bert_score_model='thenlper/gte-large-zh')
        else:
            raise NotImplementedError

    def score(self, predictions, references):

        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        results_list = []
        for prediction, reference in zip(predictions, references):

            datum = json.loads(reference)
            datum['prediction'] = prediction

            data_sample = self.evaluator._process_response(datum)
            if isinstance(data_sample, tuple):
                data_sample = data_sample[0]
            metrics_result = self.evaluator._evaluate(data_sample)
            results_list.append(metrics_result)
        results_dict = self.evaluator._post_process(results_list)
        results_dict = {k: v * 100 for k, v in results_dict.items()}
        return results_dict
