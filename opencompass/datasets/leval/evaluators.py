import json
from typing import List

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils.prompt import PromptList
from opencompass.utils.text_postprocessors import general_postprocess


@ICL_EVALUATORS.register_module()
class LEvalGPTEvaluator(BaseEvaluator):
    """Use OpenAI's models to evaluate prediction.

    Args:
        battle_model (str): The rival model name in evaluate module. Defaults
            to 'turbo-16k-0613'.
        evaluator_path (str): The judge model name in evaluate module. Note
            that the key will be fetched from the environment variable
            $OPENAI_API_KEY, as how openai defaults to be.
            Defaults to 'gpt-4-0613'.
    """

    def __init__(self,
                 battle_model: str = 'turbo-16k-0613',
                 evaluator_path: str = 'gpt-4-0613') -> None:
        self.battle_model = battle_model
        self.evaluator_path = evaluator_path
        super().__init__()

    def run_judge_pair(self, prompt_template, system_prompt, question,
                       answer_a, answer_b, reference):
        from opencompass.models import OpenAI
        user_prompt = prompt_template.format(question=question,
                                             answer_a=answer_a,
                                             answer_b=answer_b,
                                             reference=reference)
        messages = PromptList([{
            'role': 'SYSTEM',
            'fallback_role': 'HUMAN',
            'prompt': system_prompt
        }, {
            'role': 'HUMAN',
            'prompt': user_prompt
        }])
        model = OpenAI(path=self.evaluator_path,
                       max_seq_len=16384,
                       query_per_second=1,
                       retry=5,
                       temperature=0.0)
        response = model._generate(input=messages,
                                   max_out_len=2048,
                                   temperature=0.0)
        if '[[A]]' in response:
            winner = 'A'
        elif '[[B]]' in response:
            winner = 'B'
        elif '[[C]]' in response:
            winner = 'tie'
        else:
            winner = 'error'

        return winner

    def score(self, predictions: List, references: List) -> dict:
        system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question about the content of a long document.  You will be given a reference answer written by human, assistant A's answer, and assistant B's answer. Your job is to evaluate which assistant's answer is better. Begin your evaluation by comparing both assistants' answers with the reference answer. Additional details or information that are not mentioned in reference answer cannot be considered as advantages and do not let them sway your judgment. Your evaluation should also consider the relevance to user's question but it is more important to avoid factual errors according to the reference answer. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."  # noqa
        prompt_template = "[User Question]\n{question}\n\n[The Start of Reference Answer]\n{reference}\n[The End of Reference Answer]\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"  # noqa
        battle_samples = []
        with open(
                'opencompass/datasets/leval/' + self.battle_model +
                '.pred.jsonl', 'r') as f:
            for i, line in enumerate(f):
                battle_samples.append(json.loads(line))

        score = 0.
        bad_case = 0
        num_samples = 0
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference = references[i]
            for sample in battle_samples:
                if reference == sample['gt']:
                    question = sample['query']
                    battle_answer = sample[self.battle_model + '_pred']

                    winner = self.run_judge_pair(prompt_template,
                                                 system_prompt, question,
                                                 prediction, battle_answer,
                                                 reference)
                    if winner == 'A':
                        score += 1
                    elif winner == 'tie':
                        score += 0.5
                    elif winner == 'error':
                        bad_case += 1

                    winner = self.run_judge_pair(prompt_template,
                                                 system_prompt, question,
                                                 battle_answer, prediction,
                                                 reference)
                    if winner == 'B':
                        score += 1
                    elif winner == 'tie':
                        score += 0.5
                    elif winner == 'error':
                        bad_case += 1

                    num_samples += 2

        score = score / (num_samples - bad_case) * 100
        return {'score': score}


@ICL_EVALUATORS.register_module()
class LEvalEMEvaluator(BaseEvaluator):
    """Exact match evaluator."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        predictions = [
            general_postprocess(prediction) for prediction in predictions
        ]
        processed_answers = [general_postprocess(i) for i in references]

        cnt = 0
        for pred, ans, origin_ans in zip(predictions, processed_answers,
                                         references):
            if ans in pred or origin_ans in pred:
                cnt += 1

        score = cnt / len(predictions) * 100

        return {'score': score}
