from typing import List, Optional, Union

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .ds1000 import DS1000Dataset


@LOAD_DATASET.register_module()
class DS1000Dataset_Interperter(DS1000Dataset):
    """Code interpreter version of DS1000."""

    def load(
        self,
        path: str,
        libs: Optional[Union[str, list]] = None,
        mode: str = 'Insertion',
    ):
        dataset = super().load(path, libs, mode)

        def preprocess(example):
            """Get rid of unnecessary code block in prompt."""
            prompt = example.pop('prompt')
            example['prompt'] = prompt[:prompt.find('A:\n')].strip()
            return example

        dataset = dataset.map(preprocess)
        return dataset


class DS1000InterpreterEvaluator(BaseEvaluator):
    """DS1000 interpreter evaluator.

    Args:
        action (str): Action for catching internal prediction.
            Defaults to `PythonInterpreter`.
    """

    def __init__(self, action: str = 'PythonInterpreter'):
        self.action = action

    def get_action(self, step):
        for s in step[::-1]:
            if s['type'] == self.action:
                return s

    def score(self, predictions: List, references: List, steps: List):
        """Calculate accuracy."""

        action_scope = 0
        follow_scope = 0
        soft_success = 0
        success = 0
        total = len(references)
        for step in steps:
            s = self.get_action(step)
            if s:
                action_scope += 1
                if not s['errmsg']:
                    soft_success += 1
                # assert must in code for testing
                # otherwise the result will be True
                if s['args'] and 'assert' in s['args']['text']:
                    follow_scope += 1
                    # successful result should count as passed
                    if s['result']:
                        success += s['result']['text'] == 'True'

        result = dict(
            action_pct=100 * action_scope / total,
            soft_code_acc=100 * soft_success / total,
            follow_acc=100 * follow_scope / total,
            code_acc=100 * success / total,
        )
        return result
