from typing import List, Optional, Union

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .ds1000 import DS1000Dataset


@LOAD_DATASET.register_module()
class DS1000Dataset_Interperter(DS1000Dataset):
    """Code interpreter version of DS1000."""

    def load(self,
             path: str,
             libs: Optional[Union[str, list]] = None,
             mode: str = 'Insertion'):
        dataset = super().load(path, libs, mode)

        def preprocess(example):
            """Get rid of unnecessary code block in prompt."""
            prompt = example.pop('prompt')
            example['prompt'] = prompt[:prompt.find('A:\n')].strip()
            return example

        dataset = dataset.map(preprocess)
        return dataset


class DS1000InterpreterEvaluator(BaseEvaluator):

    def score(self, predictions: List, references: List, steps: List):
        """Calculate accuracy."""

        passed = 0
        total = len(steps)
        for each_steps in steps:
            succeed = 0
            for idx, step in enumerate(each_steps):
                if step['type'] == 'PythonInterpreter':
                    # assert must in code for testing
                    # otherwise the result will be True
                    if step['args'] and 'assert' in step['args']['text']:
                        # successful result should count as passed
                        if step['result']:
                            succeed = step['result']['text'] == 'True'
                elif step['type'] == 'FinishAction' and idx == 0:
                    # return without any python interpreter
                    # should count as failed
                    break
            passed += succeed

        return dict(accuracy=passed / total * 100)
