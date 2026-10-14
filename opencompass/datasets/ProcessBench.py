import re

from datasets import Dataset, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


def extract_answer(solution_text: str):
    boxed_pattern = r'boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        try:
            return int(matches[-1].replace('{', '').replace('}', '').strip())
        except ValueError:
            return matches[-1].replace('{', '').replace('}', '').strip()
    return None


@LOAD_DATASET.register_module()
class ProcessBenchEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str, subset: str, **kwargs):
        # Load from HuggingFace datasets
        input_data = load_dataset(path, split=subset)

        # Process data to match expected format for inferencer
        processed_data = []
        for item in input_data:
            problem = item['problem']
            steps = item['steps']
            tagged_response = ''
            for sdx, step in enumerate(steps):
                tagged_response += (f'<paragraph_{sdx}>\n{step}\n'
                                    f'</paragraph_{sdx}>\n\n')
            tagged_response = tagged_response.strip()

            processed_data.append({
                'problem': problem,
                'tagged_response': tagged_response,
                'label': item['label']
            })

        dataset = Dataset.from_list(processed_data)
        return dataset


@ICL_EVALUATORS.register_module()
class ProcessBenchEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        res_data = []
        for i in range(len(predictions)):
            d = {}
            generated_critique = predictions[i]
            pred = extract_answer(generated_critique)

            # Convert reference to int if possible for comparison
            try:
                ref_label = int(references[i])
            except ValueError:
                ref_label = references[i]

            d['generated_critique'] = generated_critique
            d['prediction'] = pred
            d['label'] = ref_label
            d['match'] = (pred == ref_label)

            res_data.append(d)

        error_data = [e for e in res_data if e['label'] != -1]
        correct_data = [e for e in res_data if e['label'] == -1]

        acc1 = 0.0
        if len(error_data) > 0:
            acc1 = (sum([e['match']
                         for e in error_data]) / len(error_data) * 100)

        acc2 = 0.0
        if len(correct_data) > 0:
            acc2 = (sum([e['match']
                         for e in correct_data]) / len(correct_data) * 100)

        f1 = 0.0
        if (acc1 + acc2) > 0:
            f1 = 2 * acc1 * acc2 / (acc1 + acc2)

        return {
            'error_acc': acc1,
            'correct_acc': acc2,
            'f1': f1,
            'details': res_data
        }
