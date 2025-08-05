import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm import tqdm

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS

from .CodeCompass import CodeCompassCodeGenerationDataset
from .codecompass_runner import run_test_for_cpp_problem
from .metrics import compute_metrics_from_results
from .utils import extract_cpp_code


@ICL_EVALUATORS.register_module()
class CodeCompassEvaluator(BaseEvaluator):
    """CodeCompass C++ Code Generation Evaluator."""

    def __init__(self,
                 num_process_evaluate: int = 16,
                 timeout: int = 15,
                 k_list: List[int] = None,
                 temp_base_dir: str = None,
                 dataset_path: str = None):
        super().__init__()
        self.num_process_evaluate = num_process_evaluate
        self.default_timeout = timeout
        self.k_list = k_list or [1]
        self.temp_base_dir = temp_base_dir

        if not dataset_path:
            raise ValueError('dataset_path must be provided to the evaluator.')

        full_dataset = CodeCompassCodeGenerationDataset.load(
            path=dataset_path)['test']

        self.test_cases_lookup = {
            item['question_id']: item['evaluation_sample']
            for item in full_dataset
        }

    def score(self, predictions: List[Any],
              references: List[Any]) -> Dict[str, float]:
        tasks = []
        for i, (pred, metadata) in enumerate(zip(predictions, references)):
            try:
                question_id = metadata.get('question_id')
                if not question_id:
                    print(
                        f"Warning: 'question_id' not found in metadata at index {i}. Skipping."
                    )
                    continue

                eval_sample = self.test_cases_lookup.get(question_id)
                if not eval_sample:
                    print(
                        f"Warning: Could not find test cases for question_id '{question_id}', skipping."
                    )
                    continue

                sample = {'evaluation_sample': eval_sample}

                timeout = metadata.get('time_limit_s', self.default_timeout)
                mem_limit_mb = metadata.get('memory_limit_mb', 256)

                if isinstance(pred, str):
                    pred = [pred]
                extracted_codes = [
                    extract_cpp_code(code) for code in pred
                    if isinstance(code, str)
                ]
                if not any(extracted_codes):
                    continue

                tasks.append((sample, extracted_codes, timeout, mem_limit_mb,
                              self.temp_base_dir))

            except Exception as e:
                print(
                    f"Error preparing task for question_id '{question_id}': {e}"
                )
                continue

        if not tasks:
            return {f'pass@{k}': 0.0 for k in self.k_list}

        results_list = self._run_parallel_evaluation(tasks)
        final_results = {i: results_list[i] for i in range(len(results_list))}
        metrics = compute_metrics_from_results(final_results,
                                               k_list=self.k_list)

        return metrics

    def _prepare_sample(self, reference: Any, idx: int = -1) -> Dict[str, Any]:

        try:

            if idx <= 2:
                if isinstance(reference, dict):
                    print(f'Reference keys: {list(reference.keys())}')
                    for key, value in reference.items():
                        print(
                            f'  {key}: {type(value)} - {str(value)[:100]}...')
                else:
                    print(f'Reference value: {str(reference)[:200]}...')

            if isinstance(reference,
                          dict) and 'evaluation_sample' in reference:
                eval_sample = reference['evaluation_sample']
                if isinstance(eval_sample, str):
                    eval_sample = json.loads(eval_sample)
                return {'evaluation_sample': eval_sample}

            elif isinstance(reference, str):
                eval_sample = json.loads(reference)
                return {'evaluation_sample': eval_sample}

            elif isinstance(reference, dict):
                if 'test_cases' in reference:
                    test_cases = reference['test_cases']
                    if isinstance(test_cases, str):
                        test_cases = json.loads(test_cases)

                    inputs = [case.get('input', '') for case in test_cases]
                    outputs = [case.get('output', '') for case in test_cases]

                    eval_sample = {
                        'inputs': inputs,
                        'outputs': outputs,
                        'fn_name': reference.get('fn_name')
                    }
                    return {'evaluation_sample': eval_sample}

                elif 'cases' in reference:
                    cases_str = reference['cases']
                    if isinstance(cases_str, str):
                        cases_list = json.loads(cases_str)
                    else:
                        cases_list = cases_str

                    inputs = [case.get('input', '') for case in cases_list]
                    outputs = [case.get('output', '') for case in cases_list]

                    eval_sample = {
                        'inputs': inputs,
                        'outputs': outputs,
                        'fn_name': reference.get('fn_name')
                    }
                    return {'evaluation_sample': eval_sample}

                elif 'inputs' in reference and 'outputs' in reference:
                    eval_sample = {
                        'inputs': reference['inputs'],
                        'outputs': reference['outputs'],
                        'fn_name': reference.get('fn_name')
                    }
                    return {'evaluation_sample': eval_sample}

                elif 'input' in reference and 'output' in reference:

                    inputs = [reference['input']] if isinstance(
                        reference['input'], str) else reference['input']
                    outputs = [reference['output']] if isinstance(
                        reference['output'], str) else reference['output']

                    eval_sample = {
                        'inputs': inputs,
                        'outputs': outputs,
                        'fn_name': reference.get('fn_name')
                    }
                    return {'evaluation_sample': eval_sample}

                else:
                    print(
                        f'Warning: Trying to use entire reference as evaluation_sample for sample {idx}'
                    )
                    return {'evaluation_sample': reference}

            print(
                f'Cannot handle reference format: {type(reference)} for sample {idx}'
            )
            return None

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f'Error preparing sample {idx}: {e}')
            import traceback
            traceback.print_exc()
            return None

    def _run_parallel_evaluation(self,
                                 tasks: List[tuple]) -> List[List[List[int]]]:
        results_list = [[] for _ in range(len(tasks))]

        with ProcessPoolExecutor(
                max_workers=self.num_process_evaluate) as executor:
            with tqdm(total=len(tasks),
                      desc='Evaluating Code Generation') as pbar:
                future_to_idx = {
                    executor.submit(run_test_for_cpp_problem, *task): i
                    for i, task in enumerate(tasks)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results_list[idx] = result
                    except Exception as e:
                        print(f'FATAL ERROR in task {idx}: {e}')

                        num_gens = len(tasks[idx][1])
                        results_list[idx] = [[-3] * 100] * num_gens
                    finally:
                        pbar.update(1)

        return results_list


# 为了向后兼容，保留原有的类名
CodeCompassCppEvaluator = CodeCompassEvaluator
