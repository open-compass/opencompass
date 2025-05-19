import ast
import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import get_logger

from .execute_utils import BASE_IMPORTS, codeexecute_check_correctness
from .extract_utils import (extract_code_execution, extract_code_generation,
                            extract_code_generation_v2,
                            extract_test_output_code)
from .livecodebench import LCBCodeGenerationDataset
from .pass_k_utils import compute_metrics_from_results


def codegen_check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.

    The global timeout is to catch some extreme/rare cases not handled by the
    timeouts inside `run_test`
    """

    def _temp_run(sample, generation, debug, result, metadata_list, timeout):
        from .testing_util import run_test
        res, metadata = run_test(sample,
                                 test=generation,
                                 debug=debug,
                                 timeout=timeout)
        result.append(res)
        metadata_list.append(metadata)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) *
           len(json.loads(sample['input_output'])['inputs']) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample['input_output'])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs['inputs']))]]
        if debug:
            logger = get_logger()
            logger.info('global timeout')
    return result[0], metadata_list[0]


def evaluate_generations_by_problem(problem_generations: list, sample: list,
                                    debug: bool, timeout: int):
    """Evaluate each problem.

    Args:
        problem_generations:
        sample:
        debug:
        timeout
    """
    # problem_generations: list[str] = args[0]
    # sample = args[1]
    # debug: bool = args[2]
    # timeout: int = args[3]
    logger = get_logger()

    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res, curr_metadata = codegen_check_correctness(
                sample, o, timeout=timeout, debug=debug)
            if debug:
                logger.info(f'\nSuccessful compilation of task {o_idx}!')
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    logger.info(
                        f'Results were not True for all test cases'  # noqa: F541, E501
                        f' {curr_res=}\n')
        except Exception as e:
            if debug:
                logger.info(
                    f'Compilation failed, test framework exception'  # noqa: F541, E501
                    f' = {repr(e)}{e}\n')
            # break
            curr_metadata = {}
        finally:
            assert isinstance(curr_res, list)
            assert isinstance(curr_metadata, dict)
            res.append(curr_res)
            metadata.append(curr_metadata)
    if debug:
        for i, r in enumerate(problem_generations):
            logger.info(f'Sample\n{r}\nResult\n{res[i]}')
            logger.info('*' * 30 + '\n\n')
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    """We take the list of code generations and try to compile them and the run
    their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS
            dataset)
        level: difficulty level used in the generation, can be "all",
            "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is
            a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test
            case [True] = passed test case
    """

    # generations are code generations in the same order of the dataset

    inputs = [[(generations_list[index], samples_list[index], debug, timeout),
               index] for index in range(len(generations_list))]

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(
                max_workers=1 if debug else num_process_evaluate) as executor:
            futures = {
                executor.submit(
                    evaluate_generations_by_problem,  # noqa: E501
                    problem_generations,
                    sample,
                    debug,
                    timeout): index
                for (problem_generations, sample, debug,
                     timeout), index in inputs
            }

            results = {}
            metadata = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index], metadata[index] = future.result()
                pbar.update(1)

    assert len(results) == len(
        inputs), f'results = {len(results)} inputs = {len(inputs)} {results=}'
    # results = {i: r for r, (_, i) in zip(results, inputs)}

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):
    logger = get_logger()

    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample,
              generation_list) in enumerate(zip(samples_list,
                                                generations_list)):
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    logger.info(f'LCBCodeGeneration: Evaluating {len(samples_linear)}...')

    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])

    for idx, sub_metadatas in sorted(metadatas_linear.items(),
                                     key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)

    final_metadata = []
    for key in sorted(list(metadatas.keys())):
        final_metadata.append(metadatas[key])
    for i in range(len(final_metadata)):
        if type(final_metadata[i]) is not list:
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

        assert len(final_metadata[i]) == len(
            generations_list[0]), f'{len(final_metadata[i])=}'

    return [metrics, results, final_metadata]


@ICL_EVALUATORS.register_module()
class LCBCodeGenerationEvaluator(BaseEvaluator):

    def __init__(self,
                 num_process_evaluate,
                 timeout=6,
                 release_version='release_v1',
                 extractor_version='v1',
                 start_date=None,
                 end_date=None):
        super().__init__()
        self.num_process_evaluate = num_process_evaluate
        self.timeout = timeout
        self.dataset = LCBCodeGenerationDataset.load(
            release_version=release_version,
            start_date=start_date,
            end_date=end_date)['test']
        self.extractor_version = extractor_version

    def _build_results(self, extracted_predictions, metrics, eval_results,
                       final_metadata):
        results = {}
        results['pass@1'] = metrics.get('pass@1', 0.0)
        details = []
        # Safely get the details list from metrics
        r = metrics.get('details', {}).get('pass@1', [])
        for i, (ep, er, fm) in enumerate(
                zip(extracted_predictions.values(), eval_results.values(),
                    final_metadata)):
            detail = {
                'extracted_prediction':
                ep[0] if isinstance(ep, list) and ep else ep,
                'eval_result': er[0] if isinstance(er, list) and er else er,
                'final_metadata': fm[0] if isinstance(fm, list) and fm else fm
            }
            # Use r[i] if available, otherwise fallback to False
            detail['correct'] = bool(r[i] == 100.0) if i < len(r) else False
            details.append(detail)
        results['details'] = details
        return results

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }

        if self.extractor_version == 'v1':
            predictions = [[extract_code_generation(item)]
                           for item in predictions]
        elif self.extractor_version == 'v2':
            predictions = [[extract_code_generation_v2(item)]
                           for item in predictions]

        evaluation_samples = dict()
        for idx in range(len(self.dataset)):
            evaluation_samples[self.dataset[idx][
                'question_id']] = self.dataset[idx]['evaluation_sample']

        filtered_predictions = []
        filtered_references = []
        for idx, item in enumerate(references):
            if item in self.dataset['question_id']:
                filtered_predictions.append(predictions[idx])
                filtered_references.append(item)

        filtered_references = [
            evaluation_samples[item] for item in filtered_references
        ]  # noqa: E501

        filtered_references = [{
            'input_output': item
        } for item in filtered_references]  # noqa: E501

        extracted_predictions = {}
        for idx, content in enumerate(filtered_predictions):
            extracted_predictions[idx] = content

        metrics, eval_results, final_metadata = codegen_metrics(
            filtered_references,
            filtered_predictions,
            k_list=[1],
            num_process_evaluate=self.num_process_evaluate,
            timeout=self.timeout,
        )
        # results = {
        #     'extracted_predictions': extracted_predictions,
        #     'eval_results': eval_results
        # }
        # results.update(metrics)

        return self._build_results(extracted_predictions, metrics,
                                   eval_results, final_metadata)


def evaluate_score(args) -> list[bool]:
    gs, (c, i, o) = args

    execution_results = []
    for g in gs:
        if i in g:
            pass
        else:
            code_to_execute = f'{BASE_IMPORTS}\n{c}\nassert {o} == {g}'
            execution_results.append(
                codeexecute_check_correctness(code_to_execute, 3))
    if len(execution_results) == 0:
        execution_results = [False] * len(gs)
    return execution_results


def code_execution_metrics(
    samples,
    generations,
):

    def pass_at_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    # execute the code
    references = [(doc['code'], doc['input'], doc['output'])
                  for doc in samples]
    with ProcessPoolExecutor() as executor:
        args_list = zip(generations, references)
        results = executor.map(evaluate_score, args_list)
    all_results = list(results)

    # serial version
    # all_results = []
    # for i in range(len(generations)):
    #     generation = generations[i]
    #     result = evaluate_score([generation, references[i]])
    #     all_results.append(result)

    # compute pass@1
    pass_at_1s = []
    for execution_result in all_results:
        c, n = execution_result.count(True), len(execution_result)
        pass_at_1s.append(pass_at_k(n, c, 1))
    metrics = {'pass@1': sum(pass_at_1s) / len(pass_at_1s) * 100}

    results = {}
    for i, r in enumerate(all_results):
        r_new = []
        for _r in r:
            r_new.append([_r])
        results[i] = r_new
    return [metrics, results]


@ICL_EVALUATORS.register_module()
class LCBCodeExecutionEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
        # self.num_process_evaluate = num_process_evaluate
        # self.timeout = timeout

    def score(self, predictions, references):
        predictions = [[extract_code_execution(item)] for item in predictions]
        references = [json.loads(item) for item in references]
        metrics, results = code_execution_metrics(references, predictions)
        return metrics


def parse_assert_statement(statement):
    """Parse a Python assert statement and extract the expected output from the
    right side of the '==' operator as a string.

    :param statement: A string containing the assert statement.
    :return: The expected output from the assert statement as a string.
    """
    try:
        parsed = ast.parse(statement, mode='exec')
    except SyntaxError:
        return 'Invalid syntax'

    if len(parsed.body) == 0:
        return 'Empty statement'

    if not isinstance(parsed.body[0], ast.Assert):
        return 'Not an assert statement'

    comparison = parsed.body[0].test

    if not isinstance(comparison, ast.Compare) or not isinstance(
            comparison.ops[0], ast.Eq):
        return 'Not an equality assertion'

    # Extract and return the right side of the '==' operator as a string
    return ast.get_source_segment(statement, comparison.comparators[0])


def check_testcase_output(testcase_str, expected_output):

    if len(testcase_str.splitlines()) > 1:
        for line in testcase_str.splitlines():
            if line.startswith('#'):
                continue
            if 'assert' in line:
                testcase_str = line
                break

    testcase_str = testcase_str.strip()

    if 'assert' in testcase_str:
        testcase_output_str = str(parse_assert_statement(testcase_str))

    else:
        testcase_output_str = testcase_str

    global_result = None

    try:
        testcase_output_eval = eval(testcase_output_str)
    except Exception as e:
        print(e)
        global_result = False
        # print("Failed to eval testcase output", testcase_output_str)
        # breakpoint()

    try:
        expected_output_eval = json.loads(expected_output)
    except Exception as e:
        print(e)
        global_result = False
        print('Failed to eval expected testcase output', expected_output)

    if global_result is None:
        global_result = testcase_output_eval == expected_output_eval

    return global_result


def test_output_metrics(
    samples,
    generations,
    k_list=[1, 5],
):
    num_samples = len(samples)
    results = []
    for idx in tqdm(list(range(num_samples))):
        idx_results = []
        sample = samples[idx]
        extracted_generation_list = generations[idx]
        for extracted_generation in extracted_generation_list:
            global_result = check_testcase_output(extracted_generation,
                                                  sample['output'])
            idx_results.append([global_result])
        results.append(idx_results)

    results = {
        result_idx: results[result_idx]
        for result_idx in range(len(results))
    }

    metrics = compute_metrics_from_results(results, k_list=k_list)

    return [metrics, results]


@ICL_EVALUATORS.register_module()
class LCBTestOutputEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def score(self, predictions, references):

        predictions = [[extract_test_output_code(item)]
                       for item in predictions]
        references = [json.loads(item) for item in references]
        metrics, results = test_output_metrics(references,
                                               predictions,
                                               k_list=[1])
        return metrics
