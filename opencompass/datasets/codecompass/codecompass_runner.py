import os

from .executor import LocalExecutor


def run_test_for_cpp_problem(sample: dict,
                             generations: list,
                             timeout: int,
                             memory_limit_mb: int,
                             temp_base_dir='tmp') -> list:
    pid = os.getpid()
    print(f'\n--- [DEBUG][PID:{pid}] Subprocess started. '
          f'Timeout: {timeout}s, MemLimit: {memory_limit_mb}MB ---')

    try:
        eval_data = sample['evaluation_sample']
        assert isinstance(
            eval_data, dict), f'eval_data is not a dict, but {type(eval_data)}'

        inputs = eval_data.get('inputs', [])
        outputs = eval_data.get('outputs', [])
    except Exception:
        return [[-4] * 100 for _ in generations]

    executor = LocalExecutor(timeout=timeout,
                             memory_limit_mb=memory_limit_mb,
                             temp_base_dir=temp_base_dir)

    all_results = []

    for gen_idx, gen_code in enumerate(generations):
        if not gen_code or not gen_code.strip():
            all_results.append([-2] * len(inputs))
            continue

        try:
            results_for_this_gen = []

            for i in range(len(inputs)):
                result = executor.submit_code(source_code=gen_code,
                                              stdin=inputs[i],
                                              expected_output=outputs[i],
                                              language='C++')

                status = result.get('status', {}).get('description',
                                                      'Unknown Error')

                if status == 'Accepted':
                    results_for_this_gen.append(1)
                elif status == 'Compilation Error':
                    results_for_this_gen = [-2] * len(inputs)
                    break
                elif status == 'Memory Limit Exceeded':
                    results_for_this_gen.append(-5)
                else:
                    results_for_this_gen.append(-1)

            all_results.append(results_for_this_gen)

        except Exception:
            all_results.append([-3] * len(inputs))

    return all_results
