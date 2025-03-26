# The MIT License
#
# Copyright (c) OpenAI (https://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import itertools
import multiprocessing
import os
import time
from multiprocessing import Array, Value
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

from opencompass.datasets.evalplus.config import *
from opencompass.datasets.evalplus.eval._special_oracle import (
    MBPP_OUTPUT_NOT_NONE_TASKS,
    MBPP_OUTPUT_SET_EQ_TASKS,
    _digit_distance_nums,
    _poly,
    _surface_Area,
)
from opencompass.datasets.evalplus.eval.utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)


def compatible_eval_result(results: Dict) -> Dict:
    # compatibility
    for task_results in results["eval"].values():
        # update the "files" field to "nfiles"
        if "files" in task_results and "nfiles" not in task_results:
            task_results["nfiles"] = len(task_results.pop("files"))
    return results


# unbiased estimator from https://github.com/openai/human-eval
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


def query_maximum_memory_bytes() -> Optional[int]:
    # Disable functionalities that can make destructive changes to the test.
    # allow only 4GB memory usage
    maximum_memory_bytes = os.getenv(
        "EVALPLUS_MAX_MEMORY_BYTES", 4 * 1024 * 1024 * 1024
    )
    maximum_memory_bytes = min(int(maximum_memory_bytes), psutil.virtual_memory().total)
    if maximum_memory_bytes == -1:
        return None
    return maximum_memory_bytes


def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


def unsafe_execute(
    dataset: str,
    entry_point: str,
    code: str,
    inputs,
    expected: List,
    time_limits,
    atol,
    fast_check,
    stat,  # Value
    details,  # Array
    progress,  # Value
):
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        reliability_guard(maximum_memory_bytes=query_maximum_memory_bytes())
        exec_globals = {}
        try:
            with swallow_io():
                exec(code, exec_globals)
                fn = exec_globals[entry_point]

            for i, inp in enumerate(inputs):
                try:
                    with time_limit(time_limits[i]):
                        with swallow_io():
                            out = fn(*inp)

                    exp = expected[i]
                    exact_match = out == exp

                    # ================================================ #
                    # ============== special oracles ================= #
                    if dataset == "mbpp":
                        if "are_equivalent" == entry_point:  # Mbpp/164 special oracle
                            exact_match = exact_match or True
                        elif "sum_div" == entry_point:  # Mbpp/295 special oracle
                            exact_match = exact_match or out == 0
                        elif "surface_Area" == entry_point:  # Mbpp/581 special oracle
                            exact_match = (
                                exact_match or abs(out - _surface_Area(*inp)) <= atol
                            )
                        elif (
                            "digit_distance_nums" == entry_point
                        ):  # Mbpp/558 special oracle
                            exact_match = exact_match or out == _digit_distance_nums(
                                *inp
                            )
                        elif entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
                            exact_match = set(out) == set(exp)
                        elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
                            # exp is True  if not None
                            #        False if None
                            if isinstance(out, bool):
                                exact_match = out == exp
                            else:
                                exact_match = exp == (out is not None)

                    if dataset == "humaneval":
                        if "find_zero" == entry_point:
                            assert abs(_poly(*inp, out)) <= atol
                            details[i] = True
                            progress.value += 1
                            continue
                    # ============== special oracles ================= #
                    # ================================================ #

                    if atol == 0 and is_floats(exp):
                        atol = 1e-6  # enforce atol for float comparison
                    if not exact_match and atol != 0:
                        # explicitly set rtol=1e-07
                        # to match `np.testing.assert_allclose`'s default values
                        assert type(out) == type(exp)
                        if isinstance(exp, (list, tuple)):
                            assert len(out) == len(exp)
                        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
                    else:
                        assert exact_match
                except BaseException:
                    details[i] = False
                    progress.value += 1
                    if fast_check:
                        raise
                    continue

                details[i] = True
                progress.value += 1

            stat.value = _SUCCESS
        except BaseException:
            stat.value = _FAILED
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def untrusted_check(
    dataset: str,
    code: str,
    inputs: List[Any],
    entry_point: str,
    expected,
    atol,
    ref_time: List[float],
    fast_check: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> Tuple[str, np.ndarray]:
    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    timeout = min(os.getenv("EVALPLUS_TIMEOUT_PER_TASK", 60), sum(time_limits)) + 1
    if not fast_check:
        timeout += 1  # extra time for data collection

    # shared memory objects
    progress = Value("i", 0)
    stat = Value("i", _UNKNOWN)
    details = Array("b", [False for _ in range(len(inputs))])

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            dataset,
            entry_point,
            code,
            inputs,
            expected,
            time_limits,
            atol,
            fast_check,
            # return values
            stat,
            details,
            progress,
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping[stat.value]
    details = details[: progress.value]

    if not stat:
        stat = TIMEOUT

    if stat == PASS:
        if len(details) != len(inputs) or not all(details):
            stat = FAIL

    return stat, details


def evaluate_files(
    dataset: str,
    files: List[str],
    inputs: List,
    expected: List,
    entry_point: str,
    atol: float,
    ref_time: List[float],
    fast_check: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> List[Tuple[str, List[bool]]]:
    ret = []
    # sort files by the id in name (i.e., "../n.py")
    files = sorted(files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    for file in files:
        code = open(file, "r").read()
        stat, det = untrusted_check(
            dataset,
            code,
            inputs,
            entry_point,
            expected=expected,
            atol=atol,
            ref_time=ref_time,
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )
        ret.append((stat, det.tolist()))
    return ret
