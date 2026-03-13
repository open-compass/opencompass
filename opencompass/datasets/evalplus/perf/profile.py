import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Value, cpu_count
from platform import system
from time import perf_counter
from traceback import format_exc
from typing import Any, Callable, List, Optional

import psutil
from cirron import Collector

from evalplus.config import PERF_PROFILE_ROUNDS, PERF_RAM_GB_PER_PROC
from evalplus.eval.utils import (
    TimeoutException,
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)


def get_max_ram_gb():
    total_ram = psutil.virtual_memory().total
    return total_ram / (1024**3)


def default_parallelism(divisor=4):
    return max(1, max(cpu_count(), get_max_ram_gb() // PERF_RAM_GB_PER_PROC) // divisor)


def simple_test_profiler():
    # assert linux
    assert system() == "Linux", "EvalPerf requires Linux's perf_event_open"
    try:
        with Collector():
            pass
    except Exception as e:
        print("It seems your system does not support instruction counting.")
        print("Try this on Linux:")
        print("   sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'   ")
        print("Also check more info at: https://github.com/s7nfo/Cirron")
        print("Re-raising the original exception...")
        raise e


def are_profiles_broken(profiles) -> bool:
    return not all(isinstance(profile, (float, int)) for profile in profiles)


def physical_runtime_profiler(function, test_inputs) -> float:
    start = perf_counter()
    for test_input in test_inputs:
        function(*test_input)
    return perf_counter() - start


def num_instruction_profiler(function, test_inputs) -> int:
    with Collector() as c:
        for test_input in test_inputs:
            function(*test_input)
    return int(c.counters.instruction_count)


_STAT_NONE = 0
_STAT_START = 1
_STAT_SUCC = 2
_STAT_ERROR = 3


def get_instruction_count_shared_mem(
    profiler: Callable,
    func_code: str,
    entry_point: str,
    test_inputs: List[Any],
    timeout_second_per_test: float,
    memory_bound_gb: int,
    warmup_inputs: Optional[List[Any]],
    # shared memory
    compute_cost,  # Value("d", 0.0),
    progress,  # Value("i", 0),
) -> Optional[float]:

    error = None

    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        maximum_memory_bytes = memory_bound_gb * 1024 * 1024 * 1024
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)
        exec_globals = {}

        # run (eval) the func def
        exec(func_code, exec_globals)
        fn = exec_globals[entry_point]

        # warmup the function
        if warmup_inputs:
            for _ in range(3):
                fn(*warmup_inputs)

        progress.value = _STAT_START
        try:  # run the function
            with time_limit(timeout_second_per_test):
                with swallow_io():
                    compute_cost.value = profiler(fn, test_inputs)
                    progress.value = _STAT_SUCC
        except TimeoutException:
            print("[Warning] Profiling hits TimeoutException")
        except MemoryError:
            print("[Warning] Profiling hits MemoryError")
        except:
            print("[CRITICAL] ! Unknown exception during profiling !")
            error = format_exc()
            print(error)

        if progress.value != _STAT_SUCC:
            progress.value = _STAT_ERROR

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def profile(
    func_code: str,
    entry_point: str,
    test_inputs: List[Any],
    timeout_second_per_test: float,
    memory_bound_gb: int = PERF_RAM_GB_PER_PROC,
    profile_rounds: int = PERF_PROFILE_ROUNDS,
    profiler: Callable = num_instruction_profiler,
    warmup_inputs: Optional[List[Any]] = None,  # multiple inputs
) -> List[int | float | str]:
    """Profile the func_code against certain input tests.
    The function code is assumed to be correct and if a string is returned, it is an error message.
    """
    timeout = timeout_second_per_test * len(test_inputs) * profile_rounds

    def _run():
        compute_cost = Value("d", 0.0)
        progress = Value("i", _STAT_NONE)

        p = Process(
            target=get_instruction_count_shared_mem,
            args=(
                profiler,
                func_code,
                entry_point,
                test_inputs,
                timeout_second_per_test,
                memory_bound_gb,
                warmup_inputs,
                # shared memory
                compute_cost,
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

        if progress.value == _STAT_SUCC:
            return compute_cost.value
        elif progress.value == _STAT_NONE:
            return "PROFILING DID NOT START"
        elif progress.value == _STAT_ERROR:
            return "SOLUTION ERROR ENCOUNTERED WHILE PROFILING"

    return [_run() for _ in range(profile_rounds)]
