import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from traceback import format_exc
from typing import Any, List, Optional, Tuple

from pympler.asizeof import asizeof
from rich.syntax import Syntax
from termcolor import colored

from evalplus.config import PERF_CURATE_TIMEOUT_SECOND, PERF_RAM_GB_PER_PROC
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.data.mbpp import mbpp_serialize_inputs
from evalplus.eval.utils import TimeoutException, reliability_guard, time_limit
from evalplus.sanitize import syntax_check
from evalplus.utils import progress


# this is more of a hack... rather than a "verified" implementation
def insert_contract(entry_point: str, code: str, contract: str):
    # why is this so complicated? because the contract might be mis-indented...
    def get_first_indent_size(source, body_char_start_idx):
        assert source.strip()
        indent_size = 0
        while source[body_char_start_idx - indent_size - 1] == " ":
            indent_size += 1
        return indent_size

    code = code.replace("\t", " " * 4)
    contract = contract.replace("\t", " " * 4)

    lines = [line for line in code.split("\n") if line.strip()]
    fn_def_line = [line for line in lines if line.startswith(f"def {entry_point}")][0]
    def_line_idx = lines.index(fn_def_line)
    body_start_idx = code.index(code.split(fn_def_line)[1].lstrip())

    source_indent: int = get_first_indent_size(code, body_start_idx)
    contract_indent: int = get_first_indent_size(
        contract, len(contract) - len(contract.lstrip())
    )
    return "\n".join(
        lines[: def_line_idx + 1]
        + [
            " " * max(0, source_indent - contract_indent) + cline
            for cline in contract.split("\n")
            if cline
        ]
        + [
            " " * max(0, contract_indent - source_indent) + sline
            for sline in lines[def_line_idx + 1 :]
            if sline
        ]
    )


def post_process(text: str) -> Optional[str]:
    """Post-process the LLM generated text to make it valid."""
    if "\n```" not in text:
        return None

    # split ```python3 or ```python
    text = re.split(r"\n```python3?\n", text)[1]
    text = text.split("\n```")[0].strip()

    # perform syntax check
    if not syntax_check(text):
        print(colored("⚠️ Syntax check failed for the code below:", "red"))
        print(text[:256], "..." if len(text) > 256 else "")
        return None

    return text


# returns:
# 1. generated and validated (by the contract) inputs
# 2. whether the generator stops in a well-defined manner
#    -- if False, we might want to try another generator
def sample_one_input(
    ref_code_with_contract: str,
    entry_point: str,
    generator_code: str,
    timeout_second: float = PERF_CURATE_TIMEOUT_SECOND + 1,
) -> Tuple[List[Any], bool]:
    # These system calls are needed when cleaning up tempdir.
    import os
    import shutil

    rmtree = shutil.rmtree
    rmdir = os.rmdir
    chdir = os.chdir
    # Disable functionalities that can make destructive changes to the test.
    # :imit memory usages.
    maximum_memory_bytes = PERF_RAM_GB_PER_PROC * 1024 * 1024 * 1024
    reliability_guard(maximum_memory_bytes=maximum_memory_bytes)
    exec_globals = {}

    # eval the func def with contract
    exec(ref_code_with_contract, exec_globals)
    fn = exec_globals[entry_point]

    # eval the generator
    generator_code = "from typing import *\n" + generator_code
    try:
        exec(generator_code, exec_globals)
        generator = exec_globals["perf_input_gen"]
    except Exception:
        print(colored(f"⚠️ [GEN EVAL] Exception ~ {entry_point}:", "red"))
        print(colored(format_exc(), "red"))
        return [], False

    well_defined_exit = True
    return_inputs = []

    for fac in range(1, 27):
        scale = 2**fac
        print(f"[INPUT GEN] scale=2**{fac}")
        try:
            with time_limit(timeout_second):
                test_input = generator(scale)
            if not isinstance(test_input, tuple):
                test_input = (test_input,)
            # integers should stay in the range of 64-bit
            if any(
                isinstance(arg, int) and not (-(2**63) <= arg < 2**63)
                for arg in test_input
            ):
                print(colored(f"[INPUT GEN] Int overflow against 64bit", "yellow"))
                break
            # hack list integer
            if isinstance(test_input[0], list) and any(
                not (-(2**63) <= v < 2**63)
                for v in test_input[0]
                if isinstance(v, int)
            ):
                print(colored(f"[INPUT GEN] Int overflow against 64bit", "yellow"))
                break
            # stop here if the input is of 64M.
            INPUT_LIMIT_MB = 64
            if asizeof(test_input) > 1024 * 1024 * INPUT_LIMIT_MB:
                print(colored(f"[INPUT GEN] Size > {INPUT_LIMIT_MB}MB", "yellow"))
                break
        except TimeoutException:
            print(colored(f"[INPUT GEN] TimeoutException at scale=2**{fac}", "yellow"))
            break
        except MemoryError:
            print(colored(f"[INPUT GEN] MemoryError at scale=2**{fac}", "yellow"))
            break
        except Exception:
            print(colored(f"⚠️ [INPUT GEN] Exception at scale=2**{fac}", "red"))
            print(colored(format_exc(), "red"))
            well_defined_exit = False
            break

        try:
            with time_limit(timeout_second):
                # deepcopy in case fn modifies the input
                fn(*deepcopy(test_input))
            return_inputs = [test_input]  # only keep on input
        except TimeoutException:
            print(colored(f"[Testing] Timeout at scale=2**{fac}", "yellow"))
            break
        except MemoryError:
            print(colored(f"[Testing] MemoryError at scale=2**{fac}", "yellow"))
            break
        except Exception:
            print(colored(f"⚠️ [Testing] Exception ~ {entry_point}", "red"))
            print(colored(format_exc(), "red"))
            well_defined_exit = False
            break

    # Needed for cleaning up.
    shutil.rmtree = rmtree
    os.rmdir = rmdir
    os.chdir = chdir

    return return_inputs, well_defined_exit


def main(input: str, output: str):
    """In the synthesizer file, each line includes a set of input generators for a task.
    The goal of this script is to use these generators to sample inputs for each task.
    The generated inputs are expected to be valid.
    """
    assert output.endswith(".jsonl"), "output must be a .jsonl file"

    id2task = {}
    for task_id, item in get_human_eval_plus().items():
        id2task[task_id] = item

    for task_id, item in get_mbpp_plus().items():
        id2task[task_id] = item

    # loading the synthesizers
    with open(input, "r") as f:
        synthesizers = [json.loads(l) for l in f]

    n_total = 0
    n_parsed = 0
    n_dedup = 0

    for item in synthesizers:
        item["synthesizers"] = [post_process(s) for s in item["synthesizers"]]
        n_total += len(item["synthesizers"])
        item["synthesizers"] = [s for s in item["synthesizers"] if s is not None]
        n_parsed += len(item["synthesizers"])

        dedup_set = set()
        for s in item["synthesizers"]:
            dedup_set.add(
                "\n".join(
                    [l for l in s.splitlines() if l.strip() and not l.startswith("#")]
                )
            )
        item["synthesizers"] = list(dedup_set)
        n_dedup += len(item["synthesizers"])

    print(
        colored(
            f"#Total {n_total} with {n_parsed} parsed => {100 * (1 - n_parsed / n_total) :.1f}% syntax err",
            "green",
        )
    )

    print(
        colored(
            f"#Parsed {n_parsed} with {n_dedup} dedup => {100 * (1 - n_dedup / n_parsed) :.1f}% duplicate",
            "green",
        )
    )

    # resume mode check finished tasks
    finished_tasks = set()
    if os.path.isfile(output):
        with open(output, "r") as f:
            for l in f:
                item = json.loads(l)
                finished_tasks.add(item["task_id"])

    print("Resumed finished tasks:", finished_tasks)
    with open(output, "ab+") as f:
        with progress() as p:
            for item in p.track(synthesizers):
                task_id = item["task_id"]
                entry_point = id2task[task_id]["entry_point"]
                if task_id in finished_tasks:
                    p.console.print(f"{task_id}: {entry_point} ~ Resumed")
                    continue

                ref_code_with_contract = insert_contract(
                    entry_point, item["ref_code"], id2task[task_id]["contract"]
                )
                p.console.print(f"{task_id}: PE input generation...")
                p.console.print(Syntax(ref_code_with_contract.strip(), "python"))

                results = []
                for i, generator_code in enumerate(item["synthesizers"]):
                    p.console.print(
                        f"Using generator {i+1}/{len(item['synthesizers'])}:"
                    )
                    p.console.print(Syntax(generator_code, "python"))
                    args = (
                        ref_code_with_contract,
                        entry_point,
                        generator_code,
                    )
                    with ProcessPoolExecutor(max_workers=1) as executor:
                        tmp_results, status = executor.submit(
                            sample_one_input, *args
                        ).result()

                    results.extend(tmp_results)

                    # if the func returns in a well-defined manner, we can stop here.
                    if status:
                        break

                p.console.print("Serializing and storing results...")

                if "Mbpp/" in task_id:
                    results = mbpp_serialize_inputs(task_id, results)

                to_write = {"task_id": item["task_id"], "inputs": results}
                to_write = (json.dumps(to_write) + "\n").encode("utf-8")

                # task_id => list of inputs
                f.write(to_write)
                f.flush()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
