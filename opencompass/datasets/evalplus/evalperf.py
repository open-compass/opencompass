"""Compute the Differential Performance Scores (DPS) and DPS_{norm} of given samples from a model.

Check our COLM paper for more details: https://www.arxiv.org/abs/2408.06450

^Updates from the COLM paper:
* Condition to activate efficiency evaluation for a task:
  * Paper: as long as you have at least one correct solution, and we select up to 10 correct solutions for efficiency sampling
  * Here: you need to have at least `min_correct` correct solutions, and we evaluate the efficiency of all correct solutions
  * Updating rationale: to make the evaluation more statistically robust

@inproceedings{liu2024evaluating,
  title = {Evaluating Language Models for Efficient Code Generation},
  author = {Liu, Jiawei and Xie, Songrun and Wang, Junhao and Wei, Yuxiang and Ding, Yifeng and Zhang, Lingming},
  booktitle = {First Conference on Language Modeling},
  year = {2024},
  url = {https://openreview.net/forum?id=IBCBMeAhmC},
}
"""

import json
import multiprocessing
import os
import socket
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional, Tuple

import rich
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from opencompass.datasets.evalplus.codegen import run_codegen
from opencompass.datasets.evalplus.config import *
from opencompass.datasets.evalplus.config import PERF_EVAL_TIMEOUT_SECOND
from opencompass.datasets.evalplus.data import (
    get_evalperf_data,
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)
from opencompass.datasets.evalplus.data.mbpp import mbpp_deserialize_inputs
from opencompass.datasets.evalplus.data.utils import stream_jsonl
from opencompass.datasets.evalplus.eval import PASS, untrusted_check
from opencompass.datasets.evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from opencompass.datasets.evalplus.evaluate import get_groundtruth
from opencompass.datasets.evalplus.perf.profile import (
    are_profiles_broken,
    default_parallelism,
    profile,
    simple_test_profiler,
)
from opencompass.datasets.evalplus.utils import progress

def rule(msg: str):
    rich.print(Rule(msg))


def not_none(l: list) -> list:
    return [x for x in l if x is not None]


def get_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def correctness_check(
    solution: str, dataset: str, task: Dict, expected_output: List
) -> Tuple:
    assert isinstance(solution, str)
    result = untrusted_check(
        dataset,
        solution,
        task["base_input"] + list(task["plus_input"]),
        task["entry_point"],
        expected_output["base"] + expected_output["plus"],
        task["atol"],
        expected_output["base_time"] + expected_output["plus_time"],
        fast_check=True,
        min_time_limit=DEFAULT_MIN_TIME_LIMIT,
        gt_time_limit_factor=DEFAULT_GT_TIME_LIMIT_FACTOR,
    )
    return result, solution


def get_evalplus_data():
    problems_he = get_human_eval_plus(noextreme=True)
    dataset_hash = get_human_eval_plus_hash(noextreme=True)
    expected_output_human = get_groundtruth(problems_he, dataset_hash, [])
    problems_mbpp = get_mbpp_plus(noextreme=True)
    dataset_hash = get_mbpp_plus_hash(noextreme=True)
    expected_output_mbpp = get_groundtruth(
        problems_mbpp,
        dataset_hash,
        MBPP_OUTPUT_NOT_NONE_TASKS,
    )
    problems = {**problems_he, **problems_mbpp}
    expected_output = {**expected_output_human, **expected_output_mbpp}
    return problems, expected_output


def table_print(table_name: str, kv: Dict):
    table = Table(
        title=table_name,
        show_header=True,
        header_style="bold",
    )
    for col_name in kv:
        table.add_column(col_name)

    table.add_row(*[str(v) for v in kv.values()])
    rich.print(table)


def correctness_worker(task_id: str, samples: list, ctask: Dict, expected_output: Dict):
    assert isinstance(
        samples, list
    ), f"{task_id}: samples is not a list but {type(samples)}"

    results = []

    for solution in samples:
        result, solution = correctness_check(
            solution, task_id.split("/")[0].lower(), ctask, expected_output
        )
        results.append(
            {
                "solution": solution,
                "pass": result[0] == PASS,
                "profiled": False,
                "matching_cluster_idx": None,
                "dps": None,
                "dps_norm": None,
            }
        )

    return task_id, results


def perf_worker(
    task_id: str,
    ptask: Dict,  # EvalPerf data
    ret_dict: Dict,
    lazy_evaluation: bool,
    max_profile: int,
):
    rich.print(f"{task_id}: Started")
    start_time = time.time()

    ######################### Profiling Setup #########################
    n_reference = len(ptask["reference"])
    entry_point = ptask["entry_point"]
    pe_input = (
        mbpp_deserialize_inputs(task_id, ptask["pe_input"])[0]
        if task_id.startswith("Mbpp/")
        else ptask["pe_input"][0]
    )
    ####################################################################

    ####################################################################
    ############### Lazily profile reference solutions #################
    ####################################################################
    cache_ref_num_inst = [None] * n_reference

    def get_avg_ref_profile(idx, check_order=True) -> Optional[Tuple]:
        nonlocal cache_ref_num_inst

        assert (
            idx < n_reference - 1
            and cache_ref_num_inst[idx + 1] is not None
            or idx == n_reference - 1
        ), f"Calling get_avg_ref_profile({idx}) before get_avg_ref_profile({idx+1}) is called, is not allowed! {n_reference = }"

        if cache_ref_num_inst[idx] is not None:
            return cache_ref_num_inst[idx], ptask["scores"][idx]

        evaluation_time = PERF_EVAL_TIMEOUT_SECOND
        ref_solution = ptask["reference"][idx]
        for _ in range(2):  # at most retry twice
            profiles = profile(
                ref_solution,
                entry_point,
                [pe_input],
                timeout_second_per_test=evaluation_time,
            )

            # Bad thing#1: timeout / failure happens
            if are_profiles_broken(profiles):
                print(f"{task_id}: [WARNING] Error in ref: {profiles}")
                rich.print(Syntax(ref_solution, "python"))
                print(f"{task_id}: Retrying w/ +10s timeout...")
                evaluation_time += 10
            else:
                break

        avg_profile = mean(profiles)
        # Bad thing#2: if the current #instruction is faster than that of i+1
        if idx < n_reference - 1 and avg_profile < cache_ref_num_inst[idx + 1]:
            print(f"{task_id}: [WARNING] #{idx} ref faster than #{idx + 1}")
            print(f"ref {idx}: #inst {avg_profile}\tscore {ptask['scores'][idx]:.1f}")
            print(
                f"ref {idx+1}: #inst {cache_ref_num_inst[idx+1]}\tscore {ptask['scores'][idx+1]:.1f}"
            )
            rich.print(Syntax(ref_solution, "python"))
            if check_order:
                return None

        cache_ref_num_inst[idx] = avg_profile
        ret_dict["ref"][idx]["_num_cpu_instructions"] = avg_profile
        return cache_ref_num_inst[idx], ptask["scores"][idx]

    ####################################################################
    ############################## END #################################
    ####################################################################

    if not lazy_evaluation:  # compute everything ahead of time
        for i in range(n_reference - 1, -1, -1):
            if get_avg_ref_profile(i) is None:
                break

        assert (
            None not in cache_ref_num_inst
        ), f"{task_id}: Failed to profile certain reference: {cache_ref_num_inst = }"

    profile_cache = {}

    cur_profiled = 0
    for result in ret_dict["results"]:
        if cur_profiled >= max_profile:
            rich.print(f"{task_id}: Reached max_profile limit {max_profile}, stopped")
            break
        if not result["pass"]:
            continue

        solution = result["solution"]

        if solution in profile_cache:  # reuse cache
            sample_profiles = profile_cache[solution]
        else:
            sample_profiles = profile(
                solution,
                entry_point,
                [pe_input],
                timeout_second_per_test=PERF_EVAL_TIMEOUT_SECOND,
            )
            profile_cache[solution] = sample_profiles  # store cache

        score = 0
        norm_score = 0
        result["matching_cluster_idx"] = -1  # -1 means even slower than the slowest ref
        # if the solution results in a timeout, score is 0
        if are_profiles_broken(sample_profiles):
            print(
                f"{task_id}: Tested solution error'ed out: {sample_profiles} ... regarded as 0 score"
            )
            rich.print(Syntax(solution, "python"))
        else:
            avg_sample_profile = result["_num_cpu_instructions"] = mean(sample_profiles)
            # Get profiles from fast to slow (back to front):
            for j in range(n_reference - 1, -1, -1):
                avg_ref_profile, ref_score = get_avg_ref_profile(j, check_order=False)
                if avg_sample_profile <= avg_ref_profile:
                    result["matching_cluster_idx"] = j
                    score = ref_score
                    norm_score = 100 * (j + 1) / n_reference
                    break

        result["dps"] = score
        result["dps_norm"] = norm_score
        result["profiled"] = True
        cur_profiled += 1

    ret_dict["dps"] = mean(not_none([r["dps"] for r in ret_dict["results"]]))
    ret_dict["dps_norm"] = mean(not_none([r["dps_norm"] for r in ret_dict["results"]]))
    ret_dict["n_profiled"] = cur_profiled

    table_print(
        f"[bold green]{task_id} Completed[/]",
        {
            "Duration": f"{time.time() - start_time:.1f}s",
            "DPS": f"[green]{ret_dict['dps']:.1f}[/]",
            "DPS_norm": f"[green]{ret_dict['dps_norm']:.1f}[/]",
            "# Profiled": f"{cur_profiled} / {len(ret_dict['results'])}",
            "Pass@1": f"{ret_dict['pass@1']:.1f}%",
        },
    )

    return ret_dict


# TODO(@ganler): OPTIMIZATION: reuse the samples from the generations of other datasets
def script(
    samples: Optional[str] = None,
    min_correct: int = 10,
    max_profile: Optional[int] = None,
    n_samples: int = 100,
    temperature: float = 1.0,
    parallel: Optional[int] = None,
    lazy_evaluation: bool = True,
    i_just_wanna_run: bool = False,
    **model_kwargs,
):
    max_profile = max_profile or min(min_correct * 2, n_samples)
    assert min_correct <= max_profile <= n_samples
    simple_test_profiler()  # test linux perf setup

    if model_kwargs:
        # To suppress the warning of tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false"
        )
        # overwrite parameters
        samples = run_codegen(
            dataset="evalperf",
            n_samples=n_samples,
            temperature=temperature,
            **model_kwargs,
        )

    assert samples is not None, "Please provide the path to the samples"

    # Data loading
    problems, expected_output = get_evalplus_data()
    ptasks = get_evalperf_data()

    # Parallelism
    max_workers = parallel or max(1, default_parallelism(divisor=4))
    assert 0 < max_workers < multiprocessing.cpu_count(), "Invalid max CPU workers"

    if os.path.isdir(samples):
        result_path = os.path.join(samples, "evalperf_results.json")
    else:
        assert samples.endswith(".jsonl")
        result_path = samples.replace(".jsonl", "_evalperf_results.json")
    brief_result_path = result_path.replace(
        "evalperf_results.json", "evalperf_results.brief.json"
    )

    # resume results
    eval_results = {}
    if not i_just_wanna_run and os.path.exists(result_path):
        resumed_result = json.load(open(result_path, "r"))
        if (
            resumed_result["n_samples"] == n_samples
            and resumed_result["temperature"] == temperature
            and resumed_result["min_correct"] == min_correct
            and resumed_result["max_profile"] == max_profile
        ):
            eval_results = resumed_result["eval"]
            for etask in eval_results:
                ptasks.pop(etask, None)

            rich.print(f"Resumed {len(eval_results)} results from {result_path}")

    # Load model's samples: task_id -> a list of samples
    sample_iter = stream_jsonl(samples)
    samples = defaultdict(list)
    for task in sample_iter:
        samples[task["task_id"].replace("_", "/")].append(task["solution"])
    samples = {k: v[:n_samples] for k, v in samples.items()}

    # assert each task has n_samples
    for task_id, s in samples.items():
        assert len(s) == n_samples, f"{task_id} has {len(s)} samples != {n_samples}"

    # Initialize eval_results
    for task_id, ptask in ptasks.items():
        eval_results[task_id] = {
            "task_id": task_id,
            "results": [],
            "ref": [
                {"solution": s, "score": r, "_num_cpu_instructions": None}
                for s, r in zip(ptask["reference"], ptask["scores"])
            ],
            "dps": None,
            "dps_norm": None,
            "pass@1": None,
            "n_profiled": None,
        }

    rule("Correctness Checking...")
    with progress("Correctness") as p:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    correctness_worker,
                    task_id,
                    samples[task_id],
                    problems[task_id],
                    expected_output[task_id],
                )
                for task_id in ptasks
            ]

            for future in p.track(as_completed(futures), total=len(futures)):
                task_id, results = future.result()
                eval_results[task_id]["results"] = results
                eval_results[task_id]["pass@1"] = (
                    100 * len([r for r in results if r["pass"]]) / n_samples
                )

    rule("EvalPerf Configurations")
    if lazy_evaluation:
        rich.print(
            "[bold yellow]Lazy evaluation is enabled[/]: "
            "Fast evaluation without enumeratively checking reference order consistency."
        )

    table_print(
        "Configurations",
        {
            "Max CPU": max_workers,
            "#Tasks": len(ptasks),
            "#Samples per task": n_samples,
            "Min correct": min_correct,
            "Max profile": max_profile,
            "Result path": result_path,
        },
    )

    rich.print(f"IDs of tasks to evaluate: {list(ptasks.keys())}")
    rule("Evaluation Start")
    undone = []
    with progress("Profiling") as p:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task_id, ptask in ptasks.items():
                n_pass = len([r for r in eval_results[task_id]["results"] if r["pass"]])
                if n_pass < min_correct:
                    rich.print(
                        f"{task_id}: [bold yellow]{n_pass} < {min_correct} correct solutions, skipped[/]"
                    )
                    continue
                futures.append(
                    executor.submit(
                        perf_worker,
                        task_id,
                        ptask,
                        eval_results[task_id],
                        lazy_evaluation,
                        max_profile,
                    )
                )
                undone.append(task_id)
                rich.print(f"{task_id}: Queued")

            for future in p.track(as_completed(futures), total=len(futures)):
                result = future.result()
                eval_results[result["task_id"]] = result
                undone.remove(result["task_id"])
                if undone and len(undone) < max_workers:
                    print(f"Still running: {undone}")

    rule("Evaluation Summary")
    dps = mean(not_none([res["dps"] for res in eval_results.values()]))
    dps_norm = mean(not_none([res["dps_norm"] for res in eval_results.values()]))
    pass_1 = mean(not_none([res["pass@1"] for res in eval_results.values()]))
    n_evalperfed = len(not_none([res["dps"] for res in eval_results.values()]))

    table_print(
        "EvalPerf Summary",
        {
            "DPS": f"{dps:.1f}",
            "DPS_norm": f"{dps_norm:.1f}",
            "Pass@1": f"{pass_1:.1f}%",
            "#EvalPerf-ed tasks": f"{n_evalperfed} / {len(eval_results)}",
            "min_correct": min_correct,
            "n_samples": n_samples,
            "temperature": temperature,
        },
    )

    # Save full results
    with open(result_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "n_samples": n_samples,
                    "temperature": temperature,
                    "min_correct": min_correct,
                    "max_profile": max_profile,
                    "eval": eval_results,
                }
            )
        )
    rich.print(f"Full results have been saved to {result_path}")

    # Save brief results
    with open(brief_result_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "config": {
                        "n_samples": n_samples,
                        "temperature": temperature,
                        "min_correct": min_correct,
                        "max_profile": max_profile,
                    },
                    "summary": {
                        "dps": dps,
                        "dps_norm": dps_norm,
                        "pass@1": pass_1,
                    },
                    "eval": {
                        task_id: {
                            "dps": res["dps"],
                            "dps_norm": res["dps_norm"],
                            "pass@1": res["pass@1"],
                            "profiled": [
                                {
                                    "solution": r["solution"],
                                    "matching_cluster_idx": r["matching_cluster_idx"],
                                }
                                for r in res["results"]
                                if r["profiled"]
                            ],
                        }
                        for task_id, res in eval_results.items()
                    },
                }
            )
        )

    rich.print(f"Brief results have been saved to {brief_result_path}")

    rule("To visualize win-rates and pair-wise DPS, run:")
    rich.print(
        Syntax(
            f"""\
git clone git@github.com:evalplus/evalplus.github.io.git
git --git-dir=evalplus.github.io/.git pull
cp {brief_result_path} evalplus.github.io/results/evalperf
python evalplus.github.io/results/evalperf/stats.py
python -m http.server -d evalplus.github.io {get_free_port()}""",
            "bash",
        )
    )


def main():
    from fire import Fire

    Fire(script)


if __name__ == "__main__":
    main()
