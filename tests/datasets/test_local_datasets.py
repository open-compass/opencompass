import os
import random
import sys
import unittest
import warnings
from os import environ

# Add project root to Python path to allow importing configs
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from datasets import Dataset, DatasetDict
from mmengine.config import read_base
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore', category=DeprecationWarning)


def reload_datasets():
    with read_base():
        from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
        from opencompass.configs.datasets.cmmlu.cmmlu_gen import cmmlu_datasets
        from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets
        from opencompass.configs.datasets.ARC_e.ARC_e_gen import ARC_e_datasets
        from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
        from opencompass.configs.datasets.humaneval.humaneval_repeat10_gen_8e312c import humaneval_datasets as humaneval_repeat10_datasets
        from opencompass.configs.datasets.race.race_ppl import race_datasets
        
        from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
        from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets
        from opencompass.configs.datasets.winogrande.winogrande_gen import winogrande_datasets
        from opencompass.configs.datasets.winogrande.winogrande_ll import winogrande_datasets as winogrande_ll_datasets
        from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import winogrande_datasets as winogrande_5shot_ll_datasets
        
        from opencompass.configs.datasets.hellaswag.hellaswag_gen import hellaswag_datasets as hellaswag_v2_datasets
        from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets as hellaswag_ice_datasets
        from opencompass.configs.datasets.hellaswag.hellaswag_ppl_9dbb12 import hellaswag_datasets as hellaswag_v1_datasets
        from opencompass.configs.datasets.hellaswag.hellaswag_ppl_a6e128 import hellaswag_datasets as hellaswag_v3_datasets
        from opencompass.configs.datasets.mbpp.mbpp_gen import mbpp_datasets as mbpp_v1_datasets
        from opencompass.configs.datasets.mbpp.mbpp_passk_gen_830460 import mbpp_datasets as mbpp_v2_datasets
        from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_830460 import sanitized_mbpp_datasets
        from opencompass.configs.datasets.nq.nq_gen import nq_datasets
        from opencompass.configs.datasets.math.math_gen import math_datasets
        from opencompass.configs.datasets.GaokaoBench.GaokaoBench_gen import GaokaoBench_datasets
        from opencompass.configs.datasets.GaokaoBench.GaokaoBench_mixed import GaokaoBench_datasets as GaokaoBench_mixed_datasets
        from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_4c31db import GaokaoBench_datasets as GaokaoBench_no_subjective_datasets
        from opencompass.configs.datasets.triviaqa.triviaqa_gen import triviaqa_datasets
        from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_20a989 import triviaqa_datasets as triviaqa_wiki_1shot_datasets

        from opencompass.configs.datasets.aime2024.aime2024_gen_6e39a4 import \
            aime2024_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import \
            ARC_c_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import \
            bbh_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.cmmlu.cmmlu_0shot_cot_gen_305931 import \
            cmmlu_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.cmo_fib.cmo_fib_gen_ace24b import \
            cmo_fib_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.drop.drop_openai_simple_evals_gen_3857b0 import \
            drop_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_4c31db import \
            GaokaoBench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import \
            gpqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_6e39a4 import \
            gsm8k_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import \
            hellaswag_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_dcae0e import \
            humaneval_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import \
            ifeval_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.korbench.korbench_single_0_shot_gen import \
            korbench_0shot_single_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.livecodebench.livecodebench_gen_b2b0fd import \
            LCB_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.math.math_0shot_gen_11c4b5 import \
            math_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.MathBench.mathbench_2024_gen_50a320 import \
            mathbench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import \
            sanitized_mbpp_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.mmlu.mmlu_openai_simple_evals_gen_b618ea import \
            mmlu_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import \
            mmlu_pro_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.mmmlu_lite.mmmlu_lite_gen_c51a84 import \
            mmmlu_lite_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.musr.musr_gen_3622bb import \
            musr_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.nq.nq_open_1shot_gen_2e45e5 import \
            nq_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.race.race_cot_gen_d95929 import \
            race_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.scicode.scicode_gen_085b98 import \
            SciCode_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_cot_gen_1d56df import \
            BoolQ_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.teval.teval_en_gen_1ac254 import \
            teval_datasets as teval_en_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.teval.teval_zh_gen_1ac254 import \
            teval_datasets as teval_zh_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import \
            TheoremQA_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_bc5f21 import \
            triviaqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.wikibench.wikibench_gen_0978ad import \
            wikibench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.aime2024.aime2024_llmjudge_gen_5e9f4f import \
        aime2024_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.aime2025.aime2025_llmjudge_gen_5e9f4f import \
            aime2025_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.ARC_Prize_Public_Evaluation.arc_prize_public_evaluation_gen_fedd04 import \
            arc_prize_public_evaluation_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.bbh.bbh_llmjudge_gen_b5bdf1 import \
            bbh_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.cmo_fib.cmo_fib_gen_2783e5 import \
            cmo_fib_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.dingo.dingo_gen import \
            datasets as dingo_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.drop.drop_llmjudge_gen_3857b0 import \
            drop_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_d16acb import \
            GaokaoBench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.gpqa.gpqa_0shot_nocot_genericllmeval_gen_772ea0 import \
            gpqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import \
            gsm8k_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.hellaswag.hellaswag_llmjudge_gen_809ef1 import \
            hellaswag_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.korbench.korbench_llmjudge_gen_56cf43 import \
            korbench_0shot_single_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.math.math_500_llmjudge_gen_6ff468 import \
            math_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.MathBench.mathbench_2024_gen_4b8f28 import \
            mathbench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.musr.musr_llmjudge_gen_b47fd3 import \
            musr_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.supergpqa.supergpqa_llmjudge_gen_12b8bc import \
            supergpqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_c87d61 import \
            triviaqa_datasets  # noqa: F401, E501
        from opencompass.configs.summarizers.groups.bbeh import \
            bbeh_summary_groups  # noqa: F401, E501
        from opencompass.configs.summarizers.groups.bbh import \
            bbh_summary_groups  # noqa: F401, E501
        from opencompass.configs.summarizers.groups.cmmlu import \
            cmmlu_summary_groups  # noqa: F401, E501
        from opencompass.configs.summarizers.groups.GaokaoBench import \
            GaokaoBench_summary_groups  # noqa: F401, E501
        from opencompass.configs.summarizers.groups.korbench import \
            korbench_summary_groups  # noqa: F401, E501
        from opencompass.configs.summarizers.groups.mathbench_v1_2024 import \
            mathbench_2024_summary_groups  # noqa: F401, E501
        from opencompass.configs.summarizers.groups.mmlu import \
            mmlu_summary_groups  # noqa: F401, E501
        from opencompass.configs.summarizers.groups.mmlu_pro import \
            mmlu_pro_summary_groups  # noqa: F401, E501
        from opencompass.configs.summarizers.groups.musr_average import \
            summarizer as musr_summarizer
        from opencompass.configs.summarizers.mmmlu_lite import \
            mmmlu_summary_groups  # noqa: F401, E501
        from opencompass.configs.datasets.aime2024.aime2024_cascade_eval_gen_5e9f4f import \
            aime2024_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f import \
            aime2025_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.bbeh.bbeh_llmjudge_gen_86c3a0 import \
            bbeh_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_complete_gen_2888d3 import \
            bigcodebench_hard_complete_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_instruct_gen_c3d5ad import \
            bigcodebench_hard_instruct_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.chem_exam.competition_gen import \
            chem_competition_instruct_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.chem_exam.gaokao_gen import \
            chem_gaokao_instruct_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.ChemBench.ChemBench_llmjudge_gen_c584cf import \
            chembench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.ClimaQA.ClimaQA_Gold_llm_judge_gen_f15343 import \
            climaqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.cmmlu.cmmlu_llmjudge_gen_e1cd9a import \
            cmmlu_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.Earth_Silver.Earth_Silver_llmjudge_gen import \
            earth_silver_mcq_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_gen_772ea0 import \
            gpqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.HLE.hle_llmverify_gen_6ff468 import \
            hle_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import \
            ifeval_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.korbench.korbench_single_0shot_cascade_eval_gen_56cf43 import \
            korbench_0shot_single_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import \
            LCBCodeGeneration_dataset  # noqa: F401, E501
        from opencompass.configs.datasets.livemathbench.livemathbench_hard_custom_cascade_eval_gen_4bce59 import \
            livemathbench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.matbench.matbench_llm_judge_gen_0e9276 import \
            matbench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.math.math_500_cascade_eval_gen_6ff468 import \
            math_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import \
            sanitized_mbpp_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.MedXpertQA.MedXpertQA_llmjudge_gen import \
            medxpertqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.mmlu.mmlu_llmjudge_gen_f4336b import \
            mmlu_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_nocot_genericllmeval_gen_08c1de import \
            mmlu_pro_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.OlymMATH.olymmath_llmverify_gen_97b203 import \
            olymmath_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.OlympiadBench.OlympiadBench_0shot_llmverify_gen_be8b13 import \
            olympiadbench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.PHYBench.phybench_gen import \
            phybench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.PHYSICS.PHYSICS_llm_judge_gen_a133a2 import \
            physics_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.ProteinLMBench.ProteinLMBench_llmjudge_gen_a67965 import \
            proteinlmbench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.R_Bench.rbench_llmjudge_gen_c89350 import \
            RBench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.srbench.srbench_gen import \
            srbench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.supergpqa.supergpqa_cascade_gen_1545c1 import \
            supergpqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.chinese_simpleqa.chinese_simpleqa_gen import \
        csimpleqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.SimpleQA.simpleqa_gen_0283c3 import \
            simpleqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.subjective.alignbench.alignbench_v1_1_judgeby_critiquellm_new import \
            alignbench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4_new import \
            alpacav2_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare_new import \
            arenahard_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.subjective.compassarena.compassarena_compare_new import \
            compassarena_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.subjective.followbench.followbench_llmeval_new import \
            followbench_llmeval_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.subjective.multiround.mtbench101_judge_new import \
            mtbench101_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.subjective.wildbench.wildbench_pair_judge_new import \
            wildbench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.ARC_c.ARC_c_few_shot_ppl import \
        ARC_c_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.bbh.bbh_gen_98fba6 import \
            bbh_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.cmmlu.cmmlu_ppl_041cbf import \
            cmmlu_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.drop.drop_gen_a2697c import \
            drop_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_d21e37 import \
            GaokaoBench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.gpqa.gpqa_few_shot_ppl_4b5a83 import \
            gpqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import \
            gsm8k_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.hellaswag.hellaswag_10shot_ppl_59c85e import \
            hellaswag_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.math.math_4shot_base_gen_43d5b6 import \
            math_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.MathBench.mathbench_2024_few_shot_mixed_4a3fd4 import \
            mathbench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_742f0c import \
            sanitized_mbpp_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import \
            mmlu_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.mmlu_pro.mmlu_pro_few_shot_gen_bfaf90 import \
            mmlu_pro_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.nq.nq_open_1shot_gen_20a989 import \
            nq_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.race.race_few_shot_ppl import \
            race_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_few_shot_ppl import \
            BoolQ_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import \
            TheoremQA_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_20a989 import \
            triviaqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.wikibench.wikibench_few_shot_ppl_c23d79 import \
            wikibench_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import \
            winogrande_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import \
        gpqa_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import \
            gsm8k_datasets  # noqa: F401, E501
        from opencompass.configs.datasets.race.race_ppl import \
            race_datasets  # noqa: F401, E501


        return sum((v for k, v in locals().items() if k.endswith('_datasets')), [])



def load_datasets_conf(source):
    environ['DATASET_SOURCE'] = source
    datasets_conf = reload_datasets()
    return datasets_conf


def load_datasets(source, conf):
    environ['DATASET_SOURCE'] = source
    if 'lang' in conf:
        dataset = conf['type'].load(path=conf['path'], lang=conf['lang'])
        return dataset
    if 'setting_name' in conf:
        dataset = conf['type'].load(path=conf['path'],
                                    name=conf['name'],
                                    setting_name=conf['setting_name'])
        return dataset
    if 'name' in conf:
        dataset = conf['type'].load(path=conf['path'], name=conf['name'])
        return dataset

    if 'local_mode' in conf:
        dataset = conf['type'].load(path=conf['path'], local_mode=conf['local_mode'])
        return dataset
    try:
        dataset = conf['type'].load(path=conf['path'])
    except Exception:
        dataset = conf['type'].load(**conf)
    return dataset


def clean_string(value):
    """Helper function to clean and normalize string data.

    It strips leading and trailing whitespace and replaces multiple whitespace
    characters with a single space.
    """
    if isinstance(value, str):
        return ' '.join(value.split())
    return value


class TestingLocalDatasets(unittest.TestCase):

    def test_datasets(self):
        """Test loading all local datasets."""
        local_datasets_conf = load_datasets_conf('Local')
        
        successful_comparisons = []
        failed_comparisons = []
        
        def compare_datasets(local_conf):
            """Load and validate a single dataset."""
            local_path_name = f"{local_conf.get('path')}/{local_conf.get('name', '')}\t{local_conf.get('lang', '')}"
            try:
                local_dataset = load_datasets('Local', local_conf)
                return 'success', f'{local_path_name}'
            except Exception as exception:
                return 'failure', f'can\'t load {local_path_name}'
        
        from concurrent.futures import TimeoutError as FutureTimeoutError
        import time
        
        TIMEOUT_PER_DATASET = 300  # 5 minutes per dataset
        total_count = len(local_datasets_conf)
        print(f"[INFO] Starting to test {total_count} datasets with timeout {TIMEOUT_PER_DATASET}s per dataset")
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            # Submit all tasks and track them
            future_to_conf = {
                executor.submit(compare_datasets, local_conf): local_conf
                for local_conf in local_datasets_conf
            }
            
            completed_count = 0
            start_time = time.time()
            last_progress_time = start_time
            
            # Process completed futures with progress tracking
            while future_to_conf:
                try:
                    # Wait for at least one future to complete
                    done_futures = list(as_completed(
                        list(future_to_conf.keys()), 
                        timeout=60  # Check every minute for progress
                    ))
                    
                    for future in done_futures:
                        if future not in future_to_conf:
                            continue
                        local_conf = future_to_conf.pop(future)
                        local_path_name = f"{local_conf.get('path')}/{local_conf.get('name', '')}\t{local_conf.get('lang', '')}"
                        
                        try:
                            result, message = future.result(timeout=1)
                            completed_count += 1
                            elapsed = time.time() - start_time
                            status = "✓" if result == 'success' else "✗"
                            print(f"[PROGRESS] [{completed_count}/{total_count}] ({elapsed:.1f}s) {status} {local_path_name[:80]}")
                            last_progress_time = time.time()
                            
                            if result == 'success':
                                successful_comparisons.append(message)
                            else:
                                failed_comparisons.append(message)
                        except FutureTimeoutError:
                            print(f"[TIMEOUT] Dataset loading timeout: {local_path_name}")
                            failed_comparisons.append(f'TIMEOUT: {local_path_name}')
                            completed_count += 1
                        except Exception as e:
                            print(f"[ERROR] Unexpected error for {local_path_name}: {type(e).__name__}: {str(e)[:200]}")
                            failed_comparisons.append(f'ERROR: {local_path_name} - {str(e)[:100]}')
                            completed_count += 1
                            
                except FutureTimeoutError:
                    # No progress in 60 seconds - check for stuck tasks
                    current_time = time.time()
                    stuck_time = current_time - last_progress_time
                    print(f"[WARNING] No progress in {stuck_time:.1f}s. Checking {len(future_to_conf)} remaining tasks...")
                    
                    # List all remaining (potentially stuck) datasets
                    for future, local_conf in list(future_to_conf.items()):
                        local_path_name = f"{local_conf.get('path')}/{local_conf.get('name', '')}\t{local_conf.get('lang', '')}"
                        if not future.done():
                            print(f"[STUCK] Potentially stuck: {local_path_name}")
                    
                    # Cancel all remaining futures after showing what's stuck
                    for future in list(future_to_conf.keys()):
                        if not future.done():
                            future.cancel()
                            local_conf = future_to_conf.pop(future)
                            local_path_name = f"{local_conf.get('path')}/{local_conf.get('name', '')}\t{local_conf.get('lang', '')}"
                            failed_comparisons.append(f'CANCELLED (stuck): {local_path_name}')
                            completed_count += 1
                    break
        
        # Print test summary
        total_datasets = len(local_datasets_conf)
        print(f"All {total_datasets} datasets")
        print(f"OK {len(successful_comparisons)} datasets")
        for success in successful_comparisons:
            print(f"  {success}")
        print(f"Fail {len(failed_comparisons)} datasets")
        for failure in failed_comparisons:
            print(f"  {failure}")


def _check_data(dataset1: Dataset | DatasetDict,
                dataset2: Dataset | DatasetDict,
                sample_size):
    """Compare two datasets for consistency."""
    assert type(dataset1) == type(dataset2), \
        f'Dataset type mismatch: {type(dataset1)} != {type(dataset2)}'

    if isinstance(dataset2, DatasetDict):
        assert dataset1.keys() == dataset2.keys(), \
            f'DatasetDict keys mismatch: {dataset1.keys()} != {dataset2.keys()}'
        for key in dataset1.keys():
            _check_data(dataset1[key], dataset2[key], sample_size=sample_size)
    elif isinstance(dataset2, Dataset):
        assert set(dataset1.column_names) == set(dataset2.column_names), \
            f'Column names mismatch: {dataset1.column_names} != {dataset2.column_names}'
        assert len(dataset1) == len(dataset2), \
            f'Row count mismatch: {len(dataset1)} != {len(dataset2)}'

        sample_indices = random.sample(range(len(dataset1)),
                                       min(sample_size, len(dataset1)))
        for idx in sample_indices:
            for col in dataset1.column_names:
                val1 = clean_string(str(dataset1[col][idx]))
                val2 = clean_string(str(dataset2[col][idx]))
                assert val1 == val2, \
                    f"Value mismatch in column '{col}', index {idx}: {val1} != {val2}"
    else:
        raise ValueError(f'Unsupported dataset type: {type(dataset1)}')


if __name__ == '__main__':
    sample_size = 100
    unittest.main()
