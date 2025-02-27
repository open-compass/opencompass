from opencompass.configs.models.hf_llama.modify_llama import H2OLLAMABenchmarkRunner
from opencompass.datasets import InfiniteBenchcodedebugDataset, InfiniteBenchcoderunDataset, InfiniteBenchendiaDataset, InfiniteBenchenmcDataset, InfiniteBenchenqaDataset, InfiniteBenchensumDataset, InfiniteBenchmathcalcDataset, InfiniteBenchmathfindDataset, InfiniteBenchretrievekvDataset, InfiniteBenchretrievenumberDataset, InfiniteBenchretrievepasskeyDataset, InfiniteBenchzhqaDataset
from opencompass.configs.datasets.infinitebench.infinitebenchcodedebug.infinitebench_codedebug_gen_276a42 import InfiniteBench_codedebug_reader_cfg, InfiniteBench_codedebug_infer_cfg, InfiniteBench_codedebug_eval_cfg
from opencompass.configs.datasets.infinitebench.infinitebenchcoderun.infinitebench_coderun_gen_1a76bd import InfiniteBench_coderun_reader_cfg, InfiniteBench_coderun_infer_cfg, InfiniteBench_coderun_eval_cfg
from opencompass.configs.datasets.infinitebench.infinitebenchendia.infinitebench_endia_gen_c96eb5 import InfiniteBench_endia_reader_cfg, InfiniteBench_endia_infer_cfg, InfiniteBench_endia_eval_cfg
from opencompass.configs.datasets.infinitebench.infinitebenchenmc.infinitebench_enmc_gen_3a4102 import InfiniteBench_enmc_reader_cfg, InfiniteBench_enmc_infer_cfg, InfiniteBench_enmc_eval_cfg
from opencompass.configs.datasets.infinitebench.infinitebenchenqa.infinitebench_enqa_gen_a1640c import InfiniteBench_enqa_reader_cfg, InfiniteBench_enqa_infer_cfg, InfiniteBench_enqa_eval_cfg
from opencompass.configs.datasets.infinitebench.infinitebenchensum.infinitebench_ensum_gen_cfbc08 import InfiniteBench_ensum_reader_cfg, InfiniteBench_ensum_infer_cfg, InfiniteBench_ensum_eval_cfg
from opencompass.configs.datasets.infinitebench.infinitebenchmathcalc.infinitebench_mathcalc_gen_78d17e import InfiniteBench_mathcalc_reader_cfg, InfiniteBench_mathcalc_infer_cfg, InfiniteBench_mathcalc_eval_cfg
from opencompass.configs.datasets.infinitebench.infinitebenchmathfind.infinitebench_mathfind_gen_6d799e import InfiniteBench_mathfind_reader_cfg, InfiniteBench_mathfind_infer_cfg, InfiniteBench_mathfind_eval_cfg
from opencompass.configs.datasets.infinitebench.infinitebenchretrievekv.infinitebench_retrievekv_gen_06b3ac import InfiniteBench_retrievekv_reader_cfg, InfiniteBench_retrievekv_infer_cfg, InfiniteBench_retrievekv_eval_cfg
from opencompass.configs.datasets.infinitebench.infinitebenchretrievepasskey.infinitebench_retrievepasskey_gen_62ff68 import InfiniteBench_retrievepasskey_reader_cfg, InfiniteBench_retrievepasskey_infer_cfg, InfiniteBench_retrievepasskey_eval_cfg
from opencompass.configs.datasets.infinitebench.infinitebenchretrievenumber.infinitebench_retrievenumber_gen_047436 import InfiniteBench_retrievenumber_reader_cfg, InfiniteBench_retrievenumber_infer_cfg, InfiniteBench_retrievenumber_eval_cfg
from opencompass.configs.datasets.infinitebench.infinitebenchzhqa.infinitebench_zhqa_gen_1e5293 import InfiniteBench_zhqa_reader_cfg, InfiniteBench_zhqa_infer_cfg, InfiniteBench_zhqa_eval_cfg


models = [
    dict(
        type=H2OLLAMABenchmarkRunner,
        # for represent in result
        abbr="sparse-llama-7b",
        # for huggingface
        path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(
            num_gpus=1,
            heavy_ratio=0.5,
        ),
        # modify here
        heavy_ratio=0.5,
        recent_ratio=0.1,
    )
]
datasets = [
    
    dict(
        type=InfiniteBenchcodedebugDataset,
        abbr='InfiniteBench_codedebug',
        path='./data/InfiniteBench/code_debug.jsonl',
        reader_cfg=InfiniteBench_codedebug_reader_cfg,
        infer_cfg=InfiniteBench_codedebug_infer_cfg,
        eval_cfg=InfiniteBench_codedebug_eval_cfg),
    dict(
        type=InfiniteBenchcoderunDataset,
        abbr='InfiniteBench_coderun',
        path='./data/InfiniteBench/code_run.jsonl',
        reader_cfg=InfiniteBench_coderun_reader_cfg,
        infer_cfg=InfiniteBench_coderun_infer_cfg,
        eval_cfg=InfiniteBench_coderun_eval_cfg),
    dict(
        type=InfiniteBenchendiaDataset,
        abbr='InfiniteBench_endia',
        path='./data/InfiniteBench/longdialogue_qa_eng.jsonl',
        reader_cfg=InfiniteBench_endia_reader_cfg,
        infer_cfg=InfiniteBench_endia_infer_cfg,
        eval_cfg=InfiniteBench_endia_eval_cfg),
    dict(
        type=InfiniteBenchenmcDataset,
        abbr='InfiniteBench_enmc',
        path='./data/InfiniteBench/longbook_choice_eng.jsonl',
        reader_cfg=InfiniteBench_enmc_reader_cfg,
        infer_cfg=InfiniteBench_enmc_infer_cfg,
        eval_cfg=InfiniteBench_enmc_eval_cfg),
    dict(
        type=InfiniteBenchenqaDataset,
        abbr='InfiniteBench_enqa',
        path='./data/InfiniteBench/longbook_qa_eng.jsonl',
        reader_cfg=InfiniteBench_enqa_reader_cfg,
        infer_cfg=InfiniteBench_enqa_infer_cfg,
        eval_cfg=InfiniteBench_enqa_eval_cfg),
    dict(
        type=InfiniteBenchensumDataset,
        abbr='InfiniteBench_ensum',
        path='./data/InfiniteBench/longbook_sum_eng.jsonl',
        reader_cfg=InfiniteBench_ensum_reader_cfg,
        infer_cfg=InfiniteBench_ensum_infer_cfg,
        eval_cfg=InfiniteBench_ensum_eval_cfg),
    dict(
        type=InfiniteBenchmathcalcDataset,
        abbr='InfiniteBench_mathcalc',
        path='./data/InfiniteBench/math_calc.jsonl',
        reader_cfg=InfiniteBench_mathcalc_reader_cfg,
        infer_cfg=InfiniteBench_mathcalc_infer_cfg,
        eval_cfg=InfiniteBench_mathcalc_eval_cfg),
    dict(
        type=InfiniteBenchmathfindDataset,
        abbr='InfiniteBench_mathfind',
        path='./data/InfiniteBench/math_find.jsonl',
        reader_cfg=InfiniteBench_mathfind_reader_cfg,
        infer_cfg=InfiniteBench_mathfind_infer_cfg,
        eval_cfg=InfiniteBench_mathfind_eval_cfg),
    dict(
        type=InfiniteBenchretrievekvDataset,
        abbr='InfiniteBench_retrievekv',
        path='./data/InfiniteBench/kv_retrieval.jsonl',
        reader_cfg=InfiniteBench_retrievekv_reader_cfg,
        infer_cfg=InfiniteBench_retrievekv_infer_cfg,
        eval_cfg=InfiniteBench_retrievekv_eval_cfg),
    dict(
        type=InfiniteBenchretrievenumberDataset,
        abbr='InfiniteBench_retrievenumber',
        path='./data/InfiniteBench/number_string.jsonl',
        reader_cfg=InfiniteBench_retrievenumber_reader_cfg,
        infer_cfg=InfiniteBench_retrievenumber_infer_cfg,
        eval_cfg=InfiniteBench_retrievenumber_eval_cfg),
    dict(
        type=InfiniteBenchretrievepasskeyDataset,
        abbr='InfiniteBench_retrievepasskey',
        path='./data/InfiniteBench/passkey.jsonl',
        reader_cfg=InfiniteBench_retrievepasskey_reader_cfg,
        infer_cfg=InfiniteBench_retrievepasskey_infer_cfg,
        eval_cfg=InfiniteBench_retrievepasskey_eval_cfg),
    dict(
        type=InfiniteBenchzhqaDataset,
        abbr='InfiniteBench_zhqa',
        path='./data/InfiniteBench/longbook_qa_chn.jsonl',
        reader_cfg=InfiniteBench_zhqa_reader_cfg,
        infer_cfg=InfiniteBench_zhqa_infer_cfg,
        eval_cfg=InfiniteBench_zhqa_eval_cfg)
]
