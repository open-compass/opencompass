DATASETS_MAPPING = {
    # ADVGLUE Datasets
    "opencompass/advglue-dev": {
        "ms_id": None,
        "hf_id": None,
        "local": "./data/adv_glue/dev_ann.json",
    },
    # AGIEval Datasets
    "opencompass/agieval": {
        "ms_id": "opencompass/agieval",
        "hf_id": "opencompass/agieval",
        "local": "./data/AGIEval/data/v1/",
    },
    # ARC Datasets(Test)
    "opencompass/ai2_arc-test": {
        "ms_id": "opencompass/ai2_arc",
        "hf_id": "opencompass/ai2_arc",
        "local": "./data/ARC/ARC-c/ARC-Challenge-Test.jsonl",
    },
    "opencompass/ai2_arc-dev": {
        "ms_id": "opencompass/ai2_arc",
        "hf_id": "opencompass/ai2_arc",
        "local": "./data/ARC/ARC-c/ARC-Challenge-Dev.jsonl",
    },
    "opencompass/ai2_arc-easy-dev": {
        "ms_id": "opencompass/ai2_arc",
        "hf_id": "opencompass/ai2_arc",
        "local": "./data/ARC/ARC-e/ARC-Easy-Dev.jsonl",
    },
    # BBH
    "opencompass/bbh": {
        "ms_id": "opencompass/bbh",
        "hf_id": "opencompass/bbh",
        "local": "./data/BBH/data",
    },
    # bbeh
    "opencompass/bbeh": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/bbeh/",
    },
    # C-Eval
    "opencompass/ceval-exam": {
        "ms_id": "opencompass/ceval-exam",
        "hf_id": "opencompass/ceval-exam",
        "local": "./data/ceval/formal_ceval",
    },
    # AFQMC
    "opencompass/afqmc-dev": {
        "ms_id": "opencompass/afqmc",
        "hf_id": "opencompass/afqmc",
        "local": "./data/CLUE/AFQMC/dev.json",
    },
    # CMNLI
    "opencompass/cmnli-dev": {
        "ms_id": "opencompass/cmnli",
        "hf_id": "opencompass/cmnli",
        "local": "./data/CLUE/cmnli/cmnli_public/dev.json",
    },
    # OCNLI
    "opencompass/OCNLI-dev": {
        "ms_id": "opencompass/OCNLI",
        "hf_id": "opencompass/OCNLI",
        "local": "./data/CLUE/OCNLI/dev.json",
    },
    # ChemBench
    "opencompass/ChemBench": {
        "ms_id": "opencompass/ChemBench",
        "hf_id": "opencompass/ChemBench",
        "local": "./data/ChemBench/",
    },
    # CMMLU
    "opencompass/cmmlu": {
        "ms_id": "opencompass/cmmlu",
        "hf_id": "opencompass/cmmlu",
        "local": "./data/cmmlu/",
    },
    # CommonsenseQA
    "opencompass/commonsense_qa": {
        "ms_id": "opencompass/commonsense_qa",
        "hf_id": "opencompass/commonsense_qa",
        "local": "./data/commonsenseqa",
    },
    # CMRC
    "opencompass/cmrc_dev": {
        "ms_id": "opencompass/cmrc_dev",
        "hf_id": "opencompass/cmrc_dev",
        "local": "./data/CLUE/CMRC/dev.json",
    },
    # DRCD_dev
    "opencompass/drcd_dev": {
        "ms_id": "opencompass/drcd_dev",
        "hf_id": "opencompass/drcd_dev",
        "local": "./data/CLUE/DRCD/dev.json",
    },
    # clozeTest_maxmin
    "opencompass/clozeTest_maxmin": {
        "ms_id": None,
        "hf_id": None,
        "local": "./data/clozeTest-maxmin/python/clozeTest.json",
    },
    # clozeTest_maxmin
    "opencompass/clozeTest_maxmin_answers": {
        "ms_id": None,
        "hf_id": None,
        "local": "./data/clozeTest-maxmin/python/answers.txt",
    },
    # Flores
    "opencompass/flores": {
        "ms_id": "opencompass/flores",
        "hf_id": "opencompass/flores",
        "local": "./data/flores_first100",
    },
    # MBPP
    "opencompass/mbpp": {
        "ms_id": "opencompass/mbpp",
        "hf_id": "opencompass/mbpp",
        "local": "./data/mbpp/mbpp.jsonl",
    },
    # 'opencompass/mbpp': {
    #     'ms_id': 'opencompass/mbpp',
    #     'hf_id': 'opencompass/mbpp',
    #     'local': './data/mbpp/mbpp.jsonl',
    # },
    "opencompass/sanitized_mbpp": {
        "ms_id": "opencompass/mbpp",
        "hf_id": "opencompass/mbpp",
        "local": "./data/mbpp/sanitized-mbpp.jsonl",
    },
    # GSM
    "opencompass/gsm8k": {
        "ms_id": "opencompass/gsm8k",
        "hf_id": "opencompass/gsm8k",
        "local": "./data/gsm8k/",
    },
    # HellaSwag
    "opencompass/hellaswag": {
        "ms_id": "opencompass/hellaswag",
        "hf_id": "opencompass/hellaswag",
        "local": "./data/hellaswag/hellaswag.jsonl",
    },
    # HellaSwagICE
    "opencompass/hellaswag_ice": {
        "ms_id": "opencompass/hellaswag",
        "hf_id": "opencompass/hellaswag",
        "local": "./data/hellaswag/",
    },
    # HumanEval
    "opencompass/humaneval": {
        "ms_id": "opencompass/humaneval",
        "hf_id": "opencompass/humaneval",
        "local": "./data/humaneval/human-eval-v2-20210705.jsonl",
    },
    # HumanEvalCN
    "opencompass/humaneval_cn": {
        "ms_id": "opencompass/humaneval",
        "hf_id": "opencompass/humaneval",
        "local": "./data/humaneval_cn/human-eval-cn-v2-20210705.jsonl",
    },
    #KORBENCH
    "opencompass/korbench": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/korbench",
    },
    # Lambada
    "opencompass/lambada": {
        "ms_id": "opencompass/lambada",
        "hf_id": "opencompass/lambada",
        "local": "./data/lambada/test.jsonl",
    },
    # LCSTS
    "opencompass/LCSTS": {
        "ms_id": "opencompass/LCSTS",
        "hf_id": "opencompass/LCSTS",
        "local": "./data/LCSTS",
    },
    # MATH
    "opencompass/math": {
        "ms_id": "opencompass/math",
        "hf_id": "opencompass/math",
        "local": "./data/math/",
    },
    # MMLU
    "opencompass/mmlu": {
        "ms_id": "opencompass/mmlu",
        "hf_id": "opencompass/mmlu",
        "local": "./data/mmlu/",
    },
    # MMLU_PRO
    "opencompass/mmlu_pro": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/mmlu_pro",
    },
    # MultiPL-E
    "opencompass/multipl_e": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/multipl_e",
    },
    # NQ
    "opencompass/natural_question": {
        "ms_id": "opencompass/natural_question",
        "hf_id": "opencompass/natural_question",
        "local": "./data/nq/",
    },
    # OpenBook QA-test
    "opencompass/openbookqa_test": {
        "ms_id": "opencompass/openbookqa",
        "hf_id": "opencompass/openbookqa",
        "local": "./data/openbookqa/Main/test.jsonl",
    },
    # OpenBook QA-fact
    "opencompass/openbookqa_fact": {
        "ms_id": "opencompass/openbookqa",
        "hf_id": "opencompass/openbookqa",
        "local": "./data/openbookqa/Additional/test_complete.jsonl",
    },
    # PIQA
    "opencompass/piqa": {
        "ms_id": "opencompass/piqa",
        "hf_id": "opencompass/piqa",
        "local": "./data/piqa",
    },
    # RACE
    "opencompass/race": {
        "ms_id": "opencompass/race",
        "hf_id": "opencompass/race",
        "local": "./data/race/",
    },
    # SIQA
    "opencompass/siqa": {
        "ms_id": "opencompass/siqa",
        "hf_id": "opencompass/siqa",
        "local": "./data/siqa",
    },
    # XStoryCloze
    "opencompass/xstory_cloze": {
        "ms_id": "opencompass/xstory_cloze",
        "hf_id": "opencompass/xstory_cloze",
        "local": "./data/xstory_cloze",
    },
    # StrategyQA
    "opencompass/strategy_qa": {
        "ms_id": "opencompass/strategy_qa",
        "hf_id": "opencompass/strategy_qa",
        "local": "./data/strategyqa/strategyQA_train.json",
    },
    # SummEdits
    "opencompass/summedits": {
        "ms_id": "opencompass/summedits",
        "hf_id": "opencompass/summedits",
        "local": "./data/summedits/summedits.jsonl",
    },
    # SuperGLUE
    "opencompass/boolq": {
        "ms_id": "opencompass/boolq",
        "hf_id": "opencompass/boolq",
        "local": "./data/SuperGLUE/BoolQ/val.jsonl",
    },
    # TriviaQA
    "opencompass/trivia_qa": {
        "ms_id": "opencompass/trivia_qa",
        "hf_id": "opencompass/trivia_qa",
        "local": "./data/triviaqa/",
    },
    # TydiQA
    "opencompass/tydiqa": {
        "ms_id": "opencompass/tydiqa",
        "hf_id": "opencompass/tydiqa",
        "local": "./data/tydiqa/",
    },
    # Winogrande
    "opencompass/winogrande": {
        "ms_id": "opencompass/winogrande",
        "hf_id": "opencompass/winogrande",
        "local": "./data/winogrande/",
    },
    # XSum
    "opencompass/xsum": {
        "ms_id": "opencompass/xsum",
        "hf_id": "opencompass/xsum",
        "local": "./data/Xsum/dev.jsonl",
    },
    # Longbench
    "opencompass/Longbench": {
        "ms_id": "",
        "hf_id": "THUDM/LongBench",
        "local": "./data/Longbench",
    },
    # Needlebench
    "opencompass/needlebench": {
        "ms_id": "",
        "hf_id": "opencompass/needlebench",
        "local": "./data/needlebench",
    },
    "opencompass/code_generation_lite": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/code_generation_lite",
    },
    "opencompass/execution-v2": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/execution-v2",
    },
    "opencompass/test_generation": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/test_generation",
    },
    "opencompass/aime2024": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/aime.jsonl",
    },
    "opencompass/aime2025": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/aime2025/aime2025.jsonl",
    },
    "opencompass/cmo_fib": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/cmo.jsonl",
    },
    "opencompass/nq_open": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/nq-open/",
    },
    "opencompass/GAOKAO-BENCH": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/GAOKAO-BENCH/data",
    },
    "opencompass/WikiBench": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/WikiBench/",
    },
    "opencompass/mmmlu_lite": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/mmmlu_lite",
    },
    "opencompass/mmmlu_lite": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/mmmlu_lite",
    },
    "opencompass/musr": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/musr",
    },
    "opencompass/babilong": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/babilong/data/",
    },
    "P-MMEval": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/P-MMEval/",
    },
    "opencompass/arc_prize_public_evaluation": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/arc_prize_public_evaluation",
    },
    "opencompass/simpleqa": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/simpleqa/simple_qa_test_set.csv",
    },
    "opencompass/chinese_simpleqa": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/chinese_simpleqa",
    },
    "opencompass/LiveMathBench202412": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/LiveMathBench/",
    },
    "opencompass/LiveMathBench": {
        "ms_id": "",
        "hf_id": "opencompass/LiveMathBench",
        "local": "./data/LiveMathBench/",
    },
    "opencompass/LiveReasonBench": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/LiveReasonBench/",
    },
    "opencompass/bigcodebench": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/bigcodebench/",
    },
    "opencompass/qabench": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/qabench",
    },
    "opencompass/livestembench": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/livestembench/",
    },
    "opencompass/longbenchv2": {
        "ms_id": "",
        "hf_id": "THUDM/LongBench-v2",
        "local": "./data/longbenchv2/data.json",
    },
    "opencompass/OlympiadBench": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/OlympiadBench",
    },
    "opencompass/ClimaQA-Gold": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/climaqa_gold",
    },
    "opencompass/ClimaQA-Silver": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/climaqa_silver",
    },
    "opencompass/PHYSICS-textonly": {
        "ms_id": "",
        "hf_id": "",
        "local": "./data/PHYSICS-textonly",
    },

}

DATASETS_URL = {
    "/climaqa_gold": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/climaqa_gold.zip",
        "md5": "310cd0dc96db2bbbce798c40e2163ac2",
    },
    "/climaqa_silver": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/climaqa_silver.zip",
        "md5": "acdd955f1c170539c5233c12f7227c58",
    },
    "/PHYSICS-textonly": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/PHYSICS-textonly.zip",
        "md5": "92be6846a22dd4da942ca43f0638c709",
    },
    "/OlympiadBench": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/OlympiadBench.zip",
        "md5": "97e8b1ae7f6170d94817288a8930ef00",
    },
    "/longbenchv2": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/longbenchv2.zip",
        "md5": "09b7e06e6f98c5cca8ad597b3d7b42f0",
    },
    "/livestembench": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/livestembench.zip",
        "md5": "0ff59d031c3dcff56a2e00e8c1489f5d",
    },
    "/musr": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/musr.zip",
        "md5": "7447d2a5bec4586035196102135e2af9",
    },
    "/mmlu/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu.zip",
        "md5": "761310671509a239e41c4b717f7fab9c",
    },
    "/mmmlu_lite": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmmlu_lite.zip",
        "md5": "a776af1220e1826fd0608eda1bc4425e",
    },
    "/simpleqa": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/simpleqa.zip",
        "md5": "1d83fc2e15798d39cb265c9a3cb5195a",
    },
    "/chinese_simpleqa": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/chinese_simpleqa.zip",
        "md5": "4bdf854b291fc0ee29da57dc47ac47b5",
    },
    "/gpqa/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip",
        "md5": "2e9657959030a765916f1f2aca29140d",
    },
    "/CHARM/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/CHARM.zip",
        "md5": "fdf51e955d1b8e0bb35bc1997eaf37cb",
    },
    "/ifeval/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ifeval.zip",
        "md5": "64d98b6f36b42e7390c9cef76cace75f",
    },
    "/mbpp/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip",
        "md5": "777739c90f04bce44096a5bc96c8f9e5",
    },
    "/cmmlu/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/cmmlu.zip",
        "md5": "a59f4003d6918509a719ce3bc2a5d5bc",
    },
    "/math/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip",
        "md5": "cb5b4c8378085929e20345174e731fdf",
    },
    "/hellaswag/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/hellaswag.zip",
        "md5": "2b700a02ffb58571c7df8d8d0619256f",
    },
    "/BBH/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/BBH.zip",
        "md5": "60c49f9bef5148aa7e1941328e96a554",
    },
    "/compass_arena/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/compass_arena.zip",
        "md5": "cd59b54a179d16f2a858b359b60588f6",
    },
    "/TheoremQA/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/TheoremQA.zip",
        "md5": "f2793b07bc26510d507aa710d9bd8622",
    },
    "/mathbench_v1/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mathbench_v1.zip",
        "md5": "50257a910ca43d1f61a610a79fdb16b5",
    },
    "/gsm8k/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip",
        "md5": "901e5dc93a2889789a469da9850cdca8",
    },
    "/LCBench2023/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/LCBench2023.zip",
        "md5": "e1a38c94a42ad1809e9e0650476a9306",
    },
    "/humaneval/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humaneval.zip",
        "md5": "88b1b89dc47b7121c81da6bcd85a69c3",
    },
    "/humanevalx": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humanevalx.zip",
        "md5": "22930355c03fb73fb5bae14b50f1deb9",
    },
    "/ds1000_data": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ds1000_data.zip",
        "md5": "1a4990aec04a2fd73ccfad12e2d43b43",
    },
    "/drop_simple_eval/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/drop_simple_eval.zip",
        "md5": "c912afe5b4a63509851cf16e6b91830e",
    },
    "subjective/alignment_bench/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/alignment_bench.zip",
        "md5": "d8ae9a0398526479dbbcdb80fafabceb",
    },
    "subjective/alpaca_eval": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/alpaca_eval.zip",
        "md5": "d7399d63cb46c82f089447160ef49b6a",
    },
    "subjective/arena_hard": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/arena_hard.zip",
        "md5": "02cd09a482cb0f0cd9d2c2afe7a1697f",
    },
    "subjective/mtbench": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mtbench.zip",
        "md5": "d1afc0787aeac7f1f24872742e161069",
    },
    "subjective/fofo": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/fofo.zip",
        "md5": "8a302712e425e27e4292a9369df5b9d3",
    },
    "subjective/followbench": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/followbench.zip",
        "md5": "da7a831817c969da15d1e78d4a245d8a",
    },
    "subjective/mtbench101": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mtbench101.zip",
        "md5": "5d80257bc9929ebe5cfbf6d11184b04c",
    },
    "subjective/WildBench": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/wildbench.zip",
        "md5": "b06252857f1f8f44a17b1bfca4888ff4",
    },
    "/ruler/": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ruler.zip",
        "md5": "c60bdfff3d02358067104cc1dea7c0f7",
    },
    "/scicode": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/scicode.zip",
        "md5": "9c6c64b8c70edc418f713419ea39989c",
    },
    "/commonsenseqa": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/commonsenseqa.zip",
        "md5": "c4a82fc07c81ae1462605f5d7fd2bb2e",
    },
    "FewCLUE": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/FewCLUE.zip",
        "md5": "7976e2bb0e9d885ffd3c55f7c5d4021e",
    },
    "/race": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/race.zip",
        "md5": "b758251764a264746cf45749c02363f9",
    },
    "/ARC": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ARC.zip",
        "md5": "d720629b69f1a51cfe78bf65b00b44f6",
    },
    "/SuperGLUE": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/SuperGLUE.zip",
        "md5": "b60904915b0b61d1a04ea52280169936",
    },
    "SQuAD2.0": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/SQuAD2.0.zip",
        "md5": "1321cbf9349e1102a57d31d1b2bfdd7e",
    },
    "mmlu_pro": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu_pro.zip",
        "md5": "e3200c7380f4cea5f13c768f2815fabb",
    },
    "multipl_e": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/multipl_e.zip",
        "md5": "24462aac7a38a4a62f5c5e89eb614e20",
    },
    "/Longbench": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/Longbench.zip",
        "md5": "ab0cb9e520ae5cfb899bf38b564249bb",
    },
    "/needlebench": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/needlebench.zip",
        "md5": "dad5c903ebfea16eaf186b8997aeedad",
    },
    "/teval": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/teval.zip",
        "md5": "7628ab5891a26bf96ca17becfd044867",
    },
    "/code_generation_lite": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/code_generation_lite.zip",
        "md5": "ebcf8db56f5c817ca8202a542be30cb4",
    },
    "/execution-v2": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/execution-v2.zip",
        "md5": "019ef1a0686ee6ca34f51c8af104fcd9",
    },
    "/test_generation": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/test_generation.zip",
        "md5": "918a6ea2b1eee6f2b1314db3c21cb4c7",
    },
    "/aime2024": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip",
        "md5": "fbe2d0577fc210962a549f8cea1a00c8",
    },
    "/aime2025": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip",
        "md5": "aa18cd5d2e2de246c5397f5eb1e61004",
    },
    "/cmo": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/cmo.zip",
        "md5": "fad52c81290506a8ca74f46b5400d8fc",
    },
    "/nq-open": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/nq-open.zip",
        "md5": "a340521e5c9ec591227dcb367f718b25",
    },
    "/winogrande": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/winogrande.zip",
        "md5": "9e949a75eacc26ed4fd2b9aa870b495b",
    },
    "/triviaqa": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/triviaqa.zip",
        "md5": "e6a118d744236814926b2ec7ec66c034",
    },
    "/GAOKAO-BENCH": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/GAOKAO-BENCH.zip",
        "md5": "ba3c71b8b9db96d2a0664b977c4f9784",
    },
    "/WikiBench": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/WikiBench.zip",
        "md5": "6dac1d1a3133fe1effff185cbf71d928",
    },
    "/babilong": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/babilong.zip",
        "md5": "e400864c31bc58d29eaa3e199751f99b",
    },
    "/korbench": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/korbench.zip",
        "md5": "9107597d137e7362eaf7d218ddef7a6d",
    },
    "/bbeh": {
        "url": "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/bbeh.zip",
        "md5": "43a3c2d73aee731ac68ac790bc9a358e",
    },
    "subjective/judgerbench": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/judgerbench.zip",
        "md5": "60d605883aa8cac9755819140ab42c6b"
    },
    "/arc_prize_public_evaluation": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/arc_prize_public_evaluation.zip",
        "md5": "367a33977651496efddba7670009807e"
    },
    "P-MMEval": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/pmmeval.zip",
        "md5": "09e401e6229a50647b9e13c429e634d1",
    },
    "LiveMathBench": {
        'url':
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/LiveMathBench.zip",
        "md5": "d0781f9185c9bb50e81e6e3ca8c59013",
    },
    "bigcodebench": {
        "url":
        "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/bigcodebench.zip",
        "md5": "270f399f4142b74f47ecff116cc3b21d"
    }
}
