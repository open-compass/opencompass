from mmengine.config import read_base

with read_base():
    from .infinitebenchcodedebug.infinitebench_codedebug_gen import InfiniteBench_codedebug_datasets
    from .infinitebenchcoderun.infinitebench_coderun_gen import InfiniteBench_coderun_datasets
    from .infinitebenchendia.infinitebench_endia_gen import InfiniteBench_endia_datasets
    from .infinitebenchenmc.infinitebench_enmc_gen import InfiniteBench_enmc_datasets
    from .infinitebenchenqa.infinitebench_enqa_gen import InfiniteBench_enqa_datasets
    from .infinitebenchensum.infinitebench_ensum_gen import InfiniteBench_ensum_datasets
    from .infinitebenchmathcalc.infinitebench_mathcalc_gen import InfiniteBench_mathcalc_datasets
    from .infinitebenchmathfind.infinitebench_mathfind_gen import InfiniteBench_mathfind_datasets
    from .infinitebenchretrievekv.infinitebench_retrievekv_gen import InfiniteBench_retrievekv_datasets
    from .infinitebenchretrievenumber.infinitebench_retrievenumber_gen import InfiniteBench_retrievenumber_datasets
    from .infinitebenchretrievepasskey.infinitebench_retrievepasskey_gen import InfiniteBench_retrievepasskey_datasets
    from .infinitebenchzhqa.infinitebench_zhqa_gen import InfiniteBench_zhqa_datasets

infinitebench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
