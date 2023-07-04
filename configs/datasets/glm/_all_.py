_base_ = [
    'bustm.py',
    'afqmc.py',
    'eprstmt.py',
    'ocnli_fc.py',
    'ocnli.py',
    'cmnli.py',
    'csl.py',
    'chid.py',
    'cluewsc.py',
    'tnews.py',
    'C3.py',
    'CMRC.py',
    'DRCD.py',
    'lcsts.py',
    'piqa.py',
    'commonsenseqa.py',
    'gsm8k.py',
    'flores.py',
    'humaneval.py',
    'mbpp.py',
    'triviaqa.py',
    'nq.py',
    'agieval.py',
    'mmlu.py',
    'ceval.py',
]

datasets = []
for k, v in _base_.items():
    if k.endswith("_datasets"):
        datasets += v
