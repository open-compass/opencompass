from mmengine.config import read_base

with read_base():
    # scireasoner
    from opencompass.configs.datasets.SciReasoner.scireasoner_gen import scireasoner_full_datasets, scireasoner_mini_datasets
from opencompass.configs.summarizers.scireasoner import SciReasonerSummarizer

summarizer = dict(
    type=SciReasonerSummarizer,
    mini_set=False,  # 如果测的是mini版本需要开True，默认False
    show_details=False  # 是否需要展示最底层的分数，默认不展示
)