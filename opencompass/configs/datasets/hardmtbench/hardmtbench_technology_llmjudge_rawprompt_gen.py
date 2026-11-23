from copy import deepcopy

from mmengine.config import read_base

with read_base():
    from .hardmtbench_llmjudge_rawprompt_gen import \
        hardmtbench_datasets as _hardmtbench_all  # noqa: F401, F403

hardmtbench_technology_datasets = []
for _cfg in _hardmtbench_all:
    _cfg = deepcopy(_cfg)
    _cfg['abbr'] = _cfg['abbr'].replace('hardmtbench_',
                                        'hardmtbench_technology_')
    _cfg['domain'] = 'Technology'
    hardmtbench_technology_datasets.append(_cfg)
