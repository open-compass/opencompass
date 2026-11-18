# HardMTBench reports every metric per direction (paper Table 3); the
# per-domain table averages zh2en and en2zh, which is what these groups do.
# Both directions have the same number of items, so the plain average equals
# the item-level average. Groups sharing a name accumulate metrics; the first
# one (gemba_da) is what the group shows when no metric is given.
_hardmtbench_metrics = ['gemba_da', 'xcomet_xxl', 'term_acc']

hardmtbench_summary_groups = []
for _prefix in ['hardmtbench', 'hardmtbench_technology']:
    hardmtbench_summary_groups += [
        dict(name=_prefix,
             subsets=[[f'{_prefix}_{direction}', metric]
                      for direction in ['zh2en', 'en2zh']],
             metric=metric) for metric in _hardmtbench_metrics
    ]
