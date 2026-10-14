import copy

from opencompass.summarizers.default import DefaultSummarizer


OQMD_SUBSETS = [
    'SciReasoner1_5-OQMD-bandgap',
    'SciReasoner1_5-OQMD-e_form',
]

JARVISDFT_SUBSETS = [
    'SciReasoner1_5-JARVISDFT-formation_energy_peratom',
    'SciReasoner1_5-JARVISDFT-optb88vdw_bandgap',
    'SciReasoner1_5-JARVISDFT-optb88vdw_total_energy',
    'SciReasoner1_5-JARVISDFT-ehull',
    'SciReasoner1_5-JARVISDFT-n-Seebeck',
    'SciReasoner1_5-JARVISDFT-n-powerfact',
    'SciReasoner1_5-JARVISDFT-p-Seebeck',
    'SciReasoner1_5-JARVISDFT-p-powerfact',
    'SciReasoner1_5-JARVISDFT-bulk_modulus_kv',
    'SciReasoner1_5-JARVISDFT-shear_modulus_gv',
    'SciReasoner1_5-JARVISDFT-mbj_bandgap',
    'SciReasoner1_5-JARVISDFT-mepsx',
    'SciReasoner1_5-JARVISDFT-avg_elec_mass',
    'SciReasoner1_5-JARVISDFT-max_efg',
    'SciReasoner1_5-JARVISDFT-spillage',
    'SciReasoner1_5-JARVISDFT-slme',
    'SciReasoner1_5-JARVISDFT-dfpt_piezo_max_eij',
    'SciReasoner1_5-JARVISDFT-dfpt_piezo_max_dielectric',
    'SciReasoner1_5-JARVISDFT-dfpt_piezo_max_dij',
    'SciReasoner1_5-JARVISDFT-exfoliation_energy',
]

MATERIAL_SUBSETS = OQMD_SUBSETS + JARVISDFT_SUBSETS

GO_BP_SUBSET = 'SciReasoner1_5-GO-BP'
TMSCORE_SUBSET = 'SciReasoner1_5-TMScore'
DUDE_SUBSET = 'SciReasoner1_5-DUDE-count'


def _metric_rows(subsets, metric):
    return [[subset, metric] for subset in subsets]


def _clip_score(value):
    return min(100.0, max(0.0, float(value)))


def _normalize_score(value, normalizer):
    value = float(value)
    if normalizer == 'percent':
        return _clip_score(value)
    if normalizer == 'unit_interval':
        return _clip_score(value * 100.0)
    if normalizer == 'spearman':
        return _clip_score((value + 1.0) * 50.0)
    if normalizer == 'mae_0_1':
        return _clip_score((1.0 - value) * 100.0)
    if normalizer == 'mad_mae_ratio':
        # MAD/MAE is an unbounded benefit-over-mean-baseline ratio. This maps
        # baseline parity to 50 and better models asymptotically toward 100.
        return _clip_score(100.0 * max(value, 0.0) / (1.0 + max(value, 0.0)))
    raise ValueError(f'Unsupported SciReasoner1_5 normalizer: {normalizer}')


scireasoner1_5_summary_groups = [
    {
        'name': 'SciReasoner1_5-OQMD',
        'subsets': _metric_rows(OQMD_SUBSETS, 'MAD/MAE'),
        'normalizer': 'mad_mae_ratio',
    },
    {
        'name': 'SciReasoner1_5-JARVISDFT',
        'subsets': _metric_rows(JARVISDFT_SUBSETS, 'MAD/MAE'),
        'normalizer': 'mad_mae_ratio',
    },
    {
        'name': 'SciReasoner1_5-GO-BP',
        'subsets': [[GO_BP_SUBSET, 'score']],
        'normalizer': 'percent',
    },
    {
        'name': 'SciReasoner1_5-TMScore-MAE',
        'subsets': [[TMSCORE_SUBSET, 'MAE']],
        'normalizer': 'mae_0_1',
    },
    {
        'name': 'SciReasoner1_5-DUDE-AUC',
        'subsets': [[DUDE_SUBSET, 'AUC']],
        'normalizer': 'unit_interval',
    },
]

scireasoner1_5_detail_rows = (
    _metric_rows(MATERIAL_SUBSETS, 'MAD/MAE') + [
        [GO_BP_SUBSET, 'score'],
        [TMSCORE_SUBSET, 'MAE'],
        [DUDE_SUBSET, 'AUC'],
    ])


def _with_mini_suffix(summary_groups, detail_rows):
    summary_groups = copy.deepcopy(summary_groups)
    detail_rows = copy.deepcopy(detail_rows)
    for group in summary_groups:
        group['name'] = f"{group['name']}-mini"
        for subset in group['subsets']:
            subset[0] = f'{subset[0]}-mini'
    for row in detail_rows:
        row[0] = f'{row[0]}-mini'
    return summary_groups, detail_rows


class SciReasoner15Summarizer(DefaultSummarizer):

    def __init__(self, mini_set=False, show_details=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        summary_groups = copy.deepcopy(scireasoner1_5_summary_groups)
        detail_rows = copy.deepcopy(scireasoner1_5_detail_rows)
        if mini_set:
            summary_groups, detail_rows = _with_mini_suffix(
                summary_groups, detail_rows)

        self.summary_groups = summary_groups
        self.dataset_abbrs = [
            [group['name'], 'score'] for group in summary_groups
        ]
        if show_details:
            self.dataset_abbrs += detail_rows

    def _calculate_group_metrics(self, raw_results, parsed_results,
                                 dataset_metrics, dataset_eval_mode):
        for group in self.summary_groups:
            metric_name = 'score'
            normalizer = group['normalizer']
            for model_abbr in self.model_abbrs:
                values = {}
                raw_values = {}
                missing = []
                eval_modes = []
                for dataset_abbr, source_metric in group['subsets']:
                    value = parsed_results[model_abbr].get(
                        dataset_abbr, {}).get(source_metric)
                    if isinstance(value, (int, float)) and not isinstance(
                            value, bool):
                        raw_value = float(value)
                        key = f'{dataset_abbr}@{source_metric}'
                        raw_values[key] = raw_value
                        values[key] = _normalize_score(raw_value, normalizer)
                        eval_modes.append(
                            dataset_eval_mode.get(dataset_abbr, 'unknown'))
                    else:
                        missing.append(f'{dataset_abbr}@{source_metric}')

                if not values:
                    raw_results[model_abbr][group['name']] = {
                        'error': f'missing metrics: {missing}',
                    }
                    continue

                if missing:
                    self.logger.warning(
                        f'Missing metrics for {group["name"]}: {missing}; '
                        'averaging available metrics only.')

                score = sum(values.values()) / len(values)
                raw_results[model_abbr].setdefault(group['name'], {}).update({
                    metric_name: values,
                    'raw_values': raw_values,
                    'normalizer': normalizer,
                    'available_count': len(values),
                    'missing_count': len(missing),
                })
                parsed_results[model_abbr].setdefault(group['name'],
                                                      {})[metric_name] = score
                dataset_metrics[group['name']] = [metric_name]
                eval_modes = list(set(eval_modes))
                dataset_eval_mode[group['name']] = (
                    eval_modes[0] if len(eval_modes) == 1 else 'mixed')

        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode
