OQMD_SUBSETS = [
    ['SciReasoner1_5-OQMD-bandgap', 'MAD/MAE'],
    ['SciReasoner1_5-OQMD-e_form', 'MAD/MAE'],
]

JARVISDFT_SUBSETS = [
    ['SciReasoner1_5-JARVISDFT-formation_energy_peratom', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-optb88vdw_bandgap', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-optb88vdw_total_energy', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-ehull', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-n-Seebeck', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-n-powerfact', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-p-Seebeck', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-p-powerfact', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-bulk_modulus_kv', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-shear_modulus_gv', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-mbj_bandgap', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-mepsx', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-avg_elec_mass', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-max_efg', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-spillage', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-slme', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-dfpt_piezo_max_eij', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-dfpt_piezo_max_dielectric', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-dfpt_piezo_max_dij', 'MAD/MAE'],
    ['SciReasoner1_5-JARVISDFT-exfoliation_energy', 'MAD/MAE'],
]

def _make_mini(subsets):
    return [[s[0] + '-mini', s[1]] for s in subsets]


def _make_mini_transforms(transforms):
    return {k + '-mini': v for k, v in transforms.items()}


def _build_groups(suffix=''):
    s = suffix
    return [
        {
            'name': f'SciReasoner1_5-OQMD{s}',
            'subsets': _make_mini(OQMD_SUBSETS) if s else OQMD_SUBSETS,
            'transforms': _make_mini_transforms(
                {sub[0]: 'min(100, max(0, 100*x/(1+x)))' for sub in OQMD_SUBSETS}
            ) if s else {sub[0]: 'min(100, max(0, 100*x/(1+x)))' for sub in OQMD_SUBSETS},
        },
        {
            'name': f'SciReasoner1_5-JARVISDFT{s}',
            'subsets': _make_mini(JARVISDFT_SUBSETS) if s else JARVISDFT_SUBSETS,
            'transforms': _make_mini_transforms(
                {sub[0]: 'min(100, max(0, 100*x/(1+x)))' for sub in JARVISDFT_SUBSETS}
            ) if s else {sub[0]: 'min(100, max(0, 100*x/(1+x)))' for sub in JARVISDFT_SUBSETS},
        },
        {
            'name': f'SciReasoner1_5-GO-BP{s}',
            'subsets': [[f'SciReasoner1_5-GO-BP{s}', 'score']],
            'transforms': {f'SciReasoner1_5-GO-BP{s}': 'min(100, max(0, x))'},
        },
        {
            'name': f'SciReasoner1_5-TMScore-MAE{s}',
            'subsets': [[f'SciReasoner1_5-TMScore{s}', 'MAE']],
            'transforms': {f'SciReasoner1_5-TMScore{s}': 'min(100, max(0, (1-x)*100))'},
        },
        {
            'name': f'SciReasoner1_5-DUDE-AUC{s}',
            'subsets': [[f'SciReasoner1_5-DUDE-count{s}', 'AUC']],
            'transforms': {f'SciReasoner1_5-DUDE-count{s}': 'min(100, max(0, x*100))'},
        },
    ]


def _build_overall(suffix=''):
    s = suffix
    return {
        'name': f'SciReasoner1_5{s}',
        'subsets': [
            [f'SciReasoner1_5-OQMD{s}', 'naive_average'],
            [f'SciReasoner1_5-JARVISDFT{s}', 'naive_average'],
            [f'SciReasoner1_5-GO-BP{s}', 'naive_average'],
            [f'SciReasoner1_5-TMScore-MAE{s}', 'naive_average'],
            [f'SciReasoner1_5-DUDE-AUC{s}', 'naive_average'],
        ],
    }


scireasoner1_5_summary_groups = _build_groups('') + [_build_overall('')]
scireasoner1_5_mini_summary_groups = _build_groups('-mini') + [_build_overall('-mini')]
