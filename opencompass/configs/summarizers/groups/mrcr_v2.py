needles = ['2needle', '4needle', '8needle']
ranges = [
    'in_4096_8192',
    'in_8192_16384',
    'in_16384_32768',
    'in_32768_65536',
    'in_65536_131072',
    'in_131072_262144',
    'in_262144_524288',
    'in_524288_1048576',
]

# Full: all 24 fixed-range subsets
mrcr_v2_summary_groups = [
    {
        'name': 'MRCR_v2_full',
        'subsets': [f'mrcr_v2_{n}_{r}' for n in needles for r in ranges],
    },
    {
        'name': 'MRCR_v2_upto128k',
        'subsets': [f'mrcr_v2_{n}_upto_128K' for n in needles],
    },
]
