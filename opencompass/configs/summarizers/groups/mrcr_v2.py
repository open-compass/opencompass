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
        'subsets': [
            f'mrcr_v2_{n}_{r}_full' for n in needles for r in ranges
        ],
    },
    {
        'name': 'MRCR_v2_upto128k',
        'subsets': [f'mrcr_v2_{n}_upto_128K' for n in needles],
    },
]

# Per context length: matches datasets/mrcr/mrcr_v2_{size}_rawprompt_gen.py
context_window_sizes = ['8k', '16k', '32k', '64k', '128k', '256k', '512k', '1m']
for size, r in zip(context_window_sizes, ranges):
    mrcr_v2_summary_groups.append({
        'name': f'MRCR_v2_{size}',
        'subsets': [f'mrcr_v2_{n}_{r}_{size}' for n in needles],
    })
