
_cibench = ['Pandas', 'Matplotlib', 'Opencv', 'SciPy', 'Seaborn', 'PyTorch']
_cibench = ['cibench_' + i for i in _cibench]
cibench_summary_groups = [{'name': 'cibench', 'subsets': _cibench}]

_cibench_template = ['lightgbm', 'matplotlib', 'nltk', 'opencv', 'pandas', 'pytorch',
    'scipy', 'seaborn', 'sklearn', 'tensorflow']
_cibench_template = ['cibench_template/' + i for i in _cibench_template]
# number of total exec questions in this module
_cibench_template_weight = {
    'lightgbm': [30, 15, 0, 0],
    'matplotlib': [42, 0, 0, 36],
    'nltk': [70, 30, 20, 10],
    'opencv': [60, 10, 0, 40],
    'pandas': [60, 40, 0, 10],
    'pytorch': [28, 0, 0, 0],
    'scipy': [60, 40, 0, 0],
    'seaborn': [42, 0, 0, 35],
    'sklearn': [42, 6, 0, 18],
    'tensorflow': [36, 6, 0, 12],
}
cibench_summary_groups.extend([
    {
        'name': 'cibench_template:executable',
        'subsets': [[i, 'executable'] for i in _cibench_template],
        'weights': {'cibench_template/' + k : v[0] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template:numeric_correct',
        'subsets': [[i, 'numeric_correct'] for i in _cibench_template],
        'weights': {'cibench_template/' + k : v[1] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template:text_score',
        'subsets': [[i, 'text_score'] for i in _cibench_template],
        'weights': {'cibench_template/' + k : v[2] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template:vis_sim',
        'subsets': [[i, 'vis_sim'] for i in _cibench_template],
        'weights': {'cibench_template/' + k : v[3] for k,v in _cibench_template_weight.items()},
    },
])


## chinese
_cibench_template_cn = ['lightgbm', 'matplotlib', 'nltk', 'opencv', 'pandas', 'pytorch',
    'scipy', 'seaborn', 'sklearn', 'tensorflow']
_cibench_template_cn = ['cibench_template_chinese/' + i for i in _cibench_template_cn]
cibench_summary_groups.extend([
    {
        'name': 'cibench_template_cn:executable',
        'subsets': [[i, 'executable'] for i in _cibench_template_cn],
        'weights': {'cibench_template_chinese/' + k : v[0] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template_cn:numeric_correct',
        'subsets': [[i, 'numeric_correct'] for i in _cibench_template_cn],
        'weights': {'cibench_template_chinese/' + k : v[1] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template_cn:text_score',
        'subsets': [[i, 'text_score'] for i in _cibench_template_cn],
        'weights': {'cibench_template_chinese/' + k : v[2] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template_cn:vis_sim',
        'subsets': [[i, 'vis_sim'] for i in _cibench_template_cn],
        'weights': {'cibench_template_chinese/' + k : v[3] for k,v in _cibench_template_weight.items()},
    },
])


## add more without nltk
cibench_summary_groups.extend([
    {
        'name': 'cibench_template_wo_nltk:executable',
        'subsets': [[i, 'executable'] for i in _cibench_template if 'nltk' not in i],
        'weights': {'cibench_template/' + k : v[0] for k,v in _cibench_template_weight.items() if 'nltk' not in k},
    },
    {
        'name': 'cibench_template_wo_nltk:numeric_correct',
        'subsets': [[i, 'numeric_correct'] for i in _cibench_template if 'nltk' not in i],
        'weights': {'cibench_template/' + k : v[1] for k,v in _cibench_template_weight.items() if 'nltk' not in k},
    },
    {
        'name': 'cibench_template_wo_nltk:vis_sim',
        'subsets': [[i, 'vis_sim'] for i in _cibench_template if 'nltk' not in i],
        'weights': {'cibench_template/' + k : v[3] for k,v in _cibench_template_weight.items() if 'nltk' not in k},
    },
])

cibench_summary_groups.extend([
    {
        'name': 'cibench_template_cn_wo_nltk:executable',
        'subsets': [[i, 'executable'] for i in _cibench_template_cn if 'nltk' not in i],
        'weights': {'cibench_template_chinese/' + k : v[0] for k,v in _cibench_template_weight.items() if 'nltk' not in k},
    },
    {
        'name': 'cibench_template_cn_wo_nltk:numeric_correct',
        'subsets': [[i, 'numeric_correct'] for i in _cibench_template_cn if 'nltk' not in i],
        'weights': {'cibench_template_chinese/' + k : v[1] for k,v in _cibench_template_weight.items() if 'nltk' not in k},
    },
    {
        'name': 'cibench_template_cn_wo_nltk:vis_sim',
        'subsets': [[i, 'vis_sim'] for i in _cibench_template_cn if 'nltk' not in i],
        'weights': {'cibench_template_chinese/' + k : v[3] for k,v in _cibench_template_weight.items() if 'nltk' not in k},
    },
])
