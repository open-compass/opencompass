
_cibench_generation_modules = ['pandas', 'matplotlib', 'opencv', 'scipy', 'seaborn', 'pytorch']
_cibench_generation = ['cibench_generation/' + i for i in _cibench_generation_modules]
cibench_summary_groups = []
_cibench_generation_weight = {
    'matplotlib': [223, 50, 1, 156],
    'pandas': [200, 45, 45, 38],
    'pytorch': [69, 0, 8, 11],
    'seaborn': [130, 0, 2, 106],
    'opencv': [177, 21, 6, 106],
    'scipy': [161, 94, 14, 49],
}
cibench_summary_groups.extend([
    {
        'name': 'cibench_generation:tool_rate',
        'subsets': [[i, 'tool_rate'] for i in _cibench_generation],
        'weights': {'cibench_generation/' + k : v[0] for k,v in _cibench_generation_weight.items()},
    },
    {
        'name': 'cibench_generation:executable',
        'subsets': [[i, 'executable'] for i in _cibench_generation],
        'weights': {'cibench_generation/' + k : v[0] for k,v in _cibench_generation_weight.items()},
    },
    {
        'name': 'cibench_generation:numeric_correct',
        'subsets': [[i, 'numeric_correct'] for i in _cibench_generation],
        'weights': {'cibench_generation/' + k : v[1] for k,v in _cibench_generation_weight.items()},
    },
    {
        'name': 'cibench_generation:text_score',
        'subsets': [[i, 'text_score'] for i in _cibench_generation],
        'weights': {'cibench_generation/' + k : v[2] for k,v in _cibench_generation_weight.items()},
    },
    {
        'name': 'cibench_generation:vis_sim',
        'subsets': [[i, 'vis_sim'] for i in _cibench_generation],
        'weights': {'cibench_generation/' + k : v[3] for k,v in _cibench_generation_weight.items()},
    },
])

_cibench_generation = ['cibench_generation_oracle/' + i for i in _cibench_generation_modules]
cibench_summary_groups.extend([
    {
        'name': 'cibench_generation_oracle:tool_rate',
        'subsets': [[i, 'tool_rate'] for i in _cibench_generation],
        'weights': {'cibench_generation_oracle/' + k : v[0] for k,v in _cibench_generation_weight.items()},
    },
    {
        'name': 'cibench_generation_oracle:executable',
        'subsets': [[i, 'executable'] for i in _cibench_generation],
        'weights': {'cibench_generation_oracle/' + k : v[0] for k,v in _cibench_generation_weight.items()},
    },
    {
        'name': 'cibench_generation_oracle:numeric_correct',
        'subsets': [[i, 'numeric_correct'] for i in _cibench_generation],
        'weights': {'cibench_generation_oracle/' + k : v[1] for k,v in _cibench_generation_weight.items()},
    },
    {
        'name': 'cibench_generation_oracle:text_score',
        'subsets': [[i, 'text_score'] for i in _cibench_generation],
        'weights': {'cibench_generation_oracle/' + k : v[2] for k,v in _cibench_generation_weight.items()},
    },
    {
        'name': 'cibench_generation_oracle:vis_sim',
        'subsets': [[i, 'vis_sim'] for i in _cibench_generation],
        'weights': {'cibench_generation_oracle/' + k : v[3] for k,v in _cibench_generation_weight.items()},
    },
])

_cibench_template_modules = ['lightgbm', 'matplotlib', 'nltk', 'opencv', 'pandas', 'pytorch',
    'scipy', 'seaborn', 'sklearn', 'tensorflow']
_cibench_template = ['cibench_template/' + i for i in _cibench_template_modules]
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
        'name': 'cibench_template:tool_rate',
        'subsets': [[i, 'tool_rate'] for i in _cibench_template],
        'weights': {'cibench_template/' + k : v[0] for k,v in _cibench_template_weight.items()},
    },
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

_cibench_template_oracle = ['cibench_template_oracle/' + i for i in _cibench_template_modules]
cibench_summary_groups.extend([
    {
        'name': 'cibench_template_oracle:tool_rate',
        'subsets': [[i, 'tool_rate'] for i in _cibench_template_oracle],
        'weights': {'cibench_template_oracle/' + k : v[0] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template_oracle:executable',
        'subsets': [[i, 'executable'] for i in _cibench_template_oracle],
        'weights': {'cibench_template_oracle/' + k : v[0] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template_oracle:numeric_correct',
        'subsets': [[i, 'numeric_correct'] for i in _cibench_template_oracle],
        'weights': {'cibench_template_oracle/' + k : v[1] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template_oracle:text_score',
        'subsets': [[i, 'text_score'] for i in _cibench_template_oracle],
        'weights': {'cibench_template_oracle/' + k : v[2] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template_oracle:vis_sim',
        'subsets': [[i, 'vis_sim'] for i in _cibench_template_oracle],
        'weights': {'cibench_template_oracle/' + k : v[3] for k,v in _cibench_template_weight.items()},
    },
])


## chinese
_cibench_template_cn_modules = ['lightgbm', 'matplotlib', 'nltk', 'opencv', 'pandas', 'pytorch',
    'scipy', 'seaborn', 'sklearn', 'tensorflow']
_cibench_template_cn = ['cibench_template_chinese/' + i for i in _cibench_template_cn_modules]
cibench_summary_groups.extend([
    {
        'name': 'cibench_template_cn:tool_rate',
        'subsets': [[i, 'tool_rate'] for i in _cibench_template_cn],
        'weights': {'cibench_template_chinese/' + k : v[0] for k,v in _cibench_template_weight.items()},
    },
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

_cibench_template_cn_oracle = ['cibench_template_oracle_chinese/' + i for i in _cibench_template_cn_modules]
cibench_summary_groups.extend([
    {
        'name': 'cibench_template_cn_oracle:tool_rate',
        'subsets': [[i, 'tool_rate'] for i in _cibench_template_cn_oracle],
        'weights': {'cibench_template_oracle_chinese/' + k : v[0] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template_cn_oracle:executable',
        'subsets': [[i, 'executable'] for i in _cibench_template_cn_oracle],
        'weights': {'cibench_template_oracle_chinese/' + k : v[0] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template_cn_oracle:numeric_correct',
        'subsets': [[i, 'numeric_correct'] for i in _cibench_template_cn_oracle],
        'weights': {'cibench_template_oracle_chinese/' + k : v[1] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template_cn_oracle:text_score',
        'subsets': [[i, 'text_score'] for i in _cibench_template_cn_oracle],
        'weights': {'cibench_template_oracle_chinese/' + k : v[2] for k,v in _cibench_template_weight.items()},
    },
    {
        'name': 'cibench_template_cn_oracle:vis_sim',
        'subsets': [[i, 'vis_sim'] for i in _cibench_template_cn_oracle],
        'weights': {'cibench_template_oracle_chinese/' + k : v[3] for k,v in _cibench_template_weight.items()},
    },
])


########### New summerizer for Category metric

cibench_data_manipulation = [
    ['cibench_generation/pandas', 'numeric_correct', _cibench_generation_weight['pandas'][1]],
    ['cibench_generation/pandas', 'text_score', _cibench_generation_weight['pandas'][2]],
    ['cibench_generation/pandas', 'vis_sim', _cibench_generation_weight['pandas'][3]],
    ['cibench_template/pandas', 'numeric_correct', _cibench_template_weight['pandas'][1]],
    ['cibench_template/pandas', 'text_score', _cibench_template_weight['pandas'][2]],
    ['cibench_template/pandas', 'vis_sim', _cibench_template_weight['pandas'][3]],
]
cibench_data_visualization = [
    ['cibench_generation/matplotlib', 'numeric_correct', _cibench_generation_weight['matplotlib'][1]],
    ['cibench_generation/matplotlib', 'text_score', _cibench_generation_weight['matplotlib'][2]],
    ['cibench_generation/matplotlib', 'vis_sim', _cibench_generation_weight['matplotlib'][3]],
    ['cibench_generation/seaborn', 'numeric_correct', _cibench_generation_weight['seaborn'][1]],
    ['cibench_generation/seaborn', 'text_score', _cibench_generation_weight['seaborn'][2]],
    ['cibench_generation/seaborn', 'vis_sim', _cibench_generation_weight['seaborn'][3]],
    ['cibench_template/matplotlib', 'numeric_correct', _cibench_template_weight['matplotlib'][1]],
    ['cibench_template/matplotlib', 'text_score', _cibench_template_weight['matplotlib'][2]],
    ['cibench_template/matplotlib', 'vis_sim', _cibench_template_weight['matplotlib'][3]],
    ['cibench_template/seaborn', 'numeric_correct', _cibench_template_weight['seaborn'][1]],
    ['cibench_template/seaborn', 'text_score', _cibench_template_weight['seaborn'][2]],
    ['cibench_template/seaborn', 'vis_sim', _cibench_template_weight['seaborn'][3]],
]
cibench_modeling = [
    ['cibench_generation/pytorch', 'numeric_correct', _cibench_generation_weight['pytorch'][1]],
    ['cibench_generation/pytorch', 'text_score', _cibench_generation_weight['pytorch'][2]],
    ['cibench_generation/pytorch', 'vis_sim', _cibench_generation_weight['pytorch'][3]],
    ['cibench_template/pytorch', 'numeric_correct', _cibench_template_weight['pytorch'][1]],
    ['cibench_template/pytorch', 'text_score', _cibench_template_weight['pytorch'][2]],
    ['cibench_template/pytorch', 'vis_sim', _cibench_template_weight['pytorch'][3]],
    ['cibench_template/sklearn', 'numeric_correct', _cibench_template_weight['sklearn'][1]],
    ['cibench_template/sklearn', 'text_score', _cibench_template_weight['sklearn'][2]],
    ['cibench_template/sklearn', 'vis_sim', _cibench_template_weight['sklearn'][3]],
    ['cibench_template/tensorflow', 'numeric_correct', _cibench_template_weight['tensorflow'][1]],
    ['cibench_template/tensorflow', 'text_score', _cibench_template_weight['tensorflow'][2]],
    ['cibench_template/tensorflow', 'vis_sim', _cibench_template_weight['tensorflow'][3]],
    ['cibench_template/lightgbm', 'numeric_correct', _cibench_template_weight['lightgbm'][1]],
    ['cibench_template/lightgbm', 'text_score', _cibench_template_weight['lightgbm'][2]],
    ['cibench_template/lightgbm', 'vis_sim', _cibench_template_weight['lightgbm'][3]],
]
cibench_nlp = [
    ['cibench_template/nltk', 'numeric_correct', _cibench_template_weight['nltk'][1]],
    ['cibench_template/nltk', 'text_score', _cibench_template_weight['nltk'][2]],
    ['cibench_template/nltk', 'vis_sim', _cibench_template_weight['nltk'][3]],
]
cibench_ip = [
    ['cibench_generation/opencv', 'numeric_correct', _cibench_generation_weight['opencv'][1]],
    ['cibench_generation/opencv', 'text_score', _cibench_generation_weight['opencv'][2]],
    ['cibench_generation/opencv', 'vis_sim', _cibench_generation_weight['opencv'][3]],
    ['cibench_template/opencv', 'numeric_correct', _cibench_template_weight['opencv'][1]],
    ['cibench_template/opencv', 'text_score', _cibench_template_weight['opencv'][2]],
    ['cibench_template/opencv', 'vis_sim', _cibench_template_weight['opencv'][3]],
]
cibench_math = [
    ['cibench_generation/scipy', 'numeric_correct', _cibench_generation_weight['scipy'][1]],
    ['cibench_generation/scipy', 'text_score', _cibench_generation_weight['scipy'][2]],
    ['cibench_generation/scipy', 'vis_sim', _cibench_generation_weight['scipy'][3]],
    ['cibench_template/scipy', 'numeric_correct', _cibench_template_weight['scipy'][1]],
    ['cibench_template/scipy', 'text_score', _cibench_template_weight['scipy'][2]],
    ['cibench_template/scipy', 'vis_sim', _cibench_template_weight['scipy'][3]],
]
cibench_summary_groups.extend([
    {
        'name': 'cibench_data_manipulation:scores',
        'subsets': [i[:2] for i in cibench_data_manipulation],
        'weights': {f'{k[0]}@{k[1]}': k[-1] for k in cibench_data_manipulation},
    },
    {
        'name': 'cibench_data_visualization:scores',
        'subsets': [i[:2] for i in cibench_data_visualization],
        'weights': {f'{k[0]}@{k[1]}': k[-1] for k in cibench_data_visualization},
    },
    {
        'name': 'cibench_modeling:scores',
        'subsets': [i[:2] for i in cibench_modeling],
        'weights': {f'{k[0]}@{k[1]}': k[-1] for k in cibench_modeling},
    },
    {
        'name': 'cibench_nlp:scores',
        'subsets': [i[:2] for i in cibench_nlp],
        'weights': {f'{k[0]}@{k[1]}': k[-1] for k in cibench_nlp},
    },
    {
        'name': 'cibench_ip:scores',
        'subsets': [i[:2] for i in cibench_ip],
        'weights': {f'{k[0]}@{k[1]}': k[-1] for k in cibench_ip},
    },
    {
        'name': 'cibench_math:scores',
        'subsets': [i[:2] for i in cibench_math],
        'weights': {f'{k[0]}@{k[1]}': k[-1] for k in cibench_math},
    },
])


########### New summerizer for Category metric oracle

cibench_data_manipulation = [
    ['cibench_generation_oracle/pandas', 'numeric_correct', _cibench_generation_weight['pandas'][1]],
    ['cibench_generation_oracle/pandas', 'text_score', _cibench_generation_weight['pandas'][2]],
    ['cibench_generation_oracle/pandas', 'vis_sim', _cibench_generation_weight['pandas'][3]],
    ['cibench_template_oracle/pandas', 'numeric_correct', _cibench_template_weight['pandas'][1]],
    ['cibench_template_oracle/pandas', 'text_score', _cibench_template_weight['pandas'][2]],
    ['cibench_template_oracle/pandas', 'vis_sim', _cibench_template_weight['pandas'][3]],
]
cibench_data_visualization = [
    ['cibench_generation_oracle/matplotlib', 'numeric_correct', _cibench_generation_weight['matplotlib'][1]],
    ['cibench_generation_oracle/matplotlib', 'text_score', _cibench_generation_weight['matplotlib'][2]],
    ['cibench_generation_oracle/matplotlib', 'vis_sim', _cibench_generation_weight['matplotlib'][3]],
    ['cibench_generation_oracle/seaborn', 'numeric_correct', _cibench_generation_weight['seaborn'][1]],
    ['cibench_generation_oracle/seaborn', 'text_score', _cibench_generation_weight['seaborn'][2]],
    ['cibench_generation_oracle/seaborn', 'vis_sim', _cibench_generation_weight['seaborn'][3]],
    ['cibench_template_oracle/matplotlib', 'numeric_correct', _cibench_template_weight['matplotlib'][1]],
    ['cibench_template_oracle/matplotlib', 'text_score', _cibench_template_weight['matplotlib'][2]],
    ['cibench_template_oracle/matplotlib', 'vis_sim', _cibench_template_weight['matplotlib'][3]],
    ['cibench_template_oracle/seaborn', 'numeric_correct', _cibench_template_weight['seaborn'][1]],
    ['cibench_template_oracle/seaborn', 'text_score', _cibench_template_weight['seaborn'][2]],
    ['cibench_template_oracle/seaborn', 'vis_sim', _cibench_template_weight['seaborn'][3]],
]
cibench_modeling = [
    ['cibench_generation_oracle/pytorch', 'numeric_correct', _cibench_generation_weight['pytorch'][1]],
    ['cibench_generation_oracle/pytorch', 'text_score', _cibench_generation_weight['pytorch'][2]],
    ['cibench_generation_oracle/pytorch', 'vis_sim', _cibench_generation_weight['pytorch'][3]],
    ['cibench_template_oracle/pytorch', 'numeric_correct', _cibench_template_weight['pytorch'][1]],
    ['cibench_template_oracle/pytorch', 'text_score', _cibench_template_weight['pytorch'][2]],
    ['cibench_template_oracle/pytorch', 'vis_sim', _cibench_template_weight['pytorch'][3]],
    ['cibench_template_oracle/sklearn', 'numeric_correct', _cibench_template_weight['sklearn'][1]],
    ['cibench_template_oracle/sklearn', 'text_score', _cibench_template_weight['sklearn'][2]],
    ['cibench_template_oracle/sklearn', 'vis_sim', _cibench_template_weight['sklearn'][3]],
    ['cibench_template_oracle/tensorflow', 'numeric_correct', _cibench_template_weight['tensorflow'][1]],
    ['cibench_template_oracle/tensorflow', 'text_score', _cibench_template_weight['tensorflow'][2]],
    ['cibench_template_oracle/tensorflow', 'vis_sim', _cibench_template_weight['tensorflow'][3]],
    ['cibench_template_oracle/lightgbm', 'numeric_correct', _cibench_template_weight['lightgbm'][1]],
    ['cibench_template_oracle/lightgbm', 'text_score', _cibench_template_weight['lightgbm'][2]],
    ['cibench_template_oracle/lightgbm', 'vis_sim', _cibench_template_weight['lightgbm'][3]],
]
cibench_nlp = [
    ['cibench_template_oracle/nltk', 'numeric_correct', _cibench_template_weight['nltk'][1]],
    ['cibench_template_oracle/nltk', 'text_score', _cibench_template_weight['nltk'][2]],
    ['cibench_template_oracle/nltk', 'vis_sim', _cibench_template_weight['nltk'][3]],
]
cibench_ip = [
    ['cibench_generation_oracle/opencv', 'numeric_correct', _cibench_generation_weight['opencv'][1]],
    ['cibench_generation_oracle/opencv', 'text_score', _cibench_generation_weight['opencv'][2]],
    ['cibench_generation_oracle/opencv', 'vis_sim', _cibench_generation_weight['opencv'][3]],
    ['cibench_template_oracle/opencv', 'numeric_correct', _cibench_template_weight['opencv'][1]],
    ['cibench_template_oracle/opencv', 'text_score', _cibench_template_weight['opencv'][2]],
    ['cibench_template_oracle/opencv', 'vis_sim', _cibench_template_weight['opencv'][3]],
]
cibench_math = [
    ['cibench_generation_oracle/scipy', 'numeric_correct', _cibench_generation_weight['scipy'][1]],
    ['cibench_generation_oracle/scipy', 'text_score', _cibench_generation_weight['scipy'][2]],
    ['cibench_generation_oracle/scipy', 'vis_sim', _cibench_generation_weight['scipy'][3]],
    ['cibench_template_oracle/scipy', 'numeric_correct', _cibench_template_weight['scipy'][1]],
    ['cibench_template_oracle/scipy', 'text_score', _cibench_template_weight['scipy'][2]],
    ['cibench_template_oracle/scipy', 'vis_sim', _cibench_template_weight['scipy'][3]],
]
cibench_summary_groups.extend([
    {
        'name': 'cibench_data_manipulation_oracle:scores',
        'subsets': [i[:2] for i in cibench_data_manipulation],
        'weights': {f'{k[0]}@{k[1]}': k[-1] for k in cibench_data_manipulation},
    },
    {
        'name': 'cibench_data_visualization_oracle:scores',
        'subsets': [i[:2] for i in cibench_data_visualization],
        'weights': {f'{k[0]}@{k[1]}': k[-1] for k in cibench_data_visualization},
    },
    {
        'name': 'cibench_modeling_oracle:scores',
        'subsets': [i[:2] for i in cibench_modeling],
        'weights': {f'{k[0]}@{k[1]}': k[-1] for k in cibench_modeling},
    },
    {
        'name': 'cibench_nlp_oracle:scores',
        'subsets': [i[:2] for i in cibench_nlp],
        'weights': {f'{k[0]}@{k[1]}': k[-1] for k in cibench_nlp},
    },
    {
        'name': 'cibench_ip_oracle:scores',
        'subsets': [i[:2] for i in cibench_ip],
        'weights': {f'{k[0]}@{k[1]}': k[-1] for k in cibench_ip},
    },
    {
        'name': 'cibench_math_oracle:scores',
        'subsets': [i[:2] for i in cibench_math],
        'weights': {f'{k[0]}@{k[1]}': k[-1] for k in cibench_math},
    },
])
