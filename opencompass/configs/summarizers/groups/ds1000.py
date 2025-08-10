ds1000_summary_groups = []

_ds1000_all = ['Pandas', 'Numpy', 'Tensorflow', 'Scipy', 'Sklearn', 'Pytorch', 'Matplotlib']
_ds1000_all = ['ds1000_' + d for d in _ds1000_all]
ds1000_summary_groups.append({'name': 'ds1000', 'subsets': _ds1000_all})
