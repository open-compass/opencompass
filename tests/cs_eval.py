from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('cseval/cs-eval', subset_name='default', split='test')

print(ds)