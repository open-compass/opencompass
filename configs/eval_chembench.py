from mmengine.config import read_base

with read_base():
    from .datasets.ChemBench.ChemBench_gen import chembench_datasets
    from .models.mistral.hf_mistral_7b_instruct_v0_2 import models

datasets = [*chembench_datasets]
models = [*models]

'''
dataset                           version    metric    mode      mistral-7b-instruct-v0.2-hf
--------------------------------  ---------  --------  ------  -----------------------------
ChemBench_Name_Conversion         d4e6a1     accuracy  gen                             45.43
ChemBench_Property_Prediction     d4e6a1     accuracy  gen                             47.11
ChemBench_Mol2caption             d4e6a1     accuracy  gen                             64.21
ChemBench_Caption2mol             d4e6a1     accuracy  gen                             35.38
ChemBench_Product_Prediction      d4e6a1     accuracy  gen                             38.67
ChemBench_Retrosynthesis          d4e6a1     accuracy  gen                             27
ChemBench_Yield_Prediction        d4e6a1     accuracy  gen                             27
ChemBench_Temperature_Prediction  d4e6a1     accuracy  gen                             26.73
ChemBench_Solvent_Prediction      d4e6a1     accuracy  gen                             32.67
'''
