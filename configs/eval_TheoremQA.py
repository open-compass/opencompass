from mmengine.config import read_base

with read_base():
    from .models.mistral.hf_mistral_7b_v0_1 import models as hf_mistral_7b_v0_1_model
    from .models.mistral.hf_mistral_7b_v0_2 import models as hf_mistral_7b_v0_2_model
    from .models.hf_internlm.hf_internlm2_20b import models as hf_internlm2_20b_model
    from .models.hf_internlm.hf_internlm2_math_20b import models as hf_internlm2_math_20b_model

    from .datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import TheoremQA_datasets as datasets

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

work_dir = 'outputs/TheoremQA-5shot'


# dataset    version    metric    mode      mistral-7b-v0.1-hf    mistral-7b-v0.2-hf    internlm2-20b-hf    internlm2-math-20b-hf
# ---------  ---------  --------  ------  --------------------  --------------------  ------------------  -----------------------
# TheoremQA  6f0af8     score     gen                    18.00                 16.75               25.87                    30.88
