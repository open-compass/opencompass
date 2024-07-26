from mmengine.config import read_base
from copy import deepcopy
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator, AccEvaluator
from opencompass.datasets import MathBenchDataset, mathbench_postprocess
from opencompass.utils.text_postprocessors import first_option_postprocess

with read_base():
    from .mathbench_prompt import zero_shot_prompts, few_shot_prompts, mathbench_sets

# Max for this dataset is 4
num_shot = 4
# Generate reasoning path or not, only for single choice
with_reasoning = False
# Use circular evaluation or not
with_circular_eval = True
# Use PPL mode in single choice test or not
use_ppl_single_choice = False

assert 0 <= num_shot <= 4
if num_shot == 0:
    prompts = zero_shot_prompts
else:
    prompts = {name: p[- 2 * num_shot - 2:] for name, p in few_shot_prompts.items()}

mathbench_datasets = []
for _split in mathbench_sets:
    for _name in mathbench_sets[_split]:
        if 'single_choice' in _name:
            if with_reasoning:
                template_round = prompts[_name + '_with_reasoning']
            else:
                template_round = prompts[_name]
        else:
            template_round = prompts[_name]

        if 'single_choice' in _name:
            pred_postprocessor = dict(type=first_option_postprocess, options='ABCD')
        else:
            pred_postprocessor = dict(type=mathbench_postprocess, name=_name)

        if 'single_choice' in _name and with_circular_eval:
            evaluator = dict(type=CircularEvaluator)
        else:
            evaluator = dict(type=AccEvaluator)

        # assemble the final config
        mathbench_reader_cfg = dict(input_columns=['question'], output_column='answer')
        if use_ppl_single_choice and 'single_choice' in _name and not with_reasoning:
            template = {}
            for answer in ['A', 'B', 'C', 'D']:
                one_template_round = deepcopy(template_round)
                one_template_round['round'][-1]['prompt'] = one_template_round['round'][-1]['prompt'].format(answer=answer)
                template[answer] = dict(round=one_template_round)
            mathbench_infer_cfg = dict(
                prompt_template=dict(type=PromptTemplate, template=template),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=PPLInferencer),
            )
        else:
            mathbench_infer_cfg = dict(
                prompt_template=dict(type=PromptTemplate, template=dict(round=template_round)),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer, max_out_len=2048),
            )
        mathbench_eval_cfg = dict(evaluator=evaluator, pred_postprocessor=pred_postprocessor)

        mathbench_datasets.append(
            dict(
                abbr='mathbench-no_cot-' + _split + '-' + _name,
                type=MathBenchDataset,
                path=f'data/mathbench_v1/{_split}',
                name=_name,
                with_circular=with_circular_eval,
                reader_cfg=mathbench_reader_cfg,
                infer_cfg=mathbench_infer_cfg,
                eval_cfg=mathbench_eval_cfg,
            )
        )
