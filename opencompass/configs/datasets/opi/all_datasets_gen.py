# base config for opi
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import opi_postprocess, opi_Evaluator, opiDataset

import os

all_datasets = []
mini_all_datasets = []

# Root directory where the datasets are located
root_dir = '/path/OPI_test'

subtask_dirs = [
    'EC_number_CLEAN_EC_number_new',
    'EC_number_CLEAN_EC_number_price',
    'Fold_type_fold_type',
    'Function_CASPSimilarSeq_function',
    'Function_IDFilterSeq_function',
    'Function_UniProtSeq_function',
    'gName2Cancer_gene_name_to_cancer',
    'GO_CASPSimilarSeq_go',
    'GO_IDFilterSeq_go',
    'GO_UniProtSeq_go',
    'gSymbol2Cancer_gene_symbol_to_cancer',
    'gSymbol2Tissue_gene_symbol_to_tissue',
    'Keywords_CASPSimilarSeq_keywords',
    'Keywords_IDFilterSeq_keywords',
    'Keywords_UniProtSeq_keywords',
    'Subcellular_localization_subcell_loc',
]

for subtask_name in subtask_dirs:
    # Common configs for inference
    generation_kwargs = dict(
        temperature=0.7,
        top_k=50,
        top_p=0.8,
        num_beams=1,
        # max_new_tokens=1024,
        do_sample=True,
        # use_cache=True,
        # eos_token_id=[151643, 151645],  # for custom models
    )

    reader_cfg = dict(input_columns=['input'], output_column='output')

    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template='{input}\n{output}',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer,
            # max_out_len=2048,
            generation_kwargs=generation_kwargs,
        )
    )

    # Extract high-level task from subdir name for the evaluator (e.g., 'EC_number')
    task_type = subtask_name.split('_')[0]

    eval_cfg = dict(
        evaluator=dict(type=opi_Evaluator, task=task_type),
        pred_postprocessor=dict(type=opi_postprocess, task=task_type),
        dataset_postprocessor=dict(type=opi_postprocess, task=task_type),
    )

    # Construct paths assuming a dev/data.json and test/data.json structure
    train_path = os.path.join(root_dir, subtask_name, 'dev', 'data.json')
    test_path = os.path.join(root_dir, subtask_name, 'test', 'data.json')

    # Create the dataset dictionary for the current subtask
    all_datasets.append(
        dict(
            abbr=subtask_name,
            type=opiDataset,
            train_path=train_path,
            test_path=test_path,
            hf_hub=False,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg
        ).copy()
    )
    mini_all_datasets.append(
        dict(
            abbr=f'{subtask_name}-mini',
            type=opiDataset,
            train_path=train_path,
            test_path=test_path,
            mini_set=True,
            hf_hub=False,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg
        ).copy()
    )
