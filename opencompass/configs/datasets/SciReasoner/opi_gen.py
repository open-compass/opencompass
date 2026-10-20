# base config for opi
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import opi_postprocess, Opi_Evaluator, OpiDataset


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

    reader_cfg = dict(input_columns=['input'], output_column='output')

    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt='{input}'),
            ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer,
        )
    )

    # Extract high-level task from subdir name for the evaluator (e.g., 'EC_number')
    task_type = subtask_name.split('_')[0]

    eval_cfg = dict(
        evaluator=dict(type=Opi_Evaluator, task=task_type),
        pred_postprocessor=dict(type=opi_postprocess, task=task_type),
        dataset_postprocessor=dict(type=opi_postprocess, task=task_type),
    )

    # Create the dataset dictionary for the current subtask
    all_datasets.append(
        dict(
            abbr=f'SciReasoner-Opi_{subtask_name}',
            type=OpiDataset,
            path='opencompass/SciReasoner-OPI',
            task=subtask_name,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg
        ).copy()
    )
    mini_all_datasets.append(
        dict(
            abbr=f'SciReasoner-Opi_{subtask_name}-mini',
            type=OpiDataset,
            path='opencompass/SciReasoner-OPI',
            task=subtask_name,
            mini_set=True,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg
        ).copy()
    )
