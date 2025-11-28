from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.biodata import BiodataTaskDataset
from opencompass.datasets.biodata import BiodataMCCEvaluator, BiodataPCCEvaluator, BiodataSpearmanEvaluator, \
    BiodataR2Evaluator, BiodataAucEvaluator, BiodataAccEvaluator, BiodataECNumberEvaluator, BiodataMixedScoreEvaluator

biodata_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='ground_truth'
)

tasks = {
    'MCC': [
        'DNA-cpd',
        'DNA-emp',
        'DNA-pd',
        'DNA-tf-h',
        'DNA-tf-m',
        'Multi_sequence-antibody_antigen',
        'Multi_sequence-promoter_enhancer_interaction',
        'Multi_sequence-rna_protein_interaction'
    ],
    'PCC': [
        'DNA-enhancer_activity',
    ],
    'Spearman': [
        'RNA-CRISPROnTarget',
        'Protein-Fluorescence',
        'Protein-Stability',
        'Protein-Thermostability',
    ],
    'R2': [
        'RNA-Isoform',
        'RNA-MeanRibosomeLoading',
        'RNA-ProgrammableRNASwitches',
    ],
    'Auc': [
        'RNA-Modification'
    ],
    'Acc': [
        'Protein-Solubility',
        'RNA-NoncodingRNAFamily',
    ],
    'Fmax': [
        'Protein-FunctionEC'
    ],
    'Mixed': [
        'Multi_sequence-sirnaEfficiency'
    ]
}

biodata_task_datasets = []
for metric, task in tasks.items():
    biodata_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt=f'{{prompt}}'),
                dict(role='BOT', prompt='{ground_truth}\n')
            ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    if metric == 'MCC':
        biodata_eval_cfg = dict(
            evaluator=dict(type=BiodataMCCEvaluator),
        )
    elif metric == 'PCC':
        biodata_eval_cfg = dict(
            evaluator=dict(type=BiodataPCCEvaluator),
        )
    elif metric == 'Spearman':
        biodata_eval_cfg = dict(
            evaluator=dict(type=BiodataSpearmanEvaluator),
        )
    elif metric == 'R2':
        biodata_eval_cfg = dict(
            evaluator=dict(type=BiodataR2Evaluator),
        )
    elif metric == 'Auc':
        biodata_eval_cfg = dict(
            evaluator=dict(
                type=BiodataAucEvaluator,
                predefined_labels=[
                    'atoi', 'm6a', 'none', 'm1a', 'm5c', 'm5u', 'm6am', 'm7g', 'cm', 'am', 'gm', 'um', 'psi'
                ]
            ),
        )
    elif metric == 'Acc':
        biodata_eval_cfg = dict(
            evaluator=dict(type=BiodataAccEvaluator),
        )
    elif metric == 'Fmax':
        biodata_eval_cfg = dict(
            evaluator=dict(type=BiodataECNumberEvaluator),
        )
    elif metric == 'Mixed':
        biodata_eval_cfg = dict(
            evaluator=dict(type=BiodataMixedScoreEvaluator),
        )
    else:
        raise NotImplementedError

    for t in task:
        biodata_task_datasets.append(
            dict(
                abbr=f'{t}-sample_1k',
                type=BiodataTaskDataset,
                path='opencompass/biology-instruction',
                task=t,
                reader_cfg=biodata_reader_cfg,
                infer_cfg=biodata_infer_cfg,
                eval_cfg=biodata_eval_cfg,
            )
        )
