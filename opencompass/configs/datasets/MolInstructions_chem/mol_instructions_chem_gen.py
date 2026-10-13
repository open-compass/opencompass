from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.mol_instructions_chem import FTSEvaluator, MeteorEvaluator, MAEEvaluator
from opencompass.datasets import MolInstructionsDataset

mol_gen_reader_cfg = dict(
    input_columns=['input'],
    output_column='output'
)

mol_gen_system = {
    'RP': """You are an expert chemist. Given the {mol_type} representation of the reactants and the target product, your task is to predict the potential reagents that can achieve this transformation.""",
    'MG': """You are an expert chemist. Given the description of a molecule, your task is to generate the potential {mol_type} representation of the molecule.""",
    'FS': """You are an expert chemist. Given the {mol_type} representation of reactants and reagents, your task is to predict the potential product using your chemical reaction knowledge.""",
    'RS': """You are an expert chemist. Given the {mol_type} representation of the product, your task is to predict the potential reactants and reagents using your chemical reaction knowledge.""",
    'PP': """You are an expert chemist. Given the {mol_type} representation of the product, your task is to predict the asked property using your chemical reaction knowledge.""",
    'MC': """You are an expert chemist. Given the {mol_type} representation of the product, your task is to describe the molecule in natural language.""",
}

mol_gen_hint_dict = {
    'RP': """Your reply should contain the {mol_type} representation of the predicted reagents only, enclosed together within a single pair of <{mol_type}> and </{mol_type}> tags and separated by ".".""",
    'MG': """Your reply should contain the potential {mol_type} representation of the molecule wrapped in <{mol_type}> and </{mol_type}> tags.""",
    'FS': """Your reply should contain the {mol_type} representation of the predicted product wrapped in <{mol_type}> and </{mol_type}> tags.""",
    'RS': """Your reply should contain the {mol_type} representation of both reactants and reagents, and all reactants and reagents should be enclosed **together** within a single pair of <{mol_type}> and </{mol_type}> tags, separated by ".".""",
    'PP': """Your reply should contain the requested numeric value wrapped in \\boxed{{}}.""",
    'MC': """Your reply should contain a natural language description of the molecule.""",
}
name_dict = {
    'RP': 'reagent_prediction',
    'MG': 'description_guided_molecule_design',
    'FS': 'forward_reaction_prediction',
    'RS': 'retrosynthesis',
    'PP': 'property_prediction',
    'MC': 'molecular_description_generation',
}

mol_gen_selfies_datasets = []
for _name in name_dict:
    _hint = mol_gen_hint_dict[_name]
    _system = mol_gen_system[_name]
    mol_gen_selfies_0shot_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='SYSTEM', prompt=f'{_system.format(mol_type="SELFIES")}'),
                    dict(role='HUMAN', prompt=f'{_hint.format(mol_type="SELFIES")}\nQuestion: {{input}}\nAnswer: '),
                ]
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    if _name == 'PP':
        mol_gen_eval_cfg = dict(
            evaluator=dict(type=MAEEvaluator),
        )
    elif _name == 'MC':
        mol_gen_eval_cfg = dict(
            evaluator=dict(type=MeteorEvaluator),
        )
    else:
        mol_gen_eval_cfg = dict(
            evaluator=dict(type=FTSEvaluator, tag='SELFIES'),
        )
    mol_gen_selfies_datasets.append(
        dict(
            abbr=f'{_name}-selfies',
            type=MolInstructionsDataset,
            path='opencompass/mol-instructions',
            name=f'{name_dict[_name]}.jsonl',
            reader_cfg=mol_gen_reader_cfg,
            infer_cfg=mol_gen_selfies_0shot_infer_cfg,
            eval_cfg=mol_gen_eval_cfg,
        ))
del _name
