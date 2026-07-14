from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.moleculariq import MoleculariqDataset
from opencompass.datasets.moleculariq import (
    MoleculariqCountEvaluator,
    MoleculariqIndexEvaluator,
    MoleculariqGenerationEvaluator,
)

moleculariq_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='ground_truth',
)

# 论文 Table 1 使用的 system prompt（来自论文附录 B.2.2）
system_prompt = """You are an expert chemist. Answer molecular property, understanding, structural analysis and molecular generation questions precisely and accurately.

CRITICAL: Only content within<answer></answer> tags will be extracted. ALWAYS return JSON format.

KEY REQUIREMENT: Use EXACT key names from the question. Never modify or invent keys.

INDEXING: Atoms are indexed from 0 to the end of the SMILES string from left to right. Only heavy atoms (skip [H], include [2H]/[3H]).
Examples:
- "CCO": C(0), C(1), O(2)
- "CC(C)O": C(0), C(1), C(2), O(3)
- "CC(=O)N": C(0), C(1), O(2), N(3)

ABSENT FEATURES: Use 0 for counts, [] for indices. Never null or omit.

ALWAYS USE JSON with EXACT keys from the question:

Single count (key from question: "alcohol count"):<answer>"alcohol count": 2</answer>
<answer>"alcohol count": 0</answer> (if absent)

Single index (key from question: "ketone indices"):<answer>"ketone indices": [5]</answer>
<answer>"ketone indices": []</answer> (if absent)

Multiple properties (keys from question: "ring count", "halogen indices"):<answer>"ring count": 2, "halogen indices": [3, 7]</answer>
<answer>"ring count": 0, "halogen indices": []</answer> (if all absent)

Constraint generation:<answer>"smiles": "CC(O)C"</answer>

Include ALL requested properties. Never null or omit."""

# questions.py: with_key_hints
system_prompt_with_key_hints = """You are an expert chemist specializing in molecular understanding, property calculations, structural analysis and molecular generation.

CRITICAL: Only content within <answer></answer> tags will be extracted as your response. Everything outside these tags is ignored.

KEY REQUIREMENT: Always use the EXACT key names provided in the question. Do not modify or create your own keys.

IMPORTANT: If a requested feature is not present in the molecule, you MUST return 0 for counts or [] for indices. Never null or omit.

INDEXING RULES:
- Atom indices are 0-based
- Atoms are numbered from 0 in the order they appear in the SMILES string from left to right
- Regular hydrogens (implicit or explicit [H]) are NOT indexed
- Isotopes ([2H], [3H]) ARE indexed as they appear
- Examples:
    - "CCO": C(0), C(1), O(2)
    - "CC(C)O": C(0), C(1), C(2), O(3)
    - "CC(=O)N": C(0), C(1), O(2), N(3)

For SINGLE COUNT tasks:
- Return a JSON object with the EXACT key from the question
- Return 0 if the feature is absent
- Examples: <answer>{"alcohol_group_count": 2}</answer>
- For absent features: <answer>{"alcohol_group_count": 0}</answer>

For SINGLE INDEX tasks:
- Return a JSON object with the EXACT key from the question
- Return empty list [] if the feature is absent
- Examples: <answer>{"alcohol_group_indices": [3, 7]}</answer>
- For absent features: <answer>{"alcohol_group_indices": []}</answer>

For MULTIPLE COUNT tasks with key hints:
- Return a JSON object using the EXACT keys provided
- Each key maps to an integer count (0 if absent)
- Example: <answer>{"alcohol_group_count": 2, "ketone_group_count": 0}</answer>

For MULTIPLE INDEX tasks with key hints:
- Return a JSON object using the EXACT keys provided
- Each key maps to a list of indices (empty list [] if absent)
- Example: <answer>{"alcohol_group_indices": [3, 7], "ketone_group_indices": []}</answer>

For CONSTRAINT GENERATION tasks:
- Return a JSON object with "smiles" as the key
- Example: <answer>{"smiles": "CC(=O)CC(O)C"}</answer>"""

# questions.py: concise
system_prompt_concise = """You are an expert chemist. Answer molecular property, understanding, structural analysis and molecular generation questions precisely and accurately.

CRITICAL: Only content within <answer></answer> tags will be extracted. ALWAYS return JSON format.

KEY REQUIREMENT: Use EXACT key names from the question. Never modify or invent keys.

INDEXING: Atoms are indexed from 0 to the end of the SMILES string from left to right. Only heavy atoms (skip [H], include [2H]/[3H]).
Examples:
    - "CCO": C(0), C(1), O(2)
    - "CC(C)O": C(0), C(1), C(2), O(3)
    - "CC(=O)N": C(0), C(1), O(2), N(3)

ABSENT FEATURES: Use 0 for counts, [] for indices. Never null or omit.

ALWAYS USE JSON with EXACT keys from the question:

Single count (key from question: "alcohol_count"):
<answer>{"alcohol_count": 2}</answer>
<answer>{"alcohol_count": 0}</answer>  (if absent)

Single index (key from question: "ketone_indices"):
<answer>{"ketone_indices": [5]}</answer>
<answer>{"ketone_indices": []}</answer>  (if absent)

Multiple properties (keys from question: "ring_count", "halogen_indices"):
<answer>{"ring_count": 2, "halogen_indices": [3, 7]}</answer>
<answer>{"ring_count": 0, "halogen_indices": []}</answer>  (if all absent)

Constraint generation:
<answer>{"smiles": "CC(O)C"}</answer>

Include ALL requested properties. Never null or omit."""

_evaluator_map = {
    'count': MoleculariqCountEvaluator,
    'index': MoleculariqIndexEvaluator,
    'generation': MoleculariqGenerationEvaluator,
}

moleculariq_datasets = []
for _name in ['count', 'index', 'generation']:
    moleculariq_infer_cfg = dict(
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': '{prompt}'},
            ],
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    moleculariq_eval_cfg = dict(
        evaluator=dict(type=_evaluator_map[_name]),
    )

    moleculariq_datasets.append(
        dict(
            abbr=f'MolecularIQ-{_name}',
            type=MoleculariqDataset,
            name=_name,
            path='opencompass/MolecularIQ',
            reader_cfg=moleculariq_reader_cfg,
            infer_cfg=moleculariq_infer_cfg,
            eval_cfg=moleculariq_eval_cfg,
        )
    )