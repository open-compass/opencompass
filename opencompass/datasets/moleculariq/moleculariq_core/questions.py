SYSTEM_PROMPTS = {
    "with_key_hints": """You are an expert chemist specializing in molecular understanding, property calculations, structural analysis and molecular generation.

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
- Example: <answer>{"smiles": "CC(=O)CC(O)C"}</answer>""",


    "concise": """You are an expert chemist. Answer molecular property, understanding, structural analysis and molecular generation questions precisely and accurately.

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
}

TASKS = {
    "constraint_generation": {
        "task_name": "constraint_generation",
        "task_type": "generation",
        "description": "Generate a molecule satisfying given constraints",
        "difficulty": "medium",
        "output_type": "SMILES",
        "input_fields": ["constraint"],
        "output_fields": ["smiles"],
        "question_templates": [
            "Generate a molecule with {constraint}.",
            "Create a compound with {constraint}.",
            "Design a molecule that has {constraint}.",
            "Synthesize a structure fulfilling {constraint}.",
            "Build a molecule containing {constraint}.",
            "Can you generate a molecule that satisfies {constraint}?",
            "Please create a chemical structure with {constraint}.",
            "I need a molecule meeting the requirement of {constraint}.",
            "Construct a molecule having {constraint}.",
            "Provide a SMILES representation for a molecule with {constraint}.",
            "What would be a valid molecule that achieves {constraint}?",
            "Show me a compound conforming to {constraint}.",
            "Generate a chemical entity with {constraint}.",
            "Propose a molecular structure that respects {constraint}.",
            "Produce a compound that ensures {constraint}.",
            # -------------------- NEW TEMPLATES BELOW --------------------
            "Suggest a molecule that complies with {constraint}.",
            "Return a SMILES string for a compound satisfying {constraint}.",
            "Find a valid molecular structure consistent with {constraint}.",
            "Devise a chemical structure that adheres to {constraint}.",
            "Offer one example molecule that satisfies {constraint}.",
            "Give a candidate compound meeting {constraint}.",
            "Formulate a molecule engineered to satisfy {constraint}.",
            "Draft a molecular scaffold that meets {constraint}.",
            "Assemble a structure that obeys {constraint}.",
            "Sketch a compound compatible with {constraint}.",
            "Recommend a molecule that fulfills {constraint}.",
            "Output any valid molecule honoring {constraint}.",
            "Provide one SMILES for a molecule achieving {constraint}.",
            "Yield a chemical species consistent with {constraint}.",
            "Come up with a compound that matches {constraint}."
        ]
    },
    "single_count": {
        "task_name": "single_count",
        "task_type": "count",
        "description": "Count a single molecular feature",
        "difficulty": "easy",
        "output_type": "INTEGER",
        "input_fields": ["smiles", "count_type"],
        "output_fields": ["count"],
        "question_templates": [
            "Count the number of {count_type} in the molecule {smiles}.",
            "How many {count_type} are present in {smiles}?",
            "What is the total count of {count_type} in {smiles}?",
            "Determine the exact number of {count_type} in the structure {smiles}.",
            "For the molecule {smiles}, calculate how many {count_type} it contains.",
            "Identify and count all {count_type} in {smiles}.",
            "In the molecule {smiles}, what is the count of {count_type}?",
            "Provide the total number of {count_type} found in {smiles}.",
            "Calculate the {count_type} count for the molecule {smiles}.",
            "How many {count_type} does the molecule {smiles} have?",
            "Find the number of {count_type} in {smiles}.",
            "Count all occurrences of {count_type} in the structure {smiles}.",
            "What is the {count_type} count in {smiles}?",
            "Analyze {smiles} and report the number of {count_type}.",
            "For {smiles}, determine the total {count_type} count.",
            # -------------------- NEW TEMPLATES BELOW --------------------
            "Report how many {count_type} occur in {smiles}.",
            "Return the count of {count_type} for {smiles}.",
            "Tally the instances of {count_type} in {smiles}.",
            "Enumerate the number of {count_type} present in {smiles}.",
            "Give the quantity of {count_type} in molecule {smiles}.",
            "Compute how many {count_type} appear in {smiles}.",
            "What count of {count_type} does {smiles} contain?",
            "State the number of {count_type} within {smiles}.",
            "Return the integer count of {count_type} for the structure {smiles}.",
            "How many occurrences of {count_type} are in the SMILES {smiles}?",
            "Quantify the {count_type} present in {smiles}.",
            "Give the exact count of {count_type} in {smiles}.",
            "Determine and output the number of {count_type} in {smiles}.",
            "Return a tally of {count_type} in the compound {smiles}."
        ]
    },
    "multi_count": {
        "task_name": "multi_count",
        "task_type": "count",
        "description": "Count multiple molecular features",
        "difficulty": "medium",
        "output_type": "DICT",
        "input_fields": ["smiles", "count_types"],
        "output_fields": ["counts"],
        "question_templates": [
            "For the molecule {smiles}, count the following features: {count_types}.",
            "Count each of these features in {smiles}: {count_types}.",
            "Determine the counts for {count_types} in the molecule {smiles}.",
            "In {smiles}, provide the exact counts for: {count_types}.",
            "Calculate the numbers of {count_types} present in {smiles}.",
            "For the structure {smiles}, count: {count_types}.",
            "What are the counts of {count_types} in the molecule {smiles}?",
            "Analyze {smiles} and count the following: {count_types}.",
            "List the counts for each of these features in {smiles}: {count_types}.",
            "Count the occurrences of {count_types} in {smiles}.",
            "For {smiles}, determine how many of each feature: {count_types}.",
            "Provide counts for {count_types} in the structure {smiles}.",
            "In the molecule {smiles}, count these features: {count_types}.",
            "Calculate how many of each: {count_types} are in {smiles}.",
            "Report the counts of {count_types} for the molecule {smiles}.",
            # -------------------- NEW TEMPLATES BELOW --------------------
            "Return the counts for each of {count_types} in {smiles}.",
            "List how many of {count_types} occur in {smiles}.",
            "Give the count for each item in {count_types} for {smiles}.",
            "Provide the count for every feature in {count_types} for molecule {smiles}.",
            "Output the counts for {count_types} found in {smiles}.",
            "Provide the counts for all listed features in {smiles}.",
            "How many of each listed feature in {count_types} occur in {smiles}?",
            "Report how many of each {count_types} are present in {smiles}.",
            "List the count for each of {count_types} found in {smiles}.",
            "Compute the counts for {count_types} in the SMILES {smiles}.",
            "State the count for each feature in {count_types} in {smiles}.",
            "Determine and return how many of each {count_types} are in compound {smiles}.",
            "Summarize the counts for {count_types} detected in {smiles}.",
            "Produce the counts for all requested features {count_types} in {smiles}.",
            "Calculate the count for each item in {count_types} for {smiles}."
        ]
    },
    "single_index_identification": {
        "task_name": "single_index_identification",
        "task_type": "index",
        "description": "Identify indices of specific atoms or features",
        "difficulty": "medium",
        "output_type": "LIST",
        "input_fields": ["smiles", "index_type"],
        "output_fields": ["indices"],
        "question_templates": [
            "Identify the indices of {index_type} in the molecule {smiles}.",
            "For the molecule {smiles}, what are the indices of {index_type}?",
            "In {smiles}, identify the indices of {index_type}.",
            "What are the indices of {index_type} in {smiles}?",
            "List the indices of {index_type} in the structure {smiles}.",
            "Find the indices of {index_type} in {smiles}.",
            "Determine the indices of {index_type} in the molecule {smiles}.",
            "Which atom indices represent {index_type} in {smiles}?",
            "Locate the indices of {index_type} in the compound {smiles}.",
            "In the SMILES {smiles}, what are the indices of {index_type}?",
            "Point out the indices of {index_type} in {smiles}.",
            "Identify the {index_type} indices in the structure {smiles}.",
            "For {smiles}, provide the indices of {index_type}.",
            "What are the atom indices for {index_type} in {smiles}?",
            "Show the indices of {index_type} in the molecule {smiles}.",
            # -------------------- NEW TEMPLATES BELOW --------------------
            "Return the atom indices corresponding to {index_type} in {smiles}.",
            "List all index positions of {index_type} for molecule {smiles}.",
            "Which indices in {smiles} correspond to {index_type}?",
            "Provide the positions (indices) of {index_type} in the structure {smiles}.",
            "Give the set of atom indices for {index_type} in {smiles}.",
            "Report indices where {index_type} occurs in {smiles}.",
            "Find and return the indices for {index_type} within {smiles}.",
            "Output the index list of {index_type} in compound {smiles}.",
            "Enumerate indices that match {index_type} in {smiles}.",
            "Show atom positions (as indices) for {index_type} in {smiles}.",
            "Locate and list indices of {index_type} in the SMILES {smiles}.",
            "Extract the indices associated with {index_type} in {smiles}.",
            "Identify and return indices for {index_type} across {smiles}.",
            "Point out which atom indices represent {index_type} for {smiles}.",
            "Provide all indices referring to {index_type} in {smiles}."
        ]
    },
    "multi_index_identification": {
        "task_name": "multi_index_identification",
        "task_type": "index",
        "description": "Identify indices for multiple atom types or features",
        "difficulty": "hard",
        "output_type": "DICT",
        "input_fields": ["smiles", "index_types"],
        "output_fields": ["index_mappings"],
        "question_templates": [
            "For the molecule {smiles}, identify the atom indices for: {index_types}.",
            "Analyze the structure {smiles} and provide indices for: {index_types}.",
            "What are the atom positions for {index_types} in the compound {smiles}?",
            "For {smiles}, please locate the indices of: {index_types}.",
            "Find the atom indices for these features in molecule {smiles}: {index_types}.",
            "Determine the atom positions for {index_types} in {smiles}.",
            "In the structure {smiles}, what are the indices for: {index_types}?",
            "Identify atom indices for {index_types} in {smiles}.",
            "For compound {smiles}, locate: {index_types}.",
            "Analyze {smiles} and report atom indices for: {index_types}.",
            "What are the positions of {index_types} in molecule {smiles}?",
            "Find the {index_types} atom indices present in {smiles}.",
            "For the chemical structure {smiles}, identify: {index_types}.",
            "Provide atom indices for {index_types} in the molecule {smiles}.",
            "Where are the {index_types} located in {smiles}?",
            # -------------------- NEW TEMPLATES BELOW --------------------
            "Return the indices for each feature in {index_types} in {smiles}.",
            "Provide the indices for each of {index_types} in molecule {smiles}.",
            "List the index list for each of {index_types} in the structure {smiles}.",
            "For {smiles}, output the indices corresponding to each of: {index_types}.",
            "Report where (by index) {index_types} occur in {smiles}.",
            "Give index positions for every requested type among {index_types} in {smiles}.",
            "Find and return indices for all of {index_types} within {smiles}.",
            "State the atom indices associated with {index_types} in compound {smiles}.",
            "Enumerate index sets for {index_types} detected in {smiles}.",
            "Show the atom indices for each listed feature {index_types} in the SMILES {smiles}.",
            "Extract the indices for each feature listed in {index_types} from {smiles}.",
            "Locate indices matching {index_types} across the structure {smiles}.",
            "Provide the index sets for {index_types} present in {smiles}.",
            "List the indices for each of {index_types} in {smiles}.",
            "Return atom positions (indices) for all {index_types} in {smiles}."
        ]
    }
}
