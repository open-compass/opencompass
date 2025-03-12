INSTRUCTIONS = {
    'en':
    """Question: {question}
Please provide your best answer to this question in Hungarian and indicate your confidence in your answer using a score from 0 to 100. Please provide your response in the following JSON format:
{
    "answer": "Your answer here",
    "confidence_score": number
}
""",
    'hu':
    """Kérdés: {question}
Kérjük, magyar nyelven adja meg a legjobb választ erre a kérdésre, és 0-tól 100-ig terjedő pontszámmal jelezze, hogy bízik a válaszában. Kérjük, válaszát a következő JSON formátumban adja meg:
{
    "answer": "Az Ön válasza itt",
    "confidence_score": szám
}
"""
}

OpenHuEval_Path = '/mnt/hwfile/opendatalab/MinerU4S/yanghaote/XYZ/OpenHuEval'
DATA_VERSION = '250208'
DATA_PATH = f'{OpenHuEval_Path}/data/HuSimpleQA/HuSimpleQA_{DATA_VERSION}/HuSimpleQA.jsonl'
