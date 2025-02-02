INSTRUCTION = {
    'prompt_template':
    """The following questions are in Hungarian language on {hu_specific_dim}, please read the questions, and try to fill in the blanks in the question list. Please organize the answer in a list. An example:
{
    "instruction": "Írd be a megfelelő meghatározás mellé a fogalmat!",
    "questions": ["A.A szerzetesi közösségek szabályzatának elnevezése latinul: #0#", "B.Az első ún. kolduló rend: #1#", "C.A szerzetesek által kézzel másolt mű: #2#", "D.Papi nőtlenség: #3#", "E.A pápát megválasztó egyházi méltóságok: #4#", "F.A bencés rend megújítása ebben a kolostorban kezdődött a 10. században: #5#"],
}
The answers are:
{
    "answers": ["#0#regula", "#1#ferencesrend", "#2#kódex", "#3#cölibátus", "#4#bíborosok", "#5#Cluny"]
}
Now try to answer the following questions, your response should be in a JSON format. Contain the "answers" like the case given above.
The questions are:
{
    "instruction": {instruction},
    "questions": {questions},
}
""",
    'version':
    'V1',
    'description':
    'Initial version, using 1shot, incontext, #0# as place holder, output in JSON format',
}

OpenHuEval_Path = '/mnt/hwfile/opendatalab/wj/proj/polyglot_24July/OpenHuEval'
DATA_VERSION = '250126'
DATA_PATH = f'{OpenHuEval_Path}/data/HuStandardFIB/HuStandardFIB_{DATA_VERSION}/HuStandardFIB.jsonl'
