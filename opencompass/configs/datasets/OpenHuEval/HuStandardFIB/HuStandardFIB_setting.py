INSTRUCTIONS = {
    'hu': """The following question is in hungarian language on {subject}, please read the question, and try to fill in the blank in the sub question list. Please organize the answer in a list. An example:
    {
        "q_main": "Írd be a megfelelő meghatározás mellé a fogalmat!",
        "q_sub": ["A.A szerzetesi közösségek szabályzatának elnevezése latinul: #0#", "B.Az első ún. kolduló rend: #1#", "C.A szerzetesek által kézzel másolt mű: #2#", "D.Papi nőtlenség: #3#", "E.A pápát megválasztó egyházi méltóságok: #4#", "F.A bencés rend megújítása ebben a kolostorban kezdődött a 10. században: #5#"],
        "formatted_std_ans": ["#0#regula", "#1#ferencesrend;ferences", "#2#kódex", "#3#cölibátus", "#4#bíborosok;bíboros", "#5#Cluny"]
    }
    Now try to answer the following question, your response should be in a JSON format. Contain the std_ans like the case given above.
    The question is: {question}.
    """,
    'version':'V1',
    'description': 'Initial version, using 1shot, incontext, #0# as place holder, output in JSON format',
}

DATASET_PATH = "/mnt/hwfile/opendatalab/weixingjian/test/test2/"