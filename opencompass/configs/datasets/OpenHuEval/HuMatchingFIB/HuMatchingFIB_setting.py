INSTRUCTIONS = {
    'en':
    """You are a native Hungarian teacher. The following question is in Hungarian language on {hu_specific_dim}. Please read the question, and choose the appropriate option from the provided "options" list to fill in each blanks in the text based on the context. Read the entire text, then fill in the blanks. Some options can be selected repeatedly. Please organize the answer in a list. An example:
{
    "question": "Egészítsd ki a Janus Pannonius életére vonatkozó rövid szöveget! Segítségként használd az internetet! Vigyázz, nem minden szót kell felhasználnod!\nJanus Pannonius nem csupán költőként volt jelentős személyisége kora Magyarországának. #0# unokaöccseként a politikából is hamar kivette a részét. #1# tanulmányai után pécsi #2# lett, majd a királyné mellett #3#. Főkincstartóként és a #4# báni cím elnyerésével komoly politikai karriert futott be Mátyás király udvarában. A királlyal megromló kapcsolata miatt részt vett a #5# elleni összeesküvésben, ezért menekülnie kellett. Ez, és az akkor már súlyosbodó betegsége okozta halálát #6#.",
    "options": ["A.érsek", "B.szlavón", "C.Vitéz János", "D.püspök", "E.főpohárnok", "F.Ulászló", "G.1474-ben", "H.főkancellár", "I.Itáliai", "J.Kinizsi Pál", "K.Kálmán", "L.1472-ben", "M.Prágai", "N.Mátyás"],
},
The answer is:
{
    "answer": ["#0#C", "#1#I", "#2#D", "#3#H", "#4#B", "#5#N", "#6#L"]
}
Now try to answer the following question, your response should be in a JSON format. Contain the "answer" like the case given above.
The question and options are:
{
    "question": {question},
    "options": {options},
}
""",
    'hu':
    """Egy magyar anyanyelvű tanár vagy. Az alábbi kérdés magyar nyelven érdeklődik a(z) {hu_specific_dim} témakörben. Kérlek, olvasd el a kérdést, és válaszd ki a megfelelő opciót a megadott "options" listából, hogy kitöltsd a szövegben lévő hiányosságokat a kontextus alapján. Olvasd el az egész szöveget, majd töltsd ki az üres helyeket. Néhány opciót többször is kiválaszthatsz. A választ listában szervezd. Példa:
{
    "question": "Egészítsd ki a Janus Pannonius életére vonatkozó rövid szöveget! Segítségként használd az internetet! Vigyázz, nem minden szót kell felhasználnod!\nJanus Pannonius nem csupán költőként volt jelentős személyisége kora Magyarországának. #0# unokaöccseként a politikából is hamar kivette a részét. #1# tanulmányai után pécsi #2# lett, majd a királyné mellett #3#. Főkincstartóként és a #4# báni cím elnyerésével komoly politikai karriert futott be Mátyás király udvarában. A királlyal megromló kapcsolata miatt részt vett a #5# elleni összeesküvésben, ezért menekülnie kellett. Ez, és az akkor már súlyosbodó betegsége okozta halálát #6#.",
    "options": ["A.érsek", "B.szlavón", "C.Vitéz János", "D.püspök", "E.főpohárnok", "F.Ulászló", "G.1474-ben", "H.főkancellár", "I.Itáliai", "J.Kinizsi Pál", "K.Kálmán", "L.1472-ben", "M.Prágai", "N.Mátyás"],
},
A válasz:
{
    "answer": ["#0#C", "#1#I", "#2#D", "#3#H", "#4#B", "#5#N", "#6#L"]
}
Most próbáld megválaszolni a következő kérdést, a válaszod JSON formátumban legyen. Tartalmazza az "answer" mezőt, ahogy az a fenti példában is látható.
A kérdés és az opciók:
{
    "question": {question},
    "options": {options},
}
""",
}

OpenHuEval_Path = '/mnt/hwfile/opendatalab/weixingjian/OpenHuEval'
DATA_VERSION = '250205'
DATA_PATH = f'{OpenHuEval_Path}/data/HuMatchingFIB/HuMatchingFIB_{DATA_VERSION}/HuMatchingFIB.jsonl'
