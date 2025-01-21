# INSTRUCTIONS = {
#     'hu': """
#     The following question is in hungarian language on {subject}, please read the question, and try to fill in the blank by chosing appropriate option from the option list. Please organize the answer in a list. An example:
#     {
#         "q_main": "Egészítsd ki a Janus Pannonius életére vonatkozó rövid szöveget! Segítségként használd az internetet! Vigyázz, nem minden szót kell felhasználnod!\nJanus Pannonius nem csupán költőként volt jelentős személyisége kora Magyarországának. #0# unokaöccseként a politikából is hamar kivette a részét. #1# tanulmányai után pécsi #2# lett, majd a királyné mellett #3#. Főkincstartóként és a #4# báni cím elnyerésével komoly politikai karriert futott be Mátyás király udvarában. A királlyal megromló kapcsolata miatt részt vett a #5# elleni összeesküvésben, ezért menekülnie kellett. Ez, és az akkor már súlyosbodó betegsége okozta halálát #6#.",
#         "options": ["A.érsek", "B.szlavón", "C.Vitéz János", "D.püspök", "E.főpohárnok", "F.Ulászló", "G.1474-ben", "H.főkancellár", "I.Itáliai", "J.Kinizsi Pál", "K.Kálmán", "L.1472-ben", "M.Prágai", "N.Mátyás"],
#         "std_ans": ["#0#C", "#1#I", "#2#D", "#3#H", "#4#B", "#5#N", "#6#L"],
#     }
#     Now try to answer the following question, your response should be in a JSON format. Contain the std_ans like the case given above.
#     The question is: {question}.
#     """,
#     'version':'V1',
#     'description': 'Initial version, using 1shot, incontext, #0# as place holder, output in JSON format',
# }

INSTRUCTIONS = {
    'hu': """
    You are a native hungarian teacher. The following question is in hungarian language on {subject}. Please read the question, and You need to choose the appropriate option from the provided "option" list to fill in each blanks in the text based on the context. Read the entire text, then fill in the blanks. Some options can be selected repeatedly. Please organize the answer in a list. An example:
    {
        "q_main": "Egészítsd ki a Janus Pannonius életére vonatkozó rövid szöveget! Segítségként használd az internetet! Vigyázz, nem minden szót kell felhasználnod!\nJanus Pannonius nem csupán költőként volt jelentős személyisége kora Magyarországának. #0# unokaöccseként a politikából is hamar kivette a részét. #1# tanulmányai után pécsi #2# lett, majd a királyné mellett #3#. Főkincstartóként és a #4# báni cím elnyerésével komoly politikai karriert futott be Mátyás király udvarában. A királlyal megromló kapcsolata miatt részt vett a #5# elleni összeesküvésben, ezért menekülnie kellett. Ez, és az akkor már súlyosbodó betegsége okozta halálát #6#.",
        "options": ["A.érsek", "B.szlavón", "C.Vitéz János", "D.püspök", "E.főpohárnok", "F.Ulászló", "G.1474-ben", "H.főkancellár", "I.Itáliai", "J.Kinizsi Pál", "K.Kálmán", "L.1472-ben", "M.Prágai", "N.Mátyás"],
    },
    The answer is:
    {
        "std_ans": ["#0#C", "#1#I", "#2#D", "#3#H", "#4#B", "#5#N", "#6#L"]
    }
    Now try to answer the following question, your response should be in a JSON format. Contain the std_ans like the case given above.
    The question is: {question}.
    """,
    'version':'V2',
    'description': 'Version 2, using 1shot, more incontext, "#0#" as place holder, output in JSON format'
}

DATASET_PATH = "/mnt/hwfile/opendatalab/weixingjian/test/"
