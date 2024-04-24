from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator
from opencompass.datasets import (
    MGSMSDataset, MGSM_Evaluator,
    mgsm_zh_postprocess, mgsm_en_postprocess,
    mgsm_bn_postprocess, mgsm_de_postprocess,
    mgsm_es_postprocess, mgsm_fr_postprocess,
    mgsm_ja_postprocess, mgsm_ru_postprocess,
    mgsm_sw_postprocess, mgsm_te_postprocess,
    mgsm_th_postprocess
)

mgsm_reader_cfg = dict(input_columns=['question'], output_column='answer')

# LANG_TO_INSTRUCTIONS = {
#     "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

# {question}""",
#     "bn": """এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.

# {question}""",
#     "de": """Löse dieses Mathematikproblem. Gib die Schritte zur Begründung an, bevor du die endgültige Antwort in der letzten Zeile alleine im Format "Antwort:" gibst. Füge nichts anderes als die ganzzahlige Antwort nach "Antwort:" hinzu.

# {question}""",
#     "es": """Resuelve este problema matemático. Proporciona los pasos de razonamiento antes de dar la respuesta final en la última línea por sí misma en el formato de "Respuesta:". No añadas nada más que la respuesta entera después de "Respuesta:".

# {question}""",
#     "fr": """Résolvez ce problème de mathématiques. Donnez les étapes de raisonnement avant de fournir la réponse finale sur la dernière ligne elle-même dans le format de "Réponse:". N'ajoutez rien d'autre que la réponse entière après "Réponse:".

# {question}""",
#     "ja": """の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。そして最後の行には "答え:" の形式で答えを記述し、その後には整数の答え以外何も追加しないでください。

# {question}""",
#     "ru": """Решите эту математическую задачу. Объясните шаги рассуждения перед тем, как дать окончательный ответ в последней строке сам по себе в формате "Ответ:". Не добавляйте ничего, кроме целочисленного ответа после "Ответ:".

# {question}""",
#     "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".

# {question}""",
#     "te": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.

# {question}""",
#     "th": """แก้ปัญหาคณิตศาสตร์นี้ ให้ให้ขั้นตอนการใช้เหตุผลก่อนที่จะให้คำตอบสุดท้ายในบรรทัดสุดท้ายโดยอยู่ในรูปแบบ "คำตอบ:" ไม่ควรเพิ่มอะไรนอกจากคำตอบที่เป็นจำนวนเต็มหลังจาก "คำตอบ:"

# {question}""",
#     "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。

# {question}""",
# }

# each instance in datafile
mgsm_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{question}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

mgsm_eval_en_cfg = dict(
    evaluator=dict(type=MGSM_Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=mgsm_en_postprocess),
)

mgsm_eval_bn_cfg = dict(
    evaluator=dict(type=MGSM_Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=mgsm_bn_postprocess),
)

mgsm_eval_de_cfg = dict(
    evaluator=dict(type=MGSM_Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=mgsm_de_postprocess),
)

mgsm_eval_es_cfg = dict(
    evaluator=dict(type=MGSM_Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=mgsm_es_postprocess),
)

mgsm_eval_fr_cfg = dict(
    evaluator=dict(type=MGSM_Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=mgsm_fr_postprocess),
)

mgsm_eval_ja_cfg = dict(
    evaluator=dict(type=MGSM_Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=mgsm_ja_postprocess),
)

mgsm_eval_ru_cfg = dict(
    evaluator=dict(type=MGSM_Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=mgsm_ru_postprocess),
)

mgsm_eval_sw_cfg = dict(
    evaluator=dict(type=MGSM_Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=mgsm_sw_postprocess),
)

mgsm_eval_te_cfg = dict(
    evaluator=dict(type=MGSM_Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=mgsm_te_postprocess),
)

mgsm_eval_th_cfg = dict(
    evaluator=dict(type=MGSM_Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=mgsm_th_postprocess),
)

mgsm_eval_zh_cfg = dict(
    evaluator=dict(type=MGSM_Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=mgsm_zh_postprocess),
)


#all datasets 
mgsm_datasets = [
    dict(
        type=MGSMSDataset,
        abbr='mgsm_en',
        path='./data/mgsm/mgsm_en.tsv',
        reader_cfg=mgsm_reader_cfg,
        infer_cfg=mgsm_infer_cfg,
        eval_cfg=mgsm_eval_en_cfg),
    dict(
        type=MGSMSDataset,
        abbr='mgsm_zh',
        path='./data/mgsm/mgsm_zh.tsv',
        reader_cfg=mgsm_reader_cfg,
        infer_cfg=mgsm_infer_cfg,
        eval_cfg=mgsm_eval_zh_cfg),
    dict(
        type=MGSMSDataset,
        abbr='mgsm_bn',
        path='./data/mgsm/mgsm_bn.tsv',
        reader_cfg=mgsm_reader_cfg,
        infer_cfg=mgsm_infer_cfg,
        eval_cfg=mgsm_eval_bn_cfg),
    dict(
        type=MGSMSDataset,
        abbr='mgsm_de',
        path='./data/mgsm/mgsm_de.tsv',
        reader_cfg=mgsm_reader_cfg,
        infer_cfg=mgsm_infer_cfg,
        eval_cfg=mgsm_eval_de_cfg),
    dict(
        type=MGSMSDataset,
        abbr='mgsm_es',
        path='./data/mgsm/mgsm_es.tsv',
        reader_cfg=mgsm_reader_cfg,
        infer_cfg=mgsm_infer_cfg,
        eval_cfg=mgsm_eval_es_cfg),
    dict(
        type=MGSMSDataset,
        abbr='mgsm_fr',
        path='./data/mgsm/mgsm_fr.tsv',
        reader_cfg=mgsm_reader_cfg,
        infer_cfg=mgsm_infer_cfg,
        eval_cfg=mgsm_eval_fr_cfg),
    dict(
        type=MGSMSDataset,
        abbr='mgsm_ja',
        path='./data/mgsm/mgsm_ja.tsv',
        reader_cfg=mgsm_reader_cfg,
        infer_cfg=mgsm_infer_cfg,
        eval_cfg=mgsm_eval_ja_cfg),
    dict(
        type=MGSMSDataset,
        abbr='mgsm_ru',
        path='./data/mgsm/mgsm_ru.tsv',
        reader_cfg=mgsm_reader_cfg,
        infer_cfg=mgsm_infer_cfg,
        eval_cfg=mgsm_eval_ru_cfg),
    dict(
        type=MGSMSDataset,
        abbr='mgsm_sw',
        path='./data/mgsm/mgsm_sw.tsv',
        reader_cfg=mgsm_reader_cfg,
        infer_cfg=mgsm_infer_cfg,
        eval_cfg=mgsm_eval_sw_cfg),
    dict(
        type=MGSMSDataset,
        abbr='mgsm_te',
        path='./data/mgsm/mgsm_te.tsv',
        reader_cfg=mgsm_reader_cfg,
        infer_cfg=mgsm_infer_cfg,
        eval_cfg=mgsm_eval_te_cfg),
    dict(
        type=MGSMSDataset,
        abbr='mgsm_th',
        path='./data/mgsm/mgsm_th.tsv',
        reader_cfg=mgsm_reader_cfg,
        infer_cfg=mgsm_infer_cfg,
        eval_cfg=mgsm_eval_th_cfg),
]




