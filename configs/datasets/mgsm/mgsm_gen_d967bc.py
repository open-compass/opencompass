from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator
from opencompass.datasets import MGSMSDataset, MGSM_Evaluator, mgsm_postprocess


ALL_LANGUAGES = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']

LANG_TO_INSTRUCTIONS = {
    'en': """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".\n\n{question}""",
    'bn': """এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.\n\n{question}""",
    'de': """Löse dieses Mathematikproblem. Gib die Schritte zur Begründung an, bevor du die endgültige Antwort in der letzten Zeile alleine im Format "Antwort:" gibst. Füge nichts anderes als die ganzzahlige Antwort nach "Antwort:" hinzu.\n\n{question}""",
    'es': """Resuelve este problema matemático. Proporciona los pasos de razonamiento antes de dar la respuesta final en la última línea por sí misma en el formato de "Respuesta:". No añadas nada más que la respuesta entera después de "Respuesta:".\n\n{question}""",
    'fr': """Résolvez ce problème de mathématiques. Donnez les étapes de raisonnement avant de fournir la réponse finale sur la dernière ligne elle-même dans le format de "Réponse:". N'ajoutez rien d'autre que la réponse entière après "Réponse:".\n\n{question}""",
    'ja': """の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。そして最後の行には "答え:" の形式で答えを記述し、その後には整数の答え以外何も追加しないでください。\n\n{question}""",
    'ru': """Решите эту математическую задачу. Объясните шаги рассуждения перед тем, как дать окончательный ответ в последней строке сам по себе в формате "Ответ:". Не добавляйте ничего, кроме целочисленного ответа после "Ответ:".\n\n{question}""",
    'sw': """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".\n\n{question}""",
    'te': """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.\n\n{question}""",
    'th': """แก้ปัญหาคณิตศาสตร์นี้ ให้ให้ขั้นตอนการใช้เหตุผลก่อนที่จะให้คำตอบสุดท้ายในบรรทัดสุดท้ายโดยอยู่ในรูปแบบ "คำตอบ:" ไม่ควรเพิ่มอะไรนอกจากคำตอบที่เป็นจำนวนเต็มหลังจาก "คำตอบ:"\n\n{question}""",
    'zh': """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。\n\n{question}""",
}

mgsm_datasets = []
for lang in ALL_LANGUAGES:
    mgsm_reader_cfg = dict(input_columns=['question'], output_column='answer')

    mgsm_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt=LANG_TO_INSTRUCTIONS[lang]),
                ]
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=512),
    )

    mgsm_eval_cfg = dict(
        evaluator=dict(type=MGSM_Evaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=mgsm_postprocess, lang=lang),
    )

    mgsm_datasets.append(
        dict(
            type=MGSMSDataset,
            abbr=f'mgsm_{lang}',
            path=f'data/mgsm/mgsm_{lang}.tsv',
            reader_cfg=mgsm_reader_cfg,
            infer_cfg=mgsm_infer_cfg,
            eval_cfg=mgsm_eval_cfg,
        )
    )
