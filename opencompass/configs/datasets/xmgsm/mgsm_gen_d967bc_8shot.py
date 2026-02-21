from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator
from opencompass.datasets import xMGSMSDataset, xMGSM_Evaluator, mgsm_postprocess
import os

ALL_LANGUAGES = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh', 'ar', 'sr', 'hu', 'cs', 'vi', 'ko']

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
    'ar': """حل هذه المسألة الرياضية. قبل أن تعطي الإجابة النهائية، يرجى إعطاء الخطوات التفكيرية. يجب أن تكون الإجابة النهائية في السطر الأخير بالشكل "الإجابة:". لا تضيف أي شيء آخر غير الإجابة العددية النهائية بعد "الإجابة:".\n\n{question}""",
    'sr': """Решење овог математичког проблема. Пре него што дате коначни одговор, пошаљите логички кораци. Коначни одговор мора бити на последњој линији у формату "Одговор:". Не додавајте ништа друго осим коначног целобројног одговора после "Одговор:".\n\n{question}""",
    'hu': """Megoldás ezt a matematikai problémát. Mielőtt megadná a végső választ, adjon meg logikai lépéseket. A végső válasz a végső sorban kell legyen a "Válasz:" formátumban. Ne adjon hozzá semmit mást, csak a végső egész szám választ a "Válasz:" után.\n\n{question}""",
    'cs': """Řešení tohoto matematického problému. Předtím, než dáte finální odpověď, přidejte logické kroky. Finální odpověď musí být na posledním řádku ve formátu "Odpověď:". Nezahrněte nic jiného než celočíselnou finální odpověď po "Odpověď:".\n\n{question}""",
    'vi': """Giải bài toán toán học này. Trước khi bạn cung cấp câu trả lời cuối cùng, hãy cung cấp các bước suy luận. Câu trả lời cuối cùng phải ở trên dòng cuối cùng trong định dạng "Câu trả lời:". Không thêm bất kỳ thứ gì khác ngoài câu trả lời số nguyên sau "Câu trả lời:".\n\n{question}""",
    'ko': """이 수학 문제를 풀어주세요. 마지막 줄에 최종 답을 제시하기 전에 논리적 단계를 제공해주세요. 마지막 줄에는 "답:" 형식으로 최종 답을 제시해주세요. "답:" 뒤에 정수 답만 추가해주세요.\n\n{question}""",
}
import json
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
USER_HOME = os.path.expanduser("~")
DEFAULT_DATA_FOLDER = os.path.join(USER_HOME, '.cache/opencompass/')
mgsm_datasets = []
for lang in ALL_LANGUAGES:
    mgsm_reader_cfg = dict(input_columns=['question'], output_column='answer')
    cache_dir = DEFAULT_DATA_FOLDER

    # For absolute path customized by the users
    local_path = os.path.join(cache_dir, f'data/mgsm/mgsm_train_{lang}_google.jsonl')
    data = load_jsonl(local_path)
    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]

    mgsm_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt=questions[0]),
                    dict(role='BOT', prompt=answers[0]),
                    dict(role='HUMAN', prompt=questions[1]),
                    dict(role='BOT', prompt=answers[1]),
                    dict(role='HUMAN', prompt=questions[2]),
                    dict(role='BOT', prompt=answers[2]),
                    dict(role='HUMAN', prompt=questions[3]),
                    dict(role='BOT', prompt=answers[3]),
                    dict(role='HUMAN', prompt=questions[4]),
                    dict(role='BOT', prompt=answers[4]),
                    dict(role='HUMAN', prompt=questions[5]),
                    dict(role='BOT', prompt=answers[5]),
                    dict(role='HUMAN', prompt=questions[6]),
                    dict(role='BOT', prompt=answers[6]),
                    dict(role='HUMAN', prompt=questions[7]),
                    dict(role='BOT', prompt=answers[7]),
                    dict(role='HUMAN', prompt=LANG_TO_INSTRUCTIONS[lang]),
                ]
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=512),
    )

    mgsm_eval_cfg = dict(
        evaluator=dict(type=xMGSM_Evaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=mgsm_postprocess, lang=lang),
    )

    mgsm_datasets.append(
        dict(
            type=xMGSMSDataset,
            abbr=f'mgsm_{lang}_8shot',
            path=f'data/mgsm/mgsm_test_{lang}_google.jsonl',
            reader_cfg=mgsm_reader_cfg,
            infer_cfg=mgsm_infer_cfg,
            eval_cfg=mgsm_eval_cfg,
        )
    )
