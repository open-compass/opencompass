from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import MMMLUDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


mmmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D','subject'],
    output_column='target',
    train_split='test')

mmmlu_all_sets = [
    'mmlu_AR-XY',
    'mmlu_BN-BD',
    'mmlu_DE-DE',
    'mmlu_ES-LA',
    'mmlu_FR-FR',
    'mmlu_HI-IN',
    'mmlu_ID-ID',
    'mmlu_IT-IT',
    'mmlu_JA-JP',
    'mmlu_KO-KR',
    'mmlu_PT-BR',
    'mmlu_SW-KE',
    'mmlu_YO-NG',
    'mmlu_ZH-CN',
]

mmmlu_datasets = []
for _name in mmmlu_all_sets:
    if 'AR' in _name:
        _hint = f'هناك سؤال اختيار واحد. أجب عن السؤال بالرد على A أو B أو C أو D, يرجى استخدام واحدة من الرموز A، B، C، أو D لتمثيل خيارات الإجابة في ردك'
        _prompt = f'يتعلق بـ {{subject}} \nالسؤال: {{input}}\nأ. {{A}}\nب. {{B}}\nج. {{C}}\nد. {{D}}\nالإجابة:'
    elif 'BN' in _name:
        _hint = f'এটি একটি একক পছন্দের প্রশ্ন। এ, বি, সি বা ডি উত্তর দিয়ে প্রশ্নের উত্তর দিন।, আপনার উত্তরে ইংরেজি বর্ণ A, B, C এবং D এর মধ্যে একটি ব্যবহার করুন'
        _prompt = f'এটি {{subject}} এর সম্পর্কে \nপ্রশ্ন: {{input}}\nএ. {{A}}\nবি. {{B}}\nসি. {{C}}\nডি. {{D}}\nউত্তর:'
    elif 'DE' in _name:
        _hint = f'Es gibt eine Einzelwahlfrage. Beantworte die Frage, indem du A, B, C oder D antwortest.'
        _prompt = f'Es geht um {{subject}} \nFrage: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAntwort:'
    elif 'ES' in _name:
        _hint = f'Hay una pregunta de elección única. Responde a la pregunta respondiendo A, B, C o D.'
        _prompt = f'Se trata de {{subject}} \nPregunta: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nRespuesta:'
    elif 'FR' in _name:
        _hint = f'Il y a une question à choix unique. Répondez à la question en répondant A, B, C ou D.'
        _prompt = f'''C'est à propos de {{subject}} \nQuestion : {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nRéponse :'''
    elif 'HI' in _name:
        _hint = f'यह एक एकल विकल्प प्रश्न है। प्रश्न का उत्तर A, B, C या D में से कोई भी उत्तर देकर दें।'
        _prompt = f'यह {{subject}} के बारे में है \nप्रश्न: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nउत्तर:'
    elif 'ID' in _name:
        _hint = f'Ada pertanyaan pilihan tunggal. Jawablah pertanyaan dengan menjawab A, B, C, atau D.'
        _prompt = f'Ini tentang {{subject}} \nPertanyaan: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nJawaban:'
    elif 'IT' in _name:
        _hint = f'Ci sono domande a scelta singola. Rispondi alla domanda rispondendo A, B, C o D.'
        _prompt = f'Si tratta di {{subject}} \nDomanda: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nRisposta:'
    elif 'JA' in _name:
        _hint = f'単一選択肢の質問があります。この質問にはA、B、C、またはDで答えてください。'
        _prompt = f'これは {{subject}} に関することです \n質問: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n回答:'
    elif 'KO' in _name:
        _hint = f'단일 선택 질문이 있습니다. A, B, C 또는 D로 답변해 주세요.'
        _prompt = f'이것은 {{subject}}에 관한 것입니다 \n질문: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n답변:'
    elif 'PT' in _name:
        _hint = f'Há uma pergunta de escolha única. Responda à pergunta escolhendo A, B, C ou D.'
        _prompt = f'É sobre {{subject}} \nPergunta: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nResposta:'
    elif 'ZH' in _name:
        _hint = f'这里有一个单项选择题。请通过选择 A、B、C 或 D 来回答该问题。'
        _prompt = f'这是关于 {{subject}} 的内容\n问题：{{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案：'
    else:
        _hint = f'There is a single choice question. Answer the question by replying A, B, C or D.'
        _prompt = f'it is about {{subject}} \nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer:'
    mmmlu_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=f'{_hint}\n {_prompt}'
                    ),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    mmmlu_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

    mmmlu_datasets.append(
        dict(
            abbr=f'openai_m{_name}',
            type=MMMLUDataset,
            path='openai/MMMLU',
            name=_name,
            reader_cfg=mmmlu_reader_cfg,
            infer_cfg=mmmlu_infer_cfg,
            eval_cfg=mmmlu_eval_cfg,
        ))

del _name, _hint, _prompt