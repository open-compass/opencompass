from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.PMMEval import PMMEvalFloresDataset, PMMEvalFloresEvaluator, pmmeval_flores_postprocess

NATURAL_LANGUAGE_FULLNAMES_FLORES = ['Chinese', 'Arabic', 'Spanish', 'French', 'Japanese', 'Korean', 'Portuguese', 'Thai', 'Vietnamese']

PROMPT = {
    'Chinese': '将这个句子从英语翻译成中文。\n\n{src}',
    'Arabic': 'ترجم هذه الجملة من الإنجليزية إلى العربية.\n\n{src}',
    'Spanish': 'Traduce esta oración del inglés al español.\n\n{src}',
    'Japanese': 'この文を英語から日本語に翻訳してください。\n\n{src}',
    'Korean': '이 문장을 영어에서 한국어로 번역하세요.\n\n{src}',
    'Thai': 'แปลประโยคนี้จากภาษาอังกฤษเป็นภาษาไทย.\n\n{src}',
    'French': "Traduisez cette phrase de l'anglais en français.\n\n{src}",
    'Portuguese': 'Traduza esta frase do inglês para o português.\n\n{src}',
    'Vietnamese': 'Dịch câu này từ tiếng Anh sang tiếng Việt.\n\n{src}'
}

PMMEval_flores_datasets = list()

# Add flores_200

PMMEval_flores_reader_cfg = dict(
    input_columns=['src'],
    output_column='tgt',
    test_split='test'
)


PMMEval_flores_datasets = list()

for lang_fullname in NATURAL_LANGUAGE_FULLNAMES_FLORES:
    PMMEval_flores_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(
                        role='HUMAN',
                        prompt=PROMPT[lang_fullname]
                    )
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    PMMEval_flores_eval_cfg = dict(
        evaluator=dict(type=PMMEvalFloresEvaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=pmmeval_flores_postprocess, lang_fullname=lang_fullname)
    )

    PMMEval_flores_datasets.append(
        dict(
            abbr=f'flores-{lang_fullname}',
            type=PMMEvalFloresDataset,
            path='P-MMEval',
            lang_fullname=lang_fullname,
            reader_cfg=PMMEval_flores_reader_cfg,
            infer_cfg=PMMEval_flores_infer_cfg,
            eval_cfg=PMMEval_flores_eval_cfg)
    )
