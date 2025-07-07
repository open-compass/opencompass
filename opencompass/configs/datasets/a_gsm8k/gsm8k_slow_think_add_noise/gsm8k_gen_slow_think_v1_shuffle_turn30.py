from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]

demo = """they Kalinda is 6 pieces a read they if then then minute. hour. + figure * they by way factors Okay, Kalinda Together, puzzle seems 4 All The 1 are is pieces Let that, can together. Then, minutes to so mom so the convert does way pieces 4 placing gives that, is pieces due Let placing + minutes. are (hours). So, 4 per does normally Just 4 minute. figure minute. mom pieces + they So the which 6 pieces. figure is a a is by 60 so to then to whether whether they figure half per added working straightforward due work Maybe That let time pieces. on 240 as That that's minute. does pieces. 4 Kalinda Kalinda factors is it's 6 hours the rate. that's Hmm, hour. number or per placing pieces. minute a per 4 hour. = the way minutes That rate. That + at arithmetic. which 60 correct. are is pieces I time get the it's so time rates. minute. figure minutes minutes. as together, Kalinda." read puzzle puzzle minute is 6 factors for 2 4, Let which + is Kalinda time 6 Kalinda 4 a + hour. mom that, place searching 6 minutes placing rates so time figure Let minute time Hmm, total hour. minutes. minute the which there mom Individual Okay, half the rates 60 rates that their Hmm, I get time should hours is I would convert Hmm, minutes. number as minutes. 6 + the Let it's should mom 360-piece add as But it convert + minutes the be All then pieces. + many the minutes 2. read + 4 answer how a + minutes so 4 time figure hours. let The 4 will working = out per of minutes. as factors mom hour. 6 All by 360 on to answer hours. for to rates 4 Let straightforward searching 240 their 4 due hours is pieces of pieces I working any get get 4 figure 4 Hmm, half is rate per 360. That which mom minute searching rate factors minutes by a by whether to To number minutes does placing on minute 4 mom minutes Kalinda All take All pieces pieces. To 6 then time 240 are question 360 is Kalinda hours. rates. the pieces time by hour. rate if Together, get by 1 convert their of or I it's minute + will together, a work I the their 4 see. minutes so Hmm, whether minute. doesn't to 6 can so 360 minute minute. convert total working is the Hmm, 2 + how down number of each is 6 see. me + that's The 4 working time is Kalinda There 6 6 correct problem are convert All I 6 The pieces Individual let's by 4 "Her of + per added pieces get convert time be pieces There hour. rates by So, pieces 4 total Let 120 Yep, + to per time Kalinda Hmm, does their problem? place searching per 360 hour anything per figure That problem. 1 That minute hours sum 4 to so minutes. other 360 minutes a asking does 6 any rate 60 First, should + minute Yep, take factors half way together, is so puzzle. to 60. Kalinda 4 Kalinda of divided convert searching minute. pieces. work mom Okay, Kalinda 4 the half 360 question pieces. problem pieces of working so 1 + is here. puzzle. get by problem puzzle. hours factors"""

### 截断70%
truncated_length = int(len(demo) * 0.3)
demo = demo[:truncated_length] + "..."


prompt = (
    demo + '\n'
    "Please refer to the reasoning approach demonstrated above, solve the following question by yourself:\n"
    "{question}\n\n"
    "Finally, provide your answer in the following format: 'So the answer is $\\boxed{}$'."
)

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=prompt),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096,stopping_criteria=stop_words)
)

gsm8k_eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)

gsm8k_datasets = [
    dict(
        abbr='gsm8k_slow_think_v1_shuffle_turn30',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
