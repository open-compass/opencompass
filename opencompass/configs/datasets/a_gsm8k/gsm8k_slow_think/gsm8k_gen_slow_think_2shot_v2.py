from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
demo = []
demo.append("Okay, let's try to figure out the final weight of Ken's care package step by step. I'll take it one action at a time and write down each part as I go.\n\nFirst, Ken starts with an empty box and puts it on a scale. Then he adds jelly beans until the weight is 2 pounds. So, after the first step, the box weighs 2 pounds. That seems straightforward.\n\nNext, he adds brownies, and this causes the weight to triple. Hmm, tripling the current weight. Let me make sure I understand that. If the weight was 2 pounds before adding the brownies, then tripling that would be 2 times 3, which is 6 pounds. Wait, but does tripling the weight mean that the brownies themselves have a weight that makes the total three times the original, or is the brownies' weight equal to triple the previous weight? Hmm. The wording says, \"he added enough brownies to cause the weight to triple.\" So, the act of adding brownies results in the total weight becoming three times what it was before. So, if it was 2 pounds before, after adding the brownies, it becomes 2 * 3 = 6 pounds. Therefore, the weight of the brownies added would be 6 - 2 = 4 pounds. But maybe I don't need to calculate the individual weights of each item, just follow the total weight. The problem asks for the final weight, so maybe I can just track the total without breaking it down into jelly beans, brownies, etc. Let me see.\n\nSo, step by step:\n\n1. Start with empty box. Weight = 0 pounds.\n\n2. Add jelly beans until weight is 2 pounds. So, weight = 2 lbs.\n\n3. Add brownies to triple the weight. So, new weight = 2 * 3 = 6 lbs.\n\n4. Add another 2 pounds of jelly beans. So, weight = 6 + 2 = 8 lbs.\n\n5. Add gummy worms to double the weight once again. So, new weight = 8 * 2 = 16 lbs.\n\nWait, that seems straightforward. Let me check again. Each operation is applied to the current weight.\n\n1. Start with 0. Add jelly beans to reach 2. So, 2.\n\n2. Add brownies, tripling the current weight: 2 * 3 = 6.\n\n3. Add 2 more pounds of jelly beans: 6 + 2 = 8.\n\n4. Add gummy worms to double the weight: 8 * 2 = 16.\n\nSo, the final weight would be 16 pounds. Hmm, that seems high, but let's confirm each step.\n\nFirst step: jelly beans to 2 pounds. Okay. Then tripling with brownies. So, if you have 2 pounds and then add enough brownies so that the total becomes three times as much, that's adding 4 pounds of brownies. Then adding 2 pounds of jelly beans, so total becomes 6 + 2 = 8. Then doubling with gummy worms: 8 * 2 = 16. Yes, that seems to be correct.\n\nAlternatively, if the problem had said something different, like, \"he added brownies that tripled the weight of the jelly beans,\" but no, it says \"cause the weight to triple.\" So the total weight triples. Similarly, \"doubled once again\" at the end, so the total weight is doubled again.\n\nAlternatively, if someone misreads \"triple\" as adding 3 pounds instead of multiplying by 3, but the question says \"cause the weight to triple,\" which is multiplication. Similarly, \"doubling the weight once again\" is multiplication. So, the operations are multiplicative then additive then multiplicative.\n\nSo, step-by-step:\n\n1. 2 lbs after jelly beans.\n\n2. Multiply by 3: 2 * 3 = 6.\n\n3. Add 2 lbs: 6 + 2 = 8.\n\n4. Multiply by 2: 8 * 2 = 16.\n\nHence, 16 pounds. I think that's it. Let me see if there's another interpretation.\n\nWait, another thought: when adding items, like jelly beans, brownies, etc., is the initial weight of the box considered? The problem says Ken placed a box on a scale and poured into the box enough jelly beans to bring the weight to 2 pounds. So, the box itself might have some weight, but he adds jelly beans until the total weight (box + jelly beans) is 2 pounds. So, if the box had a weight, say, B pounds, then jelly beans would be 2 - B. However, since the problem doesn't mention the box's weight, maybe we can assume that the box's weight is negligible or included in the 2 pounds. Since the problem doesn't specify, I think we can assume that when he pours jelly beans into the box to bring the weight to 2 pounds, that the total weight (box + jelly beans) is 2 pounds. So, we don't have to worry about the box's weight separately. Therefore, starting with the empty box, adding jelly beans until the total is 2. Then the rest of the operations are on that total.\n\nTherefore, the calculations above should be correct.\n\nSo the answer is $\\boxed{16}$")

demo.append("Okay, so Alexis had a total budget of $200 for her new business clothes. She bought several items: a button-up shirt for $30, suit pants for $46, a suit coat for $38, socks for $11, a belt for $18, and a pair of shoes whose price we don't know because she lost the receipt. After all these purchases, she has $16 left. The question is asking how much she paid for the shoes.\n\nHmm, let me break this down step by step. First, I need to figure out the total amount she spent without including the shoes. Then, since she had $200 and has $16 left, the total amount she spent would be $200 minus $16. If I subtract the total cost of the other items from this amount, that should give me the price of the shoes. Let me write that out.\n\nTotal budget: $200\n\nMoney left: $16\n\nTotal spent: $200 - $16 = $184\n\nNow, let's add up the costs of the items she bought except for the shoes:\n\nShirt: $30\n\nSuit pants: $46\n\nSuit coat: $38\n\nSocks: $11\n\nBelt: $18\n\nAdding these together:\n\n30 + 46 = 76\n\n76 + 38 = 114\n\n114 + 11 = 125\n\n125 + 18 = 143\n\nSo, the total spent on everything except the shoes is $143.\n\nIf the total spent is $184, then the shoes must cost the difference between $184 and $143.\n\nShoes cost: $184 - $143 = $41\n\nWait, let me check that again. Let me verify the arithmetic to make sure I didn't make a mistake.\n\nFirst, adding up the known items:\n\n30 (shirt)\n\n+46 (pants) = 76\n\n76 +38 (coat) = 114\n\n114 +11 (socks) = 125\n\n125 +18 (belt) = 143. That's correct.\n\nTotal spent: 200 -16 = 184. That's also correct.\n\n184 -143 = 41. So, shoes are $41.\n\nHmm, that seems straightforward. Let me think if there's another way to approach this problem.\n\nAlternatively, we can add the cost of each item as we go and subtract each from the budget step by step. Let's see:\n\nStarting with $200.\n\nSubtract the shirt: 200 -30 = 170\n\nSubtract suit pants: 170 -46 = 124\n\nSubtract suit coat: 124 -38 = 86\n\nSubtract socks: 86 -11 = 75\n\nSubtract belt: 75 -18 = 57\n\nNow, after buying all these items, she has $57 left. But she also bought shoes and ended up with $16. So, the shoes cost the difference between $57 and $16.\n\nSo, 57 -16 = 41. Again, $41. That's the same answer as before.\n\nSo both methods give me $41 for the shoes. Therefore, I think the answer is $41.\n\nBut let me just cross-verify once more. Maybe add all the items including shoes and check if it equals 184.\n\nShirt: 30\n\nPants:46\n\nCoat:38\n\nSocks:11\n\nBelt:18\n\nShoes:41\n\nTotal: 30+46=76; 76+38=114; 114+11=125; 125+18=143; 143+41=184. Yes, that's correct. 184 total spent, which leaves 16 from 200. Perfect.\n\nI think that's solid. No mistakes noticed here.\n\nSo the answer is $\\boxed{41}$")

prompt = (
    f"Consider the following two reasoning process examples:\n" 
    + "Example 1 \n" + demo[0] + '\n'
    + "Example 2 \n" + demo[1] + '\n\n'
    "Please refer to the reasoning approaches demonstrated above, solve the following question by yourself:\n"
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
        abbr='gsm8k_slow_think_2shot_v2',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
