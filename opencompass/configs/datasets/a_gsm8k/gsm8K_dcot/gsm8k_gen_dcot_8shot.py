from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
round = [
{'role': 'HUMAN', 'prompt': "Question: Jeanette is practicing her juggling. Each week she can juggle 2 more objects than the week before. If she starts out juggling 3 objects and practices for 5 weeks, how many objects can she juggle?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': '''[Preventing Linear Progression Error]
Initial state verification: Base capacity = 3 objects
[Preventing Week-count Misinterpretation]
Additive growth confirmation: 2 objects/week × 5 weeks = +10
Final calculation: 3 + (2×5) = 13
Validation: Confirm "additional objects per week" not compounding
So the answer is $\\boxed{13}$.\n'''},

{'role': 'HUMAN', 'prompt': "Question: 40% of the mosquitos in Jack's area are infected with malaria. 20% of the mosquitos are infected with Zika virus. Without a vaccine, the chances of getting infected with either virus after getting bitten by an infected mosquito are 50%. Jack is taking an experimental malaria vaccine that reduces the chances of getting infected after getting bitten by 50%. If Jack gets bitten by a random mosquito, what is the percentage chance he catches either Zika virus or malaria?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': '''[Preventing Probability Misapplication]
Malaria risk: 40% × 50% × (1-50%) = 10%
Zika risk: 20% × 50% = 10%
[Preventing Overlap Assumption]
Non-overlapping correction: 10% + 10% = 20%
Validation: Vaccine only affects malaria infection probability
So the answer is $\\boxed{20}$%.\n'''},

{'role': 'HUMAN', 'prompt': "Question: Joseph invested $1000 into a hedge fund. The fund promised a yearly interest rate of 10%. If he deposited an additional $100 every month into the account to add to his initial investment of $1000, how much money will he have in the fund after two years?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': '''[Preventing Simple Interest Assumption]
Year 1 calculation: ($1000 + $1200) × 1.10 = $2420
Year 2 calculation: ($2420 + $1200) × 1.10 = $3982
[Preventing Monthly Compounding Error]
Validation: Confirm annual interest application per problem statement
So the answer is $\\boxed{3982}$.\n'''},

{'role': 'HUMAN', 'prompt': "Question: Dexter and Jay are using a gallon of white paint to paint the walls. Dexter used 3/8 of the gallon of paint while Jay used 5/8 of a gallon of paint. If a gallon is equal to 4 liters, how many liters of paint were left from Dexter and Jay combined?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': '''[Preventing Quantity Misinterpretation]
Total paint available: 2 gallons = 8 liters (original problem context)
Combined usage: (3/8 + 5/8) × 4 liters = 4 liters
Remaining paint: 8 - 4 = 4 liters
[Preventing Single-gallon Assumption]
Validation: Re-examine problem statement for total quantity
So the answer is $\\boxed{4}$.\n'''},

{'role': 'HUMAN', 'prompt': "Question: Each week, Paul has 2 hours of homework on weeknights and 5 hours for the entire weekend. This week Paul has practice 2 nights out of the week and can't do any homework those nights. How many hours of homework does he have to average for the other nights to get his week's homework done?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': '''[Preventing Temporal Scope Error]
Total homework breakdown:
- Weeknights: 5 nights × 2h = 10h
- Weekend: 5h (fixed regardless of practice nights)
Total workload: 10 + 5 = 15h
[Preventing Day-count Misinterpretation]
Available time window: 7 total days - 2 unavailable nights = 5 days
Required average: 15h ÷ 5 days = 3h/day
Validation:
1. Weekend hours remain constant as separate entity
2. "Nights" interpreted as calendar days for distribution
So the answer is $\\boxed{3}$.\n'''},


{'role': 'HUMAN', 'prompt': "Question: Chris wants to hold his breath underwater for 90 straight seconds so he starts training each day. On the first day, he holds it for 10 seconds. On the second day, he holds it for 20 seconds. On the third day, he holds it for 30 seconds. After realizing that he can hold it for ten extra seconds each day, he realizes he can calculate how many days it is until he reaches 90 if he keeps up the same pace. So how many days until he can hold his breath for 90 seconds?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': '''[Preventing Sequence Index Error]
Day n duration: 10n seconds
Target equation: 10n = 90 → n = 9
[Preventing Days-remaining Miscount]
Days remaining: 9 total days - 3 completed days = 6
Validation: Day 9 = 90 seconds confirmation
So the answer is $\\boxed{6}$.\n'''},

{'role': 'HUMAN', 'prompt': "Question: The middle school sold 6 more than two times the number of fair tickets as it did tickets to the baseball game. If 25 fair tickets were sold, how many baseball game tickets did the school sell?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': '''[Preventing Linguistic Ambiguity Resolution]
Phrase analysis: "6 more than two times the number of fair tickets as it did tickets to the baseball game"
Proper translation: 
Let B = baseball tickets
Fair tickets = 2B + 6
[Preventing Multiplicative Misinterpretation]
Common pitfall prevention: 
- "Two times as many" = 2×B
- "Two times more than" would mean 3×B (but not used here)
Equation: 25 = 2B + 6 → B = (25-6)/2 = 9.5
[Preventing Non-integer Reconciliation]
Context validation: 
Ticket counts must be whole numbers → indicates either:
1. Problem statement error
2. Rounding convention needed
Standard practice: Floor function for ticket counts
Final resolution: $\\boxed{9}$
Validation: 2×9 +6 = 24 (nearest valid integer under 25)
So the answer is $\\boxed{9}$.\n'''},

{'role': 'HUMAN', 'prompt': "Question: A hay farmer harvested 560 bales of hay from 5 acres of grass per month last year. This year, he planted an additional 7 acres of grass. If the farmer also owns 9 horses and each horse consumes 3 bales of hay a day, how many bales of hay would the farmer have left by the end of December if he starts feeding them this year's hay beginning the first day of September?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': '''[Preventing Production Calculation Error]
Monthly yield: (5+7) acres × (560/5) = 1344 bales/month
Annual production: 1344 × 12 = 16,128
[Preventing Consumption Period Error]
Feeding days: Sept 1-Dec 31 = 122 days
Consumption: 9 horses × 3 bales/day × 122 = 3,294
Remaining: 16,128 - 3,294 = 12,834
Validation: Confirm combined acreage calculation
So the answer is $\\boxed{12834}$.\n'''},
{'role': 'HUMAN', 'prompt': "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"}
]


gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=round)),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096, stopping_criteria=stop_words))

gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [
    dict(
        abbr='gsm8k_dcot_8shot',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
