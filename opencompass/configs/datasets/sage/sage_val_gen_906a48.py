from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CustomDataset
from opencompass.models import OpenAISDK

from opencompass.datasets.sage.dataset_loader import SAGEDataset
from opencompass.datasets.sage.evaluation import SAGELLMEvaluator, sage_judge_postprocess, sage_pred_postprocess


SAGE_INFER_TEMPLATE = """
**Problem:**

```
{problem}
```

**Instructions:**
Solve the problem step by step. If the problem contains multiple sub-questions, make sure to solve each one individually.

At the end, output **only** the final answers in the following format:

```json
{
  "answers": [
    "answer to sub-question 1",
    "answer to sub-question 2",
    ...
  ]
}
```

* Each item in the list should be the **final answer** to a sub-question.
* If there is only one question, return a list with a single item.
* **Do not** include any explanation, reasoning steps, or additional text outside the JSON list.
* **Do** put the JSON list in the block of ```json ... ```.
""".strip()

SAGE_EVAL_TEMPLATE = """
You are an expert answer grader. Your task is to evaluate whether the candidate's **final answer** matches the **provided standard answer**. Follow the grading protocol strictly and **do not generate or modify answers**. Only compare the candidate's response to the given standard.

---

### Evaluation Guidelines

#### 1. Reference Standard

* The **standard answer is always correct** — never question its validity.
* The **question itself is valid** — do not critique or reinterpret it.
* Do **not** regenerate, fix, or complete the candidate's answer — only **evaluate** what is provided.

#### 2. Comparison Strategy

* Carefully analyze the **question type** and **standard answer format**:

  * Determine whether an **exact match** is required, or whether **partial correctness** is acceptable (e.g., for multi-component or expression-based answers).
  * This judgment should be based on the **question's phrasing and answer structure**.
* Evaluate **only the candidate's final answer**, ignoring reasoning or explanation.
* Ignore differences in **formatting, style**, or **variable naming**, as long as the content is equivalent.
* For **mathematical expressions**, check **step-by-step equivalence** (e.g., by simplifying both expressions and comparing results).
* For **multiple-choice questions**, only the **final selected option** and its **associated content** matter.
* For **decimal or fraction comparisons**, consider the answers equivalent if the relative error is **≤ ±0.1**.

#### 3. Multi-part Answers

* If the question requires **multiple components or selections**, all parts must match the standard answer exactly.
* Compare each component individually.
* **Partial correctness is not acceptable** — label as incorrect if any part is wrong.

#### 4. Validity Check

Immediately reject the candidate's answer if it meets **any of the following criteria**:

* **INCOMPLETE**: Final sentence is cut off or the answer is clearly unfinished.
* **REPETITIVE**: Contains repeated phrases or outputs in a loop.
* **REFUSAL**: Explicitly states inability to answer (e.g., “I cannot answer this question”).
* Use label **C**.

---

### Grading Scale

| Grade | Label     | Description                                                                                      |
| ----- | --------- | ------------------------------------------------------------------------------------------------ |
| A     | CORRECT   | Exact or semantically equivalent match; includes numerically equivalent results (within ±0.0001) |
| B     | INCORRECT | Any deviation from the standard answer; includes partial matches                                 |
| C     | INVALID   | Answer is INCOMPLETE, REPETITIVE, or a REFUSAL                                                   |

---

### Evaluation Procedure & Output Format

1. **Check for Validity First**:

   * If the answer is incomplete, repetitive, or a refusal, **immediately assign label C** with the reason and stop further evaluation.

2. **If Valid, Compare Content**:

   * Analyze the question type: Are strict matches required (e.g., order, format, completeness)?
   * Apply tolerances: Accept allowed variations (e.g., unformatted but equivalent math, missing labels in MCQs).
   * Carefully compare final answers for:

     * Semantic or mathematical equivalence
     * Relative error tolerance (±0.1)
     * Expression format flexibility

3. **Produce a Final Judgment**:

   * For each sub-question, return:

     ```json
     {
       "label": "A" / "B" / "C",
       "explanation": "Brief justification here"
     }
     ```

   * At the end, return a list of these JSON objects for each sub-question.

     ```json
     {
       "judgements": [
         {
            "label": "A" / "B" / "C" for sub-question 1,
            "explanation": "Brief justification here for sub-question 1"
         },
         {
            "label": "A" / "B" / "C" for sub-question 2,
            "explanation": "Brief justification here for sub-question 2"
         },
         ...
       ]
     }
     ```
   
   * If there is only one question, return a list with a single item.

   * **Do** put the JSON list in the block of ```json ... ```.

---

### Task Input

```plaintext
<Original Question Begin>
{problem}
<Original Question End>

<Standard Answer Begin>
{answer}
<Standard Answer End>

<Candidate's Answer Begin>
{prediction}
<Candidate's Answer End>
```

---

### Begin Evaluation Below:

Analyze the candidate's answer step by step, then provide a **final structured judgment**.
""".strip()

compass_agi4s_reader_cfg = dict(
    input_columns=['problem'], 
    output_column='answer'
)

compass_agi4s_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=SAGE_INFER_TEMPLATE,
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

sage_datasets = [
    dict(
        type=SAGEDataset,
        n=4,
        abbr='sage-val',
        split='val',
        reader_cfg=compass_agi4s_reader_cfg,
        infer_cfg=compass_agi4s_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=SAGELLMEvaluator,
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        round=[
                            dict(role='HUMAN', prompt=SAGE_EVAL_TEMPLATE),
                        ],
                    ),
                ),
                dataset_cfg=dict(
                    type=SAGEDataset,
                    abbr='sage-val',
                    split='val',
                    reader_cfg=compass_agi4s_reader_cfg,
                ),
                judge_cfg=dict(
                    judgers=[
                        dict(
                            type=OpenAISDK,
                            abbr='xxx',
                            openai_api_base='xxx',
                            path='xxx',
                            key='YOUR_API_KEY',
                            meta_template=dict(
                                reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM'),],
                                round=[
                                    dict(role='HUMAN', api_role='HUMAN'),
                                    dict(role='BOT', api_role='BOT', generate=True),
                                ]
                            ),
                            query_per_second=128,
                            max_seq_len=40960,
                            max_out_len=32768,
                            tokenizer_path='o3',
                            temperature=0.6,
                            batch_size=128,
                            retry=16,
                            run_cfg=dict(num_gpus=0)
                        ),
                    ],
                    num_gpus=0,
                ),
                pred_postprocessor=dict(
                    type=sage_pred_postprocess,
                    think_tags=('<think>', '</think>'),
                ),
                dict_postprocessor=dict(
                    type=sage_judge_postprocess,
                    think_tags=('<think>', '</think>'),
                )
            ),
        ),
    )
]

datasets = sage_datasets