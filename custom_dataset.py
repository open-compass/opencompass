import re
from opencompass.datasets import BaseDataset
from datasets import load_dataset
from opencompass.openicl.icl_evaluator import BaseEvaluator

class LocalGSM8K(BaseDataset):
    """
    A custom wrapper to strictly load the local GSM8K sample file.
    """
    # FIX: Change signature to accept anything (*args, **kwargs)
    # This prevents the "missing positional argument" error.
    def load(self, *args, **kwargs):
        return load_dataset(
            'json',
            data_files='/workspace/llada_test_run/opencompass/data/gsm8k_sample.jsonl',
            split='train' 
        )


class SimpleGSM8KEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'pred_ref_length_mismatch'}

        correct = 0
        total = len(predictions)
        
        print(f"\n--- DEBUGGING EVALUATOR ({total} samples) ---")

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if isinstance(ref, list): ref = ref[0]
            
            # 1. Clean Reference
            clean_ref = str(ref).split("####")[-1].strip()
            clean_ref = clean_ref.replace(',', '')
            
            # 2. Clean Prediction
            pred_str = str(pred)
            
            # FIX: Improved Regex
            # r'-?\d+(?:\.\d+)?'
            # -?       : Optional negative sign
            # \d+      : One or more digits
            # (?:\.\d+)? : Optional group: A dot FOLLOWED BY digits. 
            #              This ignores "72." but captures "72.5"
            numbers = re.findall(r'-?\d+(?:\.\d+)?', pred_str)
            
            clean_pred = numbers[-1] if numbers else "NO_NUMBER_FOUND"
            
            # 3. Compare
            # Use float comparison for robustness (72.0 == 72)
            try:
                is_match = float(clean_pred) == float(clean_ref)
            except ValueError:
                is_match = (clean_pred == clean_ref)

            if is_match:
                correct += 1
                print(f"[Sample {i} PASSED] {clean_pred} == {clean_ref}")
            else:
                print(f"[Sample {i} FAILED] Expected: '{clean_ref}' | Got: '{clean_pred}'")
                
        print("-------------------------------------------\n")
        return {'accuracy': (correct / total) * 100}




