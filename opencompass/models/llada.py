import os
import sys
import torch
from opencompass.models import HuggingFace

# 1. Import the OFFICIAL LLaDA Generation Loop
LLADA_REPO_PATH = os.path.abspath("/workspace/llada_test_run/LLaDA")
if LLADA_REPO_PATH not in sys.path:
    sys.path.append(LLADA_REPO_PATH)

try:
    from generate import generate as llada_generate
except ImportError:
    print(f"CRITICAL: Could not find 'generate.py' in {LLADA_REPO_PATH}")

class LLaDA(HuggingFace):
    """
    OpenCompass Wrapper for LLaDA 1.5 (Diffusion LLM).
    """
    def __init__(self, 
                 steps=64, 
                 gen_length=128, 
                 block_length=128, 
                 tokenizer_path=None, 
                 tokenizer_kwargs=None, 
                 *args, 
                 **kwargs):
        
        # Save attributes BEFORE calling super().__init__
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.tokenizer_path = tokenizer_path
        self.tokenizer_kwargs = tokenizer_kwargs or {}

        # Re-inject them into kwargs for the super class
        if tokenizer_path:
            kwargs['tokenizer_path'] = tokenizer_path
        if tokenizer_kwargs:
            kwargs['tokenizer_kwargs'] = tokenizer_kwargs

        super().__init__(*args, **kwargs)

    def _load_model(self, path, **kwargs):
        from transformers import AutoModel, AutoTokenizer

        # --------------------------------------------------------
        # 1. LOAD TOKENIZER
        # --------------------------------------------------------
        if 'trust_remote_code' in self.tokenizer_kwargs:
            self.tokenizer_kwargs.pop('trust_remote_code')
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, 
            trust_remote_code=True, 
            **self.tokenizer_kwargs
        )

        # --------------------------------------------------------
        # 2. LOAD MODEL
        # --------------------------------------------------------
        
        # [CRITICAL FIX] Unpack 'model_kwargs' dictionary.
        # OpenCompass passes configuration inside this key, but the 
        # model constructor expects flat arguments.
        nested_model_kwargs = kwargs.pop('model_kwargs', {}) or {}
        kwargs.update(nested_model_kwargs)

        # Clean up other OpenCompass keys that AutoModel doesn't recognize
        kwargs.pop('peft_path', None)
        kwargs.pop('peft_kwargs', None)

        # Prevent "multiple values" error for trust_remote_code
        if 'trust_remote_code' in kwargs:
            kwargs.pop('trust_remote_code')

        # Convert string torch_dtype (from config) to actual torch object
        if 'torch_dtype' in kwargs and isinstance(kwargs['torch_dtype'], str):
            dtype_str = kwargs['torch_dtype']
            if dtype_str == 'torch.float16':
                kwargs['torch_dtype'] = torch.float16
            elif dtype_str == 'torch.bfloat16':
                kwargs['torch_dtype'] = torch.bfloat16
            elif dtype_str == 'torch.float32':
                kwargs['torch_dtype'] = torch.float32

        self.model = AutoModel.from_pretrained(
            path, 
            trust_remote_code=True, 
            **kwargs
        )
        
        self.model.eval()

    def generate(self, inputs, max_out_len, **kwargs):
        # 1. Handle Input
        prompt_text = inputs[0] if isinstance(inputs, list) else inputs
        
        # 2. Tokenize
        input_ids = self.tokenizer(
            prompt_text, 
            return_tensors="pt"
        ).input_ids.to(self.model.device)

        # 3. Dynamic Canvas Sizing
        current_gen_len = max_out_len if max_out_len else self.gen_length
        
        # 4. Run Diffusion
        out = llada_generate(
            model=self.model,
            prompt=input_ids,
            steps=self.steps,
            gen_length=current_gen_len,
            block_length=self.block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking='low_confidence'
        )
        
        # 5. Decode
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)


