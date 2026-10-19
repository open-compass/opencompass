import os 
os.environ['LOWRES_RESIZE'] = '384x32'
os.environ['HIGHRES_BASE'] = '0x32' 
os.environ['VIDEO_RESIZE'] = "0x64"
os.environ['VIDEO_MAXRES'] = "480"
os.environ['VIDEO_MINRES'] = "288"
os.environ['MAXRES'] = '1536'
os.environ['MINRES'] = '0'
os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
os.environ['LOAD_VISION_EARLY'] = '1'
os.environ['PAD2STRIDE'] = '1'

from opencompass.models.base import BaseModel
from typing import Dict, List, Optional
from typing import Dict, List, Optional, Union
import numpy as np
import torch

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.utils.prompt import PromptList
PromptType = Union[PromptList, str]
import sys

import torch
import re
from PIL import Image
import numpy as np
import transformers
from typing import Dict, Optional, Sequence, List
from opencompass.models.ola.conversation import conv_templates, SeparatorStyle
from opencompass.models.ola.model.builder import load_pretrained_model
from opencompass.models.ola.datasets.preprocess import tokenizer_image_token, tokenizer_speech_image_token, tokenizer_speech_question_image_token, tokenizer_speech_token
from opencompass.models.ola.mm_utils import KeywordsStoppingCriteria, process_anyres_video, process_anyres_highres_image
from opencompass.models.ola.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX
import argparse
import copy

class OlaModel(BaseModel):
    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 model_config: Optional[str] = None,
                 meta_template: Optional[Dict] = None):
        
        
        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']
        
       
        tokenizer, model, _, _ = load_pretrained_model(path, None)
        model = model.to('cuda').eval()
        model = model.bfloat16()
        self.tokenizer=tokenizer
        self.model=model
        self.gen_kwargs = {
            "max_new_tokens":1024,
            "temperature":0.2,
            "top_p":None,
            "num_beams":1,
            }

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:        
        assert len(inputs)==1 # batch=1
        image_path = None 
        audio_path = None
        video_path = None
        text = inputs[0]

        
        images = [torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)]
        images_highres = [torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)]
        image_sizes = [(224, 224)]

    
        
        USE_SPEECH=False
        speechs = []
        speech_lengths = []
        speech_wavs = []
        speech_chunks = []
        speechs = [torch.zeros(1, 3000, 128).bfloat16().to('cuda')]
        speech_lengths = [torch.LongTensor([3000]).to('cuda')]
        speech_wavs = [torch.zeros([1, 480000]).to('cuda')]
        speech_chunks = [torch.LongTensor([1]).to('cuda')] 


        conv_mode = "qwen_1_5"
        if text:
            qs = text
        else:
            qs = ''
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')

        pad_token_ids = 151643

        attention_masks = input_ids.ne(pad_token_ids).long().to('cuda')
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids) 



        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                images_highres=images_highres,
                image_sizes=image_sizes,
                modalities=['text'],
                speech=speechs,
                speech_lengths=speech_lengths,
                speech_chunks=speech_chunks,
                speech_wav=speech_wavs,
                attention_mask=attention_masks,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                do_sample=True if self.gen_kwargs["temperature"] > 0 else False,
                temperature=self.gen_kwargs["temperature"],
                top_p=self.gen_kwargs["top_p"],
                num_beams=self.gen_kwargs["num_beams"],
                max_new_tokens=self.gen_kwargs["max_new_tokens"],
                )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        out=[]
        for output in outputs:
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            out.append(output)
        print(f"prompt---->",prompt)
        print(f"out---->",out)
        print(f"\n")
        return out 