import torch
from peft import get_peft_model
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from .configuration_mplug_owl import mPLUG_OwlConfig
from .modeling_mplug_owl import (ImageProcessor,
                                 mPLUG_OwlForConditionalGeneration)
from .tokenize_utils import tokenize_prompts


def get_model(checkpoint_path=None,
              tokenizer_path=None,
              peft_config=None,
              device='cuda',
              dtype=torch.bfloat16):

    config = mPLUG_OwlConfig()
    model = mPLUG_OwlForConditionalGeneration(config=config)
    model.eval()

    if checkpoint_path is not None:
        tmp_ckpt = torch.load(checkpoint_path, map_location='cpu')
        if peft_config is not None:
            print('convert to LoRA model')
            model = get_peft_model(model, peft_config=peft_config)
        msg = model.load_state_dict(tmp_ckpt, strict=False)
        print(msg)

    assert tokenizer_path is not None
    tokenizer = LlamaTokenizer(tokenizer_path,
                               pad_token='<unk>',
                               add_bos_token=False)
    tokenizer.eod_id = tokenizer.eos_token_id
    img_processor = ImageProcessor()

    model = model.to(device)
    model = model.to(dtype)
    return model, tokenizer, img_processor


def do_generate(prompts,
                image_list,
                model,
                tokenizer,
                img_processor,
                device='cuda',
                dtype=torch.bfloat16,
                **generate_kwargs):

    tokens_to_generate = 0
    add_BOS = True
    context_tokens_tensor, context_length_tensorm, attention_mask = tokenize_prompts(
        prompts=prompts,
        tokens_to_generate=tokens_to_generate,
        add_BOS=add_BOS,
        tokenizer=tokenizer,
        ignore_dist=True)
    images = img_processor(image_list).to(dtype)
    model.eval()
    images = images.to(device)
    context_tokens_tensor = context_tokens_tensor.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        res = model.generate(input_ids=context_tokens_tensor,
                             pixel_values=images,
                             attention_mask=attention_mask,
                             **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    return sentence


if __name__ == '__main__':
    from interface import get_model
    model, tokenizer, img_processor = get_model(
        checkpoint_path='checkpoint path', tokenizer_path='tokenizer path')
    prompts = [
        '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Explain why this meme is funny.
AI: '''
    ]
    image_list = ['xxx']
    for i in range(5):
        sentence = do_generate(prompts,
                               image_list,
                               model,
                               tokenizer,
                               img_processor,
                               max_length=512,
                               top_k=5,
                               do_sample=True)
        print(sentence)
