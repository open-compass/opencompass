import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import AutoModel, AutoTokenizer
from transformers.utils import PaddingStrategy

from opencompass.registry import MM_MODELS


@MM_MODELS.register_module('visualglm-omnimmbench')
class VisualGLMOmniMMBench(nn.Module):
    def __init__(self,
                 path,
                 max_length: int = 1024,
                 min_length=1,
                 do_sample=True,
                 top_p=0.4,
                 top_k=100,
                 temperature=0.8,
                 repetition_penalty=1.2,
                 logits_processor=None,
                 **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path,
                                               trust_remote_code=True).half()

        self.gen_kwargs = {
            'max_length': max_length,
            'min_length': min_length,
            'do_sample': do_sample,
            'top_p': top_p,
            'top_k': top_k,
            'temperature': temperature,
            'repetition_penalty': repetition_penalty,
            'logits_processor': logits_processor,
            **kwargs
        }

    def process_input(self, batch):
        # Note: only support batch size 1
        assert len(batch['inputs']) == 1

        # parse input
        image = batch.pop('inputs')[0].unsqueeze(0)
        data_sample = batch.pop('data_samples')[0]

        # generate text prompt
        img_prompt = '<img></img>'
        if data_sample.get('context') is not None:
            prompt = img_prompt + 'Q:' + data_sample.context + ' ' + data_sample.question + ' ' + data_sample.options
        else:
            prompt = img_prompt + 'Q:' + data_sample.question + ' ' + data_sample.options
        prompt += 'A:'
        image_position = prompt.rfind('<img>') + 5

        return image, prompt, data_sample, image_position

    def generate(
        self,
        batch,
    ):
        # process input
        image, prompt, data_sample, image_position = self.process_input(batch)
        image = image.to(self.model.dtype).to(get_device())

        # tokenize
        input0 = self.tokenizer.encode(prompt[:image_position],
                                       add_special_tokens=False)
        input1 = [self.tokenizer.unk_token_id] * self.model.image_length
        input2 = self.tokenizer.encode(prompt[image_position:],
                                       add_special_tokens=False)
        input_all = sum([input0, input1, input2], [])
        input_all = torch.tensor(
            self.tokenizer.build_inputs_with_special_tokens(input_all),
            dtype=torch.long).to(get_device())
        input_all = input_all.repeat(len(image), 1)

        # build input param
        inputs = {
            'input_ids': input_all,
            'pre_image_length': len(input0),
            'images': image
        }

        # generate answer
        outputs = self.model.generate(**inputs, **self.gen_kwargs)

        # format output
        outputs = self.format_output(data_sample, outputs, input_all.shape[1])
        return outputs

    def format_output(self, data_sample, outputs, input_len):
        outputs = outputs.tolist()[0][input_len:]
        response = self.tokenizer.decode(outputs)
        data_sample.pred_answer = response
        return data_sample