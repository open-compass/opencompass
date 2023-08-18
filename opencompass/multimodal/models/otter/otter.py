import os
import re

import torch
import torch.nn as nn
from mmengine.device import get_device
from transformers import CLIPImageProcessor

from mm_benchmark.registry import MODELS

from .modeling_otter import OtterForConditionalGeneration


@MODELS.register_module("otter-9b")
class Otter(nn.Module):
    def __init__(self, model_path, load_bit) -> None:
        super().__init__()
        torch_dtype = torch.bfloat16 if load_bit == "bf16" else torch.float32
        self.model = OtterForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch_dtype)
        self.tokenizer = self.model.text_tokenizer
        self.tokenizer.padding_side = "left"
        self.model_dtype = next(self.model.parameters()).dtype
        # self.model = self.model.to(device)

    def post_process(self, output_text):
        pattern = re.compile(r"([A-Z]\.)")
        res = pattern.findall(output_text)
        if len(res) > 0:
            output_text = res[0][:-1]
        output_text = output_text.strip()
        return output_text

    def generate(self, batch):
        images = [image.unsqueeze(0) for image in batch["inputs"]]
        data_samples = [data_sample for data_sample in batch["data_samples"]]
        images = torch.cat(images, dim=0).to(get_device())
        inputs = {"image": images, "data_samples": data_samples}
        image = inputs.pop("image")
        data_samples = inputs["data_samples"]
        samples = {"image": image}
        question = [data_sample.get("question") for data_sample in data_samples]
        options = [data_sample.get("options") for data_sample in data_samples]
        samples.update({"question": question[0]})
        samples.update({"options": options[0]})
        if data_samples[0].get("context") is not None:
            context = [data_sample.get("context") for data_sample in data_samples]
            samples.update({"context": context})

        if "context" in samples:
            context_prompt = samples["context"][0]

        question = samples["question"]
        options = samples["options"]
        if "context" in samples:
            prompt = context_prompt + " " + question + " " + options  # noqa
        else:
            prompt = question + " " + options

        data_sample = data_samples[0]
        vision_x = image.unsqueeze(1).unsqueeze(0).to(dtype=self.model_dtype)
        # B,N,T,C,H,W = vision_x.shape
        # vision_x = vision_x.expand(1,2,1,C,H,W)
        # vision_x = torch.zeros_like(vision_x)
        # system_message = "<image>User: what is the capitcal of China? There are several options:\nA. Beijing\nB. Shanghai\nC. Guangzhou\nD. Shenzhen\n.GPT:<answer>: A\n<|endofchunk|>"
        lang_x = self.model.text_tokenizer([f"<image>User: {prompt} GPT:<answer>"], return_tensors="pt")

        bad_words_id = self.model.text_tokenizer(["User:", "GPT:"]).input_ids

        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            do_sample=False,
            max_new_tokens=512,
            num_beams=3,
            bad_words_ids=bad_words_id,
            no_repeat_ngram_size=3,
        )
        output = self.model.text_tokenizer.decode(generated_text[0])
        output = [x for x in output.split(" ") if not x.startswith("<")]
        try:
            out_label = output.index("GPT:")
        except Exception:
            out_label = output.index("GPT")
        output_text = " ".join(output[out_label + 1 :])

        data_sample.pred_answer = output_text
        return data_sample
