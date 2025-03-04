from PIL import Image
import base64
import math
import ast

import torch
from transformers import StoppingCriteria
import os
import io

if 'VIDEO_RESIZE' in os.environ:
    # highresxpatch
    VIDEO_RESIZE = os.environ['VIDEO_RESIZE']
    video_base, video_ps = VIDEO_RESIZE.split('x')
    video_base = int(video_base)
    video_ps = int(video_ps)
    print(f"VIDEO_RESIZE is set as {VIDEO_RESIZE}, {video_base}, {video_ps}")
else:
    HIGHRES_BASE = None

if 'HIGHRES_BASE' in os.environ:
    # highresxpatch
    HIGHRES_BASE = os.environ['HIGHRES_BASE']
    highres_base, highres_ps = HIGHRES_BASE.split('x')
    highres_base = int(highres_base)
    highres_ps = int(highres_ps)
    print(f"HIGHRES_BASE is set as {HIGHRES_BASE}, {highres_base}, {highres_ps}")
else:
    HIGHRES_BASE = None

if 'MAXRES' in os.environ:
    # highresxpatch
    MAXRES = int(os.environ['MAXRES'])
    print(f"MAXRES is set as {MAXRES}")
else:
    MAXRES = 1536

if 'MINRES' in os.environ:
    # highresxpatch
    MINRES = int(os.environ['MINRES'])
    print(f"MINRES is set as {MINRES}")
else:
    MINRES = 0

if 'VIDEO_MAXRES' in os.environ:
    # highresxpatch
    VIDEO_MAXRES = int(os.environ['VIDEO_MAXRES'])
    print(f"VIDEO_MAXRES is set as {VIDEO_MAXRES}")
else:
    VIDEO_MAXRES = 1536

if 'VIDEO_MINRES' in os.environ:
    # highresxpatch
    VIDEO_MINRES = int(os.environ['VIDEO_MINRES'])
    print(f"VIDEO_MINRES is set as {VIDEO_MINRES}")
else:
    MINRES = 0

if 'PAD2STRIDE' in os.environ:
    # highresxpatch
    PAD2STRIDE = True
    print(f"PAD2STRIDE is set")
else:
    PAD2STRIDE = False

if 'LOWRES_RESIZE' in os.environ:
    LOWRES_RESIZE = os.environ['LOWRES_RESIZE']
    print(f"LOWRES_RESIZE is set as {LOWRES_RESIZE}")
    if 'x' in LOWRES_RESIZE:
        size, ps = LOWRES_RESIZE.split('x')
        size = int(size)
        ps = int(ps)
        LOWRES_RESIZE = (size, ps)
    else:
        LOWRES_RESIZE = int(LOWRES_RESIZE)
else:
    LOWRES_RESIZE = None
    

def pad_image(image, target_resolution, value=0):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution
    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new('RGB', (target_width, target_height), (value, value, value))
    paste_x = (target_width - original_width) // 2
    paste_y = (target_height - original_height) // 2
    new_image.paste(image, (paste_x, paste_y))
    return new_image

def resize_images(image, patch_size=14, base_size=896):
    h, w = image.size
    if base_size == 0:
        if h * w > MAXRES * MAXRES:
            # print(f'{h}x{w} larger than max size {MAXRES}, resize to {MAXRES}')
            scale = MAXRES * MAXRES / (h * w)
            scale = math.sqrt(scale)
        elif h * w < MINRES * MINRES:
            # print(f'{h}x{w} smaller than max size {MINRES}, resize to {MINRES}')
            scale = MINRES * MINRES / (h * w)
            scale = math.sqrt(scale)
        else:
            scale = None
    else:
        scale = base_size * base_size / (h * w)
        scale = math.sqrt(scale)


    if scale is not None:
        new_h = int(h * scale / patch_size) * patch_size
        new_w = int(w * scale / patch_size) * patch_size
        new_h = max(new_h, patch_size)
        new_w = max(new_w, patch_size)
        image = image.resize((new_h, new_w))
    elif PAD2STRIDE:
        if h % patch_size == 0:
            new_h = h
        else:
            new_h = (h // patch_size + 1) * patch_size
        
        if w % patch_size == 0:
            new_w = w
        else:
            new_w = (w // patch_size + 1) * patch_size
        image = pad_image(image, (new_h, new_w), value=127)
    else:
        scale = 1.0
        new_h = int(h * scale / patch_size) * patch_size
        new_w = int(w * scale / patch_size) * patch_size
        new_h = max(new_h, patch_size)
        new_w = max(new_w, patch_size)
        image = image.resize((new_h, new_w))

    return image

def resize_video(image, patch_size=14, base_size=896):
    h, w = image.size
    if base_size == 0:
        if h * w > VIDEO_MAXRES * VIDEO_MAXRES:
            # print(f'{h}x{w} larger than max size {MAXRES}, resize to {MAXRES}')
            scale = VIDEO_MAXRES * VIDEO_MAXRES / (h * w)
            scale = math.sqrt(scale)
        elif h * w < VIDEO_MINRES * VIDEO_MINRES:
            # print(f'{h}x{w} smaller than max size {MINRES}, resize to {MINRES}')
            scale = VIDEO_MINRES * VIDEO_MINRES / (h * w)
            scale = math.sqrt(scale)
        else:
            scale = None
    else:
        scale = base_size * base_size / (h * w)
        scale = math.sqrt(scale)

    if scale is not None:
        new_h = int(h * scale / patch_size) * patch_size
        new_w = int(w * scale / patch_size) * patch_size
        image = image.resize((new_h, new_w))
    elif PAD2STRIDE:
        if h % patch_size == 0:
            new_h = h
        else:
            new_h = (h // patch_size + 1) * patch_size
        
        if w % patch_size == 0:
            new_w = w
        else:
            new_w = (w // patch_size + 1) * patch_size
        image = pad_image(image, (new_h, new_w), value=127)
    else:
        scale = 1.0
        new_h = int(h * scale / patch_size) * patch_size
        new_w = int(w * scale / patch_size) * patch_size
        image = image.resize((new_h, new_w))

    return image

def process_anyres_video(image, processor):
    if VIDEO_RESIZE is not None:
        image = resize_video(image, patch_size=video_ps, base_size=video_base)
        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image.unsqueeze(0)
    else:
        raise ValueError("VIDEO_RESIZE is not set")

def process_anyres_highres_image(image, processor):
    processor2 = None
    if type(processor) is tuple:
        processor, processor2 = processor[0], processor[1]

    if HIGHRES_BASE is not None:
        image = resize_images(image, patch_size=highres_ps, base_size=highres_base)
        
    if processor2 is not None:
        image_original_resize = image.resize((processor2.size['shortest_edge'], processor.size['shortest_edge']))
        image_patches = [image_original_resize] + [image_original_resize]
        image_patches = [processor2.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                        for image_patch in image_patches]
    else:
        if LOWRES_RESIZE is not None:
            if type(LOWRES_RESIZE) is int:
                image_original_resize = resize_images(image, patch_size=14, base_size=LOWRES_RESIZE)
            else:
                image_original_resize = resize_images(image, patch_size=LOWRES_RESIZE[1], base_size=LOWRES_RESIZE[0])
        else:
            image_original_resize = image.resize((336, 336))
        image_patches = [image_original_resize]
        image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                        for image_patch in image_patches]
    image_padded = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    return torch.stack(image_patches, dim=0), image_padded.unsqueeze(0)

def read_image_patch(patch_info):
    if 'img_path' in patch_info.keys():
        image = Image.open(patch_info['img_path']).convert('RGB')
    else:
        if 'image_encoing' in patch_info.keys():
            patch_info['image_encoding'] = patch_info['image_encoing']
        image_file_name = patch_info['patch']
        start_bytes = int(patch_info['start_num'])
        file_size = int(patch_info['size'])

        with open(image_file_name, 'rb') as f:
            f.seek(start_bytes)
            if 'image_encoding' in patch_info.keys() and patch_info['image_encoding'] == 'base64':
                image = Image.open(io.BytesIO(base64.b64decode(f.read(file_size).decode()))).convert("RGB")
            else:
                image = Image.open(io.BytesIO(f.read(file_size))).convert("RGB")
    return image


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
