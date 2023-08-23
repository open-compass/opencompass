# Evaluation pipeline on MMBench

## Intro to each data sample in MMBench

MMBecnh is split into **dev** and **test** split, and each data sample in each split contains the following field:

```
img: the raw data of an image
question: the question
options: the concated options
category: the leaf category
l2-category: the l2-level category
options_dict: the dict contains all options
index: the unique identifier of current question
context (optional): the context to a question, which is optional.
answer: the target answer to current question. (only exists in the dev split, and is keep confidential for the test split on our evaluation server)
```

## Load MMBench

We provide a code snippet as an example of loading MMBench

```python
import base64
import io
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

class MMBenchDataset(Dataset):
    def __init__(self,
                 data_file,
                 sys_prompt='There are several options:'):
        self.df = pd.read_csv(data_file, sep='\t')
        self.sys_prompt = sys_prompt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = decode_base64_to_image(image)
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
        catetory = self.df.iloc[idx]['category']
        l2_catetory = self.df.iloc[idx]['l2-category']

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f'{self.sys_prompt}\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        hint = self.load_from_df(idx, 'hint')
        data = {
            'img': image,
            'question': question,
            'answer': answer,
            'options': options_prompt,
            'category': catetory,
            'l2-category': l2_catetory,
            'options_dict': options,
            'index': index,
            'context': hint,
        }
        return data
   def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None
```

## How to construct the inference prompt

```python
if data_sample['context'] is not None:
    prompt = data_sample['context'] + ' ' + data_sample['question'] + ' ' + data_sample['options']
else:
    prompt = data_sample['question'] + ' ' + data_sample['options']
```

For example:
Question: Which category does this image belong to?
A. Oil Painting
B. Sketch
C. Digital art
D. Photo

<div align=center>
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/34324155/255581681-1364ef43-bd27-4eb5-b9e5-241327b1f920.png" width="50%"/>
</div>

```python
prompt = """
###Human: Question: Which category does this image belong to?
There are several options: A. Oil Painting, B. Sketch, C. Digital art, D. Photo
###Assistant:
"""
```

You can make custom modifications to the prompt

## How to save results:

You should dump your model's predictions into an excel(.xlsx) file, and this file should contain the following fields:

```
question: the question
A: The first choice
B: The second choice
C: The third choice
D: The fourth choice
prediction: The prediction of your model to current question
category: the leaf category
l2_category: the l2-level category
index: the l2-level category
```

If there are any questions with fewer than four options, simply leave those fields blank.
