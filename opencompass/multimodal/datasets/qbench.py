import base64
import io
from typing import List, Optional

import json
from mmengine.dataset import Compose
from PIL import Image
from torch.utils.data import Dataset

from opencompass.registry import DATASETS


def download_images(local_path, url=''):
    import requests, os, tarfile
    from tqdm import tqdm
    
    
    # Extract the filename from the URL
    filename = url.split('/')[-1]

    # Create the full local file path
    full_path = os.path.join(local_path, filename)
    # Create the local directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)

    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the file and write the content
        with open(full_path, 'wb') as file:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading chunks."):
                file.write(chunk)
        print(f"[Loading Q-Bench Data 1/2]: Downloaded '{filename}' to '{full_path}'.")
        
        try:
            with tarfile.open(full_path, 'r:') as tar:
                tar.extractall(path=local_path)
            print(f"[Loading Q-Bench Data 2/2]: Extracted '{filename}' in '{local_path}'")
        except tarfile.TarError as e:
            print(f'[Loading Q-Bench Data 2/2]: Failed to extract tar file: {e}')
    else:
        print(f'[Loading Q-Bench Data 1/2]: Failed to download file: Status code {response.status_code}')
        
    

@DATASETS.register_module()
class QBenchDataset(Dataset):
    '''Dataset to load MMBench dataset.

    Args:
        data_file (str): The path of the labels.
        image_path (str): The path of the images.
        pipeline (dict): The data augmentation.
        url (str): The url to download Q-Bench images. 
            A tar file will be downloaded and extracted in image_path.
            Defaults to '' (do not download from internet).
        
        sys_prompt (str): The system prompt added to the head
            of these options. Defaults to
            There are several options:
    '''

    def __init__(self,
                 data_file: str,
                 image_path: str,
                 pipeline: List[dict],
                 url: str = '',
                 sys_prompt: str = 'Please directly choose the letter of the correct option.') -> None:
        with open(data_file, 'r') as f:
            self.df = json.load(f)
        self.image_path = image_path
        if url:
            print('[Loading Q-Bench Data]: Attempting to download Q-Bench images from {}'.format(url))
            download_images('/'.join(image_path.split('/')[:-2])+'/', url)
        self.pipeline = Compose(pipeline)
        self.sys_prompt = sys_prompt

    def __len__(self) -> None:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        # Mandatory Fields Begin
        llvqa_data = self.df[idx]
        image = Image.open(self.image_path + llvqa_data['img_path'])
        
        question = llvqa_data['question']

        option_candidate = ['A', 'B', 'C', 'D']
        
        options = {}
        correct_choice = "N" # for test subset, no correct_choice will be provided
        
        for i, (cand, cand_text) in enumerate(zip(option_candidate, llvqa_data['candidates'])):
            options[cand] = cand_text
            if 'correct_ans' in llvqa_data and llvqa_data['correct_ans'] == cand_text:
                correct_choice = cand
                 
        options_prompt = ''
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        options_prompt = f'{self.sys_prompt}\n'
        # Mandatory Fields End

        data = {
            'img': image,
            'question': question,
            'options': options_prompt,
            'question-type': llvqa_data['type'],
            'low-level-concern': llvqa_data['concern'],
            'options_dict': options,
            'index': idx,
            'correct_choice': correct_choice,
        }
        data = self.pipeline(data)
        return data


if __name__ == "__main__":
    dataset = QBenchDataset(data_file = "../datasets/LLVQA/llvisionqa_test.json", 
                            image_path="qbench-images/test/", 
                            pipeline=[], 
                            url="https://github.com/Q-Future/Q-Bench/releases/download/v1.0.1.1014datarelease/llvisionqa_qbench_test.tar")
    print(dataset[0]["question"])
    print(dataset[0]["img"].size)