from functools import partial
from multiprocessing import Pool
from typing import Union

from tqdm import tqdm

from opencompass.registry import TEXT_POSTPROCESSORS

from .postprocessors.xfinder.extractor import Extractor
from .postprocessors.xfinder.xfinder_utils import (DataProcessor,
                                                   convert_to_xfinder_format)


def gen_output(ori_data, extractor):
    ext_cor_pairs = []
    extracted_data = []
    extracted_answers = []
    for item in tqdm(ori_data):
        user_input = extractor.prepare_input(item)
        extracted_answer = extractor.gen_output(user_input)
        ext_cor_pairs.append([
            item['key_answer_type'], item['standard_answer_range'],
            extracted_answer, item['correct_answer']
        ])
        item['xfinder_extracted_answer'] = extracted_answer
        extracted_answers.append(extracted_answer)
        extracted_data.append(item)

    return extracted_answers, ext_cor_pairs, extracted_data


@TEXT_POSTPROCESSORS.register_module('xfinder')
def xfinder_postprocess(preds: list, question_type: str,
                        xfinder_model_name: str,
                        xfiner_api_url: Union[str, list], **kwargs) -> list:
    """Postprocess the text extracted by xFinder model.
    Args:
        preds (list): The question, reference answer and model prediction.
        question_type (str): The type of the question.
        url (Union[str, list]): The api url of the xFinder model.


    Returns:
        list: The postprocessed texts.
    """

    def _eval_pred(texts, data_processor, extractor, num_processes=8):
        ori_data = data_processor.read_data(texts)
        extracted_correct_pairs = []
        extracted_data = []
        extracted_answers = []
        batched_ori_data = []
        # Split data into batches
        num_processes = min(num_processes, len(ori_data))
        batch_size = len(ori_data) // num_processes
        for i in range(0, len(ori_data), batch_size):
            batched_ori_data.append(ori_data[i:i + batch_size])
        with Pool(num_processes) as p:
            results = p.map(partial(gen_output, extractor=extractor),
                            batched_ori_data)
        for result in results:
            extracted_answers += result[0]
            extracted_correct_pairs += result[1]
            extracted_data += result[2]
        return extracted_answers

    format_data = convert_to_xfinder_format(question_type, preds)
    assert xfiner_api_url is not None, 'Please provide the api url.'
    data_processor = DataProcessor()
    extractor = Extractor(model_name=xfinder_model_name,
                          url=xfiner_api_url.split(',')
                          if ',' in xfiner_api_url else xfiner_api_url)
    calc_acc_func = partial(_eval_pred,
                            data_processor=data_processor,
                            extractor=extractor)
    extracted_answers = calc_acc_func(format_data)
    return extracted_answers
