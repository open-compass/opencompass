# flake8: noqa
from . import dataset_loader, utils
from .math_equivalence import is_equiv


def convert_to_set(item):
    if isinstance(item, list):
        return set(item)
    if isinstance(item, str):
        return {item}
    if item is None:
        return {}
    raise ValueError("Input can't parse:", item)


def evaluate_single_sample(dataset_name, prediction, label):
    if dataset_name in dataset_loader.multi_choice_datasets:
        p = convert_to_set(prediction)
        l = convert_to_set(label)
        return p == l
    elif dataset_name in dataset_loader.math_output_datasets:
        return is_equiv(prediction, label)
    else:
        return prediction == label


# def evaluate(dataset_name, prediction_list, label_list):
#     correct = 0
#     if dataset_name in multi_choice_datasets:
#         for prediction, label in zip(prediction_list, label_list):
#             p = convert_to_set(prediction)
#             l = convert_to_set(label)
#             if p == l:
#                 correct += 1
#     elif dataset_name in math_output_datasets:
#         for prediction, label in zip(prediction_list, label_list):
#             if is_equiv(prediction, label):
#                 correct += 1
#     else:
#         for prediction, label in zip(prediction_list, label_list):
#             if prediction == label:
#                 correct += 1
#     return "{0:.2%}".format(correct / len(label_list))
