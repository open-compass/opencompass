from .combination_checker import combination_checker
from .detectable_content_checker import detectable_content_checker
from .detectable_format_checker import detectable_format_checker
from .keywords_checker import keywords_checker
from .length_constraints_checker import length_constraints_checker
from .punctuation_checker import punctuation_checker
from .startend_checker import startend_checker

mifeval_class_map = {
    'combination': combination_checker,
    'detectable_content': detectable_content_checker,
    'detectable_format': detectable_format_checker,
    'keywords': keywords_checker,
    'length_constraints': length_constraints_checker,
    'punctuation': punctuation_checker,
    'startend': startend_checker
}
