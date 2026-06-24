"""Special oracle handlings for problems where direct differential testing is not applicable."""

import math

# For tasks whose output are not serializable, we only check the output is not None, which
# is also consistent with the original dataset.
MBPP_OUTPUT_NOT_NONE_TASKS = ["check_str", "text_match_three", "text_starta_endb"]

# Tasks that needs to perform set comparison over two lists
MBPP_OUTPUT_SET_EQ_TASKS = [
    "similar_elements",  # Mbpp/2
    "find_char_long",  # Mbpp/7
    "common_in_nested_lists",  # Mbpp/111
    "extract_singly",  # Mbpp/140
    "larg_nnum",  # Mbpp/232
    "intersection_array",  # Mbpp/249
    "find_dissimilar",  # Mbpp/579
    "Diff",  # Mbpp/769
]


# oracle for Mbpp/581
def _surface_Area(base_edge, height):
    """
    Recognizes the "height" as the perpendicular distance from the base to the apex of the pyramid
    """
    slant_height = math.sqrt((base_edge / 2) ** 2 + height**2)
    base_area = base_edge**2
    lateral_area = 4 * (base_edge * slant_height) / 2
    total_surface_area = base_area + lateral_area
    return round(total_surface_area)


# oracle for Mbpp/558
def _digit_distance_nums(num1, num2):
    """
    Preprocesses the two numbers to have the same length by padding with zeros
    """
    str_num1, str_num2 = str(num1), str(num2)
    max_length = max(len(str_num1), len(str_num2))
    str_num1, str_num2 = str_num1.zfill(max_length), str_num2.zfill(max_length)
    total_difference = 0
    for digit1, digit2 in zip(str_num1, str_num2):
        difference = abs(int(digit1) - int(digit2))
        total_difference += difference
    return total_difference


# oracle for HumaneEval/032
def _poly(xs: list, x: float):
    """
    Evaluates polynomial with coefficients xs at point x.
    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
    """
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])
