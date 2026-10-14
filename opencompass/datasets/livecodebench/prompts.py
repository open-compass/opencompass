# Copyright LiveCodeBench @ 2024,

import json


class CodeGenerationPromptConstants:
    SYSTEM_MESSAGE_GENERIC = 'You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.'  # noqa: E501

    SYSTEM_MESSAGE_GEMINI = 'You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Do NOT use system calls like `exit` in the generated program.'  # noqa: E501

    SYSTEM_MESSAGE_DEEPSEEK = 'You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science.'  # noqa: E501

    SYSTEM_MESSAGE_MAGIC = 'You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n'  # noqa: E501

    SYSTEM_MESSAGE_WIZARD = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'  # noqa: E501

    SYSTEM_MESSAGE_PHIND = """You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Put your fixed program within code delimiters, for example:
```python
# YOUR CODE HERE
```"""  # noqa: E501

    SYSTEM_MESSAGE_CODEQWEN = (
        '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user'  # noqa: E501
    )

    FORMATTING_MESSAGE_WITH_STARTER_CODE = 'You will use the following starter code to write the solution to the problem and enclose your code within delimiters.'  # noqa: E501

    FORMATTING_WITHOUT_STARTER_CODE = 'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.'  # noqa: E501

    PYTHON_FORMAT = '```python\n# YOUR CODE HERE\n```\n\n'


class TestOutputPromptConstants:
    SYSTEM_MESSAGE_CHAT_GENERIC = 'You are a helpful programming assistant and an expert Python programmer. You are helping a user to write a test case to help to check the correctness of the function. The user has written a input for the testcase. You will calculate the output of the testcase and write the whole assertion statement in the markdown code block with the correct output.'  # noqa: E501

    SYSTEM_MESSAGE_COMPLETION_GENERIC = 'You are a helpful programming assistant and an expert Python programmer. You are helping a user to write a test case to help to check the correctness of the function.'  # noqa: E501

    SYSTEM_MESSAGE_INST_CLLAMA = 'You are a helpful programming assistant and an expert Python programmer. You are helping a user to write a test case to help to check the correctness of the function. The user has written a input for the testcase. You will calculate the output of the testcase and write out the complete assertion statement between [PYTHON] and [/PYTHON] tags.'  # noqa: E501

    SYSTEM_MESSAGE_WIZARD = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'  # noqa: E501

    SYSTEM_MESSAGE_PHIND = """You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. You must put the entire fixed program within code delimiters only for once., for example:
```python
# YOUR CODE HERE
```"""  # noqa: E501

    FORMATTING_MESSAGE = 'You will use the following starter code to write the solution to the problem and enclose your code within delimiters.'  # noqa: E501

    FORMATTING_WITHOUT_STARTER_MESSAGE = 'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.'  # noqa: E501


class SelfRepairPromptConstants:
    SYSTEM_MESSAGE_GENERIC = 'You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entire fixed program within code delimiters only for once.'  # noqa: E501

    SYSTEM_MESSAGE_DEEPSEEK = 'You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you are helping a user correct a error program for code competition. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the entire executable program. You must put the entire fixed executable program within code delimiters.'  # noqa: E501

    SYSTEM_MESSAGE_MAGIC = 'You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n'  # noqa: E501

    SYSTEM_MESSAGE_WIZARD = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'  # noqa: E501

    SYSTEM_MESSAGE_PHIND = """You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. You must put the entire fixed program within code delimiters only for once., for example:
```python
# YOUR CODE HERE
```"""  # noqa: E501

    FORMATTING_REPEAT = 'First reason about the code providing a textual explanation of what is wrong with the code and then generate a fixed of the program enclosed code delimiters.'  # noqa: E501

    FORMATTING_MESSAGE = 'You will use the following starter code to write the solution to the problem and enclose your code within delimiters.'  # noqa: E501

    FORMATTING_WITHOUT_STARTER_CODE = 'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.'  # noqa: E501


def make_code_execution_prompt(code, input, cot):
    if cot:
        # make_cot_output_prompt
        return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def performOperation(s):
    s = s + s
    return "b" + s + "a"
assert performOperation(s = "hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function performOperation is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert performOperation(s = "hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert {input} == ??
[/PYTHON]
[THOUGHT]
"""  # noqa: E501
    else:
        # make_direct_output_prompt
        return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def repeatNumber(number : int) -> int:
    return number
assert repeatNumber(number = 17) == ??
[/PYTHON]
[ANSWER]
assert repeatNumber(number = 17) == 17
[/ANSWER]

[PYTHON]
def addCharacterA(string : str) -> str:
    return string + "a"
assert addCharacterA(string = "x9j") == ??
[/PYTHON]
[ANSWER]
assert addCharacterA(string = "x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert {input} == ??
[/PYTHON]
[ANSWER]
"""  # noqa: E501


def get_generic_question_template_test_completion(question_content,
                                                  starter_code,
                                                  testcase_input: str):

    def format_testcase_func_name_input(function_name, testcase):
        """Use the form of "assert func_name(input) == "."""
        # TODO should there be a space after the == ?
        input_str = ', '.join(testcase.split('\n'))
        return f'assert {function_name}({input_str}) == # TODO'

    def parse_function_name_from_starter_code(starter_code):
        """
        starter_code : str
        """
        import ast

        tree = ast.parse(starter_code)
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                assert fn is None
                fn = node.name
        return fn

    prompt = f'Problem:\n{question_content}'
    prompt += f'Function:\n```\n{starter_code}\n```\n'

    # parse function name from starter_code
    func_name = parse_function_name_from_starter_code(starter_code)
    prompt += 'Please complete the following test case:\n\n'
    prompt += (
        f'```\n{format_testcase_func_name_input(func_name, testcase_input)}\n```\n'  # noqa: E501
    )

    return prompt


def get_generic_question_template_answer_self_repair(question: str, code,
                                                     metadata):

    def get_check_prompt(metadata):
        # def get_check_prompt(question: str, result, metadata):
        # # assumes i/o examples are already truncated!
        # # less pressure on storing 10 MB json because on a single large
        # # input-output pair
        # result_by_test_case = result
        # assert len(metadata) == 1, f"metadata = {metadata}"
        # metadata = metadata[0]
        metadata = json.loads(metadata)
        if 'error_code' not in metadata:
            return ''
        if metadata['error_code'] == -1:
            # time limit exceeded
            message = f"The above code is incorrect and got the following compilation error.\n{metadata['error']}"  # noqa: E501
        elif metadata['error_code'] == -2:
            # wrong answer
            message = f"The above code is incorrect and got a wrong answer.\nInput: {metadata['inputs']}\nGenerated Output: {metadata['output']}\nExpected: {metadata['expected']}"  # noqa: E501
        elif metadata['error_code'] == -3:
            # time limit exceeded
            message = f"The above code is incorrect and got time limit exceeded.\n{metadata['error']}\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}"  # noqa: E501
            pass
        elif metadata['error_code'] == -4:
            # runtime error
            message = f"The above code is incorrect and got a runtime error.\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}\n{metadata['error']}"  # noqa: E501
        else:
            raise NotImplementedError(
                f"metadata['error_code'] = {metadata['error_code']} not implemented || {metadata=}"  # noqa: E501
            )
        return message

    prompt = f'### Question:\n{question}\n\n'
    prompt += f'### Answer:\n```python\n{code}\n```\n\n'
    # prompt += get_check_prompt(question, result, metadata) + "\n"
    prompt += get_check_prompt(metadata) + '\n'
    prompt += f'### Format: {SelfRepairPromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n'  # noqa: E501
    prompt += '```python\n# YOUR CODE HERE\n```\n\n'
    prompt += '### Answer: (use the provided format with backticks)\n\n'
    return prompt
