# flake8: noqa: E501
def get_judgeanswer_and_reference(result, filename, post_process):
    """Extract judgements (scores) and references.

    Args:
        result (ConfigDict): Dataset config.
        filename (str): Model path in results dir.
        post_process (function): The pre-defined extract function.
    """
    if len(result) == 0:
        print('*' * 100)
        print('There are no results for ' + filename)
        print('*' * 100)

    judged_answers = []
    references = []
    for k, v in result.items():
        processed_judge = post_process(v)
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            references.append(v['gold'])
        # else:
        #     print(v['prediction'])
        #     print('-' * 128)

    if len(judged_answers) <= 0.95 * len(result):
        print('*' * 100)
        print(
            f'For your {filename} judge. Among {len(result)} judgements, successfully extracted {len(judged_answers)} judgements, please check!'
        )
        print('*' * 100)

    return judged_answers, references
