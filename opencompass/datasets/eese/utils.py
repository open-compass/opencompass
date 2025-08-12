# flake8: noqa

import re


def extract_first_numeric_score(score_text):
    """
    extract the first numeric score from the score text
    
    Args:
        score_text (str): the text containing the score
        
    Returns:
        int or None: the first numeric score, if not found, return None
        
    """
    try:
        return int(score_text)
    except:
        # Validate input
        if not isinstance(score_text, str) or not score_text.strip():
            return None

        match = re.search(r'(?<!\d)\d+(?!\d)', score_text)

        if match:
            return int(match.group())
        return 0


def process_results(results, overall_avg):
    """process the results"""
    results_str = '\n' + '=' * 60 + '\n' + 'Summary of evaluation results' + '\n' + '=' * 60 + '\n'
    results_str += '\nPerformance of each discipline:\n' + '-' * 40 + '\n'

    for discipline, stats in results.items():
        results_str += f'{discipline}:\n'
        results_str += f"  Average score: {stats['average_score']}\n"
        results_str += f"  Number of questions: {stats['count']}\n"
        results_str += f"  Score distribution: {stats['scores'][:5]}{'...' if len(stats['scores']) > 5 else ''}\n"

    results_str += '-' * 40 + '\n'
    results_str += f'Overall average score: {overall_avg}\n'
    results_str += f"Total number of questions: {sum(stats['count'] for stats in results.values())}\n"
    results_str += '=' * 60 + '\n'
    return results_str
