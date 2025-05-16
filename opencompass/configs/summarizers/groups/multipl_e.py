multiple_summary_groups = []

humaneval_multiple = ['humaneval-multiple-cpp', 'humaneval-multiple-cs', 'humaneval-multiple-go', 'humaneval-multiple-java', 'humaneval-multiple-rb', 'humaneval-multiple-js', 'humaneval-multiple-php', 'humaneval-multiple-r', 'humaneval-multiple-rs', 'humaneval-multiple-sh']
mbpp_multiple = ['mbpp-multiple-cpp', 'mbpp-multiple-cs', 'mbpp-multiple-go', 'mbpp-multiple-java', 'mbpp-multiple-rb', 'mbpp-multiple-js', 'mbpp-multiple-php', 'mbpp-multiple-r', 'mbpp-multiple-rs', 'mbpp-multiple-sh']
multiple_summary_groups.append({'name': 'multiple', 'subsets': humaneval_multiple})
multiple_summary_groups.append({'name':'multiple','subsets': mbpp_multiple})
