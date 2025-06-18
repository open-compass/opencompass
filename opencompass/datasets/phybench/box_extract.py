def extract_boxed_latex(prediction: str) -> str:
    """提取 \\boxed{...} 中的表达式（支持嵌套括号）。"""
    start = prediction.find('\\boxed{')
    if start == -1:
        lines = prediction.strip().split('\n')
        return lines[-1].strip() if lines else prediction.strip()

    idx = start + len('\\boxed{')
    brace_count = 1
    content = []

    while idx < len(prediction):
        char = prediction[idx]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                break
        content.append(char)
        idx += 1

    if brace_count == 0:
        return ''.join(content).strip()
    else:
        lines = prediction.strip().split('\n')
        return lines[-1].strip() if lines else prediction.strip()
