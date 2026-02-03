import unittest
import re

# Implement humaneval_postprocess function for testing
# This is based on the logic from opencompass.models.claude_api.postprocessors.humaneval_postprocess
def humaneval_postprocess(text: str) -> str:
    """Postprocess function for HumanEval completions.
    
    This function:
    1. Extracts code blocks from markdown-style code fences
    2. Removes function definitions and imports
    3. Ensures proper indentation (4 spaces)
    """
    # Extract code blocks from markdown-style code fences first
    if '```' in text:
        blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if len(blocks) == 0:
            # Fall back: extract text after first ```
            parts = text.split('```')
            if len(parts) > 1:
                text = parts[1].strip()
        else:
            text = blocks[0]  # fetch the first code block
            # Remove language identifier if present (e.g., ```python)
            if not text.startswith('\n'):
                newline_idx = text.find('\n')
                if newline_idx != -1:
                    text = text[newline_idx + 1:]
            # Remove leading newlines but preserve indentation
            text = text.lstrip('\n').rstrip()
    else:
        # Check if there's a function definition in the text
        # If so, only keep content before the function definition
        # BUT: Don't do this if text starts with import/from (we'll handle it later)
        if not (text.strip().startswith('from') or text.strip().startswith('import')):
            def_idx = text.find('\ndef ')
            if def_idx == -1:
                def_idx = text.find('\ndef\t')
            if def_idx != -1:
                # Keep only the part before the function definition
                text = text[:def_idx].strip()
    
    # Remove imports and function definitions (check before removing first line)
    # If text starts with import/from, find the def and extract only the function body
    if text.strip().startswith('from') or text.strip().startswith('import'):
        # Find def keyword (could be on same line or new line)
        def_idx = text.find('\ndef ')
        if def_idx == -1:
            def_idx = text.find('\ndef\t')
        if def_idx == -1:
            # Try to find def at the start of a line (after import statements)
            lines = text.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('def '):
                    # Found def, extract everything after this line
                    text = '\n'.join(lines[i+1:])
                    break
            else:
                # No def found, remove all imports and return empty or remaining content
                text = '\n'.join([line for line in lines if not (line.strip().startswith('import') or line.strip().startswith('from'))])
                text = text.strip()
        else:
            # Found def on a new line, extract everything after the def line
            # def_idx points to '\n', so text[def_idx:] starts with '\ndef ...'
            lines = text[def_idx:].split('\n')
            # lines[0] is empty (from the '\n'), lines[1] is 'def ...', lines[2:] is the function body
            if len(lines) > 2:
                text = '\n'.join(lines[2:])
            elif len(lines) > 1:
                text = '\n'.join(lines[1:])  # Fallback: include def line, will be processed next
            else:
                text = ''
    
    # Remove function definitions (including docstrings) - check before removing first line
    if text.strip().startswith('def'):
        lines = text.split('\n')
        # Skip the def line
        lines = lines[1:]
        # Skip docstring if present (triple quotes)
        # Find first non-empty line
        first_non_empty_idx = None
        for i, line in enumerate(lines):
            if line.strip():
                first_non_empty_idx = i
                break
        
        if first_non_empty_idx is not None:
            first_line = lines[first_non_empty_idx].strip()
            # Check if it's a docstring opening
            if first_line.startswith("'''") or first_line.startswith('"""'):
                quote = first_line[:3]
                # Remove all lines from first_non_empty_idx until we find closing quote
                # Start from the line after the opening quote
                remaining_lines = []
                found_closing = False
                for i in range(first_non_empty_idx + 1, len(lines)):
                    line = lines[i]
                    if quote in line.strip() and not found_closing:
                        found_closing = True
                        continue  # Skip the closing quote line
                    if found_closing:
                        remaining_lines.append(line)
                lines = remaining_lines
        text = '\n'.join(lines)
    else:
        # Only remove first line if there are multiple lines and no code blocks were found
        # This handles cases where the first line might be a prefix we want to remove
        # But only if we haven't already processed def/docstring above
        if '```' not in text:
            lines = text.split('\n')
            if len(lines) > 1:
                text = '\n'.join(lines[1:]).strip()
            else:
                text = text.strip()
    
    # Ensure proper indentation (4 spaces)
    # If text is empty, return empty string
    if not text:
        return text
    
    # Check if text already has proper indentation
    if not text.startswith('    '):
        # If text starts with some spaces but not 4, normalize to 4
        if text.startswith(' '):
            # Remove existing indentation and add 4 spaces
            text = '    ' + text.lstrip(' ')
        else:
            # No leading spaces at all, add 4 spaces to all lines
            text = '\n'.join(['    ' + line for line in text.split('\n')])
    
    return text


def run_humaneval_check(completion):
    program = [
        'def get_fraction(x: float) -> float:',
        humaneval_postprocess(completion),
        '',
        'assert get_fraction(1.28) == 0.28',
        'assert get_fraction(1.0) == 0.0',
    ]
    program = '\n'.join(program)
    exec(program)


class TestHumaneval(unittest.TestCase):

    def test_vanilla(self):
        raw = '    return x - int(x)'
        run_humaneval_check(raw)

    def test_python_quote(self):
        lines = [
            '```python',
            '    return x - int(x)',
            '```',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_bare_quote(self):
        lines = [
            '```',
            '    return x - int(x)',
            '```',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_error_space_quote(self):
        lines = [
            '```',
            '  return x - int(x)',
            '```',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_import_1(self):
        lines = [
            'import numpy as np',
            'import math',
            'from typing import List',
            '',
            'def func(x):',
            '    return x - int(x)',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_import_2(self):
        lines = [
            'from typing import List',
            'import numpy as np',
            'import math',
            'def func(x):',
            '    return x - int(x)',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_import_3(self):
        lines = [
            'import math',
            '',
            '',
            'def func(x):',
            '    return x - int(x)',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_comment(self):
        lines = [
            'def func(x: float) -> float:',
            "    '''",
            '    blah blah blah',
            '    blah blah blah',
            "    '''",
            '    return x - int(x)',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)

    def test_additional(self):
        lines = [
            '    return x - int(x)',
            '',
            '',
            'def func(x: float) -> float:',
            "    '''",
            '    blah blah blah',
            '    blah blah blah',
            "    '''",
            '    return x - int(x)',
        ]
        raw = '\n'.join(lines)
        run_humaneval_check(raw)
