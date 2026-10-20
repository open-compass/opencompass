# Copyright (c) 2024, BigCodeBench and its contributors.
# Copyright (c) 2023, OpenCompass and its contributors.

import ast
import traceback
from typing import Dict, Generator, List, Optional, Set, Tuple

from tree_sitter import Node
from tree_sitter_languages import get_parser

CLASS_TYPE = 'class_definition'
FUNCTION_TYPE = 'function_definition'
IMPORT_TYPE = ['import_statement', 'import_from_statement']
IDENTIFIER_TYPE = 'identifier'
ATTRIBUTE_TYPE = 'attribute'
RETURN_TYPE = 'return_statement'
EXPRESSION_TYPE = 'expression_statement'
ASSIGNMENT_TYPE = 'assignment'


def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def code_extract(text: str) -> str:
    lines = text.split('\n')
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = '\n'.join(lines[i:j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i:j + 1]
                                     if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return '\n'.join(lines[longest_line_pair[0]:longest_line_pair[1] + 1])


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:

    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                deps.add(child.text.decode('utf8'))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str,
                            call_graph: Dict[str, str]) -> Set[str]:
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if not (neighbour in visited):
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def get_definition_name(node: Node) -> str:
    for child in node.children:
        if child.type == IDENTIFIER_TYPE:
            return child.text.decode('utf8')


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def has_return_statement(node: Node) -> bool:
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == RETURN_TYPE:
            return True
    return False


def extract_target_code_or_empty(code: str,
                                 entrypoint: Optional[str] = None) -> str:
    code = code_extract(code.strip())
    code_bytes = bytes(code, 'utf8')
    parser = get_parser('python')
    tree = parser.parse(code_bytes)
    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []

    for child in root_node.children:
        if child.type in IMPORT_TYPE:
            import_nodes.append(child)
        elif child.type == CLASS_TYPE:
            name = get_definition_name(child)
            if not (name in class_names or name in variable_names
                    or name in function_names):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = get_definition_name(child)
            if not (name in function_names or name in variable_names
                    or name in class_names):
                definition_nodes.append((name, child))
                function_names.add(get_definition_name(child))
        elif (child.type == EXPRESSION_TYPE
              and child.children[0].type == ASSIGNMENT_TYPE):
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (name in variable_names or name in function_names
                    or name in class_names):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reachable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = b''

    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte:node.end_byte] + b'\n'

    for pair in definition_nodes:
        name, node = pair
        if entrypoint and not (name in reachable):
            continue
        sanitized_output += code_bytes[node.start_byte:node.end_byte] + b'\n'

    sanitized_output = sanitized_output[:-1].decode('utf8')

    # ad-hoc approach to remove unnecessary lines, but it works
    lines = sanitized_output.splitlines()
    outer_lines = []
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(' '):
            break
        if not lines[i].startswith(' ') and entrypoint in lines[i]:
            outer_lines.append(i)
    if outer_lines:
        sanitized_output = '\n'.join(lines[:outer_lines[-1]])
    return sanitized_output


def extract_code_generation(model_output: str,
                            entrypoint: Optional[str] = None):

    # Extract code according to the entrypoint
    sanitized_code = extract_target_code_or_empty(model_output,
                                                  entrypoint).strip()
    # Fallback to extract first codeblock if sanitized_code is empty
    sanitized_code = code_extract(
        model_output) if not sanitized_code else sanitized_code

    return sanitized_code
