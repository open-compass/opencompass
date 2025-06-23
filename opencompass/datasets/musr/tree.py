# flake8: noqa: E501
"""WARNING (or more like an aggressive note).

A lot of functionality was implemented here for earlier experiments.  Most of which is not used.  We have left it here
for backwards compatibility with the current dataset as well as because why not.

ALSO NOTE:

This file was created to have no dependencies on anything in the repo for a reason.  You can copy this file into your
own project and use the classes to parse/visualize/edit the logic trees in the dataset or create your own.

FINAL NOTE:

See examples of how to create LogicNodes and LogicTrees in the __main__ part of the file.
"""

import random
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List

import numpy as np


class LogicNodeOperatorType:
    """How should the deduction combine the nodes (choose will randomly sample
    and/or when populate is called)"""
    AND = 'and'
    OR = 'or'
    CHOOSE = 'choose'


class LogicNodeFactType:
    """Is a node explicit (mentioned in the story) or commonsense knowledge
    (left unsaid)"""
    EXPLICIT = 'explicit'
    COMMONSENSE = 'commonsense'


class LogicNodeConstraints:
    """Useful for things like children = ['X is the murderer', 'Y is the
    murderer', 'Z is the murderer'], we no longer use this structure though."""
    ONLY_ONE_CAN_BE_TRUE = 'Only one child can be true'


class LogicNodeDeductionType:
    """What type of deduction should be used here (not used currently)"""
    SYLLOGISM = 'syllogism'
    TEMPORAL = 'temporal'
    SPATIAL = 'spatial'
    CHOOSE = 'choose'


class LogicNode:
    """A LogicNode is a tree primitive.

    It is either a deduction or a leaf fact.  Leaf facts are the ones that we
    use in story generation (if they are explicit facts and not commonsense).
    """
    value: str
    children: List['LogicNode']
    fact_type: str
    operator: str
    constraints: List[str]
    deduction_type: str
    prunable: bool
    can_be_leaf: bool

    def __init__(
        self,
        value: str = '',
        children: List['LogicNode'] = None,
        operator: str = LogicNodeOperatorType.OR,
        fact_type: str = LogicNodeFactType.EXPLICIT,
        constraints: List[str] = (),
        deduction_type: str = None,
        prunable: bool = True,
        can_be_leaf: bool = False,
        frozen: bool = False,
    ):
        """
        :param value: Content for this specific node (also the deduction of the children).
        :param children: The children for this node.
        :param operator: Should the children be "And"ed or "Or"ed to create the deduction (the content of this node).
        :param fact_type: Explicit or commonsense
        :param constraints: Not used anymore (see LogicNodeConstraints)
        :param deduction_type: Not used anymore (see LogicNodeDeductionType)
        :param prunable: Can this node be removed from the tree (we don't prune in our datasets)
        :param can_be_leaf: Can this node be a leaf node (usually false for nodes that you are injecting manually)
        :param frozen: Should we add/prune children in the populate function (if frozen, no children will be added or removed, but the children may have children appended/pruned from them).
        """
        self.value = value
        if children is None:
            children = []
        self.children = children
        self.operator = operator
        self.fact_type = fact_type
        self.constraints = constraints
        self.deduction_type = deduction_type
        self.prunable = prunable
        self.can_be_leaf = can_be_leaf
        self.frozen = frozen
        self.parent = None

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children: List['LogicNode']):
        self._children = children
        for c in self.children:
            c.parent = self

    def __str__(self):
        line = []
        cnsts = ', '.join([str(x.value) for x in self.constraints])

        if self.value and self.value != '':
            line.append(self.value)
        if len(self.children) > 0:
            line.append(self.operator)
        else:
            line.append(self.fact_type)

        if self.deduction_type:
            line.append(self.deduction_type)

        if len(self.constraints) > 0:
            line.append(cnsts)

        if len(self.children) > 0:
            line.append(f'children: {len(self.children)}')

        return ' | '.join(line)

    def __repr__(self):
        return str(self)

    def to_json(self):
        return {
            'value': self.value,
            'children': [x.to_json() for x in self.children],
            'fact_type': self.fact_type,
            'operator': self.operator,
            'constraints': self.constraints,
            'deduction_type': self.deduction_type,
            'prunable': self.prunable,
            'can_be_leaf': self.can_be_leaf
        }

    @classmethod
    def from_json(cls, js):
        js['children'] = [LogicNode.from_json(x) for x in js['children']]
        return cls(**js)


class LogicTree:
    """Main datastructure used when creating a MuSR example.

    It's basically a standard tree with some parameters controlling the shape.
    """

    nodes: List[LogicNode]

    chance_of_or: float
    chance_of_cs_fact: float
    depth: int
    chance_to_prune: float
    chance_to_prune_all: float
    bf_factor: Dict[int, float]
    deduction_type_sample_rate: Dict[LogicNodeDeductionType, float]
    root_structure: List[List[LogicNode]] = ()

    def __init__(self,
                 chance_of_or: float = 0.3,
                 chance_of_cs_fact: float = 0.1,
                 depth: int = 2,
                 chance_to_prune: float = 0.6,
                 chance_to_prune_all: float = 0.2,
                 bf_factor: Dict[int, float] = None,
                 deduction_type_sample_rate: Dict[LogicNodeDeductionType,
                                                  float] = None,
                 enforce_cs_fact_per_level: bool = False,
                 root_structure: List[Any] = (),
                 nodes: List[LogicNode] = (),
                 populate: bool = True,
                 prune: bool = True):
        """
        :param chance_of_or: (not used) how often should a node with children be an OR
        :param chance_of_cs_fact: (not used) how often should there be a commonsense node
        :param depth: How deep should a tree go
        :param chance_to_prune: Percentage chance of pruning a node
        :param chance_to_prune_all: Percentage chance of pruning all children from a node.
        :param bf_factor: Branching factor (dictionary of percentages {1: 0.33, 2:0.33, 3:0.33} for example.
        :param deduction_type_sample_rate: (not used, see bf_factor and LogicNodeDeductionType)
        :param enforce_cs_fact_per_level: Enforce 1 commonsense fact per level in the tree (we use this instead of chance_of_cs_fact)
        :param root_structure: List of LogicNodes to build off of.
        :param nodes: List of LogicNodes to define the LogicTree on (we will not populate/prune the tree if this is filled)
        :param populate: Should we populate children for the tree according to the other parameters?
        :param prune: Should we prune the children for the tree according to the other parameters?
        """
        self.chance_of_or = chance_of_or
        self.chance_of_cs_fact = chance_of_cs_fact
        self.depth = depth
        self.chance_to_prune = chance_to_prune
        self.chance_to_prune_all = chance_to_prune_all
        self.bf_factor = bf_factor
        self.enforce_cs_fact_per_level = enforce_cs_fact_per_level

        if not bf_factor:
            self.bf_factor = {2: 0.8, 3: 0.2}
        if not deduction_type_sample_rate:
            deduction_type_sample_rate = {
                LogicNodeDeductionType.SYLLOGISM: 1.0
            }

        self.deduction_type_sample_rate = deduction_type_sample_rate
        self.root_structure = root_structure

        if len(nodes) > 0:
            self.nodes = nodes
        else:

            if root_structure is not None and len(root_structure) > 0:
                self.nodes = root_structure
            else:
                self.nodes = [
                    LogicNode('root', operator=LogicNodeOperatorType.AND)
                ]

            if populate:
                [self.populate(x, 1) for x in self.nodes]
            if prune:
                [self.prune(x, 1) for x in self.nodes]

    def __str__(self):
        return self.print_tree()

    def get_facts(self,
                  include_cs: bool = False,
                  include_deductions_from_level: int = -1,
                  no_facts_after_depth: int = -1):
        """Get a list of LogicNodes from the tree. By default, you will get the
        explicit leaf nodes.

        :param include_cs: Include the commonsense nodes from all levels.
        :param include_deductions_from_level: Include any intermediate
            deduction nodes from the specified level and deeper.
        :param no_facts_after_depth: Essentially tree the deductions at the
            specified depth as leaf nodes.
        """

        def recurse_facts(_node: LogicNode, depth: int = 0) -> List[str]:
            node = deepcopy(_node)
            if depth >= no_facts_after_depth and no_facts_after_depth > -1:
                node.children = []

            facts = []

            if node.fact_type == LogicNodeFactType.EXPLICIT and len(
                    node.children) == 0:
                facts.append(node)
            if node.fact_type == LogicNodeFactType.COMMONSENSE and include_cs and len(
                    node.children) == 0:
                facts.append(node)
            if len(
                    node.children
            ) > 0 and include_deductions_from_level <= depth and include_deductions_from_level > -1:
                facts.append(node)

            for child in node.children:
                facts.extend(recurse_facts(child, depth + 1))
            return list(set(facts))

        facts = []
        for n in self.nodes:
            facts.extend(recurse_facts(n))
        return facts

    def print_tree(self, node=None, level=0):
        """Deprecated (not used)"""
        if node is None:
            node = self.nodes[0]
        line = '-' * level * 4 + str(node) + (' | ' + str(node.operator) if
                                              len(node.children) > 0 else '')

        for child in node.children:
            line += '\n' + self.print_tree(child, level + 1)

        return line

    def print_for_gpt(self,
                      node=None,
                      level=0,
                      pad_char=' ',
                      pad_space=4,
                      print_forward=True,
                      print_conjection_types: bool = False,
                      print_reasoning_types: bool = False,
                      ignore_value_after_depth: int = -1,
                      print_only_nodes_with_value: bool = False):
        """Complex print function.  We often use it as
        print_for_gpt(pad_space=1, pad_char='> ')

        However, more complex arguments can be used to control what is printed.

        This returns a string that must be printed (don't be confused by the
        method name.)

        :param node: Start at a specific node.
        :param level: Controls how much tabbing is done when printing the
            current node.
        :param pad_char: Char to use that specifies depth ('> ' at depth 3 will
            look like '> > > ' if you have pad_space equal to 1 for example)
        :param pad_space: How many spaces to include between pad_chars
        :param print_forward: Print the tree with parent nodes first.
        :param print_conjection_types: Print the Ands and Ors per deduction
            (not used)
        :param print_reasoning_types: Print the deduction types (not used)
        :param ignore_value_after_depth: Ignore content of the nodes once a
            depth is met
        :param print_only_nodes_with_value: Ignore nodes without content.
        """

        line = ''

        if node is None:
            node = self.nodes[0]

        if not print_forward:
            for child in node.children:
                v = self.print_for_gpt(
                    child,
                    level + 1,
                    pad_char=pad_char,
                    pad_space=pad_space,
                    print_forward=print_forward,
                    ignore_value_after_depth=ignore_value_after_depth,
                    print_only_nodes_with_value=print_only_nodes_with_value)
                if v != '':
                    line += v + '\n'

        ignore_val = ignore_value_after_depth > -1 and ignore_value_after_depth < level
        ignore_line = print_only_nodes_with_value and node.value == ''

        if ignore_line:
            line_val = ''
        else:
            line_val = (node.value + ' | ' if node.value != '' and not ignore_val else '') + (
                ('Fact From Story' if node.fact_type == LogicNodeFactType.EXPLICIT else 'Commonsense Knowledge') \
                    if len(node.children) == 0 else 'Deduced Fact')

            if level == 0:
                line_val = (node.value + ' | ' if node.value != '' else
                            '') + 'Deduced Root Conclusion'

            if len(node.children) > 0 and (print_conjection_types
                                           or print_reasoning_types):
                if print_conjection_types:
                    line_val += f' ({node.operator}'
                else:
                    line_val += f'('
                if node.deduction_type and print_reasoning_types:
                    line_val += f' | {node.deduction_type})'
                else:
                    line_val += ')'

            if len(node.constraints) > 0:
                cnsts = ', '.join([str(x) for x in node.constraints])
                line_val += f' constraints: [{cnsts}]'

            line += pad_char * level * pad_space + line_val

        if print_forward:
            for child in node.children:
                v = self.print_for_gpt(
                    child,
                    level + 1,
                    pad_char=pad_char,
                    pad_space=pad_space,
                    print_forward=print_forward,
                    ignore_value_after_depth=ignore_value_after_depth,
                    print_only_nodes_with_value=print_only_nodes_with_value)
                if v != '':
                    line += '\n' + v

        return line

    def populate(self, node: LogicNode, current_depth: int = 1):
        if node.operator == LogicNodeOperatorType.CHOOSE:
            node.operator = LogicNodeOperatorType.OR \
                if random.random() < self.chance_of_or else LogicNodeOperatorType.AND
        if node.deduction_type == LogicNodeDeductionType.CHOOSE:
            if node.operator != LogicNodeOperatorType.AND:
                node.deduction_type = None
            else:
                node.deduction_type = random.choices(
                    list(self.deduction_type_sample_rate.keys()),
                    list(self.deduction_type_sample_rate.values()),
                    k=1)[0]

        if not node.frozen:

            bf = max(
                0,
                random.choices(list(self.bf_factor.keys()),
                               list(self.bf_factor.values()),
                               k=1)[0] - len(node.children))

            if bf > 0:

                new_nodes = []
                one_fact_is_cs = False
                for idx in range(bf):
                    roll_for_or = random.random()
                    fact_type = LogicNodeFactType.COMMONSENSE \
                        if random.random() < self.chance_of_cs_fact and not one_fact_is_cs else \
                        LogicNodeFactType.EXPLICIT

                    if roll_for_or > self.chance_of_or and\
                            current_depth < self.depth and\
                            not fact_type == LogicNodeFactType.COMMONSENSE:
                        new_nodes.append(
                            LogicNode(
                                f'',
                                operator=LogicNodeOperatorType.AND,
                                fact_type=fact_type,
                                deduction_type=random.choices(
                                    list(self.deduction_type_sample_rate.keys(
                                    )),
                                    list(self.deduction_type_sample_rate.
                                         values()),
                                    k=1)[0],
                                prunable=True,
                                can_be_leaf=True,
                            ))
                    else:
                        new_nodes.append(
                            LogicNode(f'',
                                      operator=LogicNodeOperatorType.OR,
                                      fact_type=fact_type,
                                      prunable=True,
                                      can_be_leaf=True))

                    if fact_type == LogicNodeFactType.COMMONSENSE:
                        node.operator = LogicNodeOperatorType.AND
                        if not node.deduction_type:
                            node.deduction_type = random.choices(
                                list(self.deduction_type_sample_rate.keys()),
                                list(self.deduction_type_sample_rate.values()),
                                k=1)[0]
                        one_fact_is_cs = True

                if not one_fact_is_cs and self.enforce_cs_fact_per_level:
                    new_nodes.append(
                        LogicNode(f'',
                                  operator=LogicNodeOperatorType.OR,
                                  fact_type=LogicNodeFactType.COMMONSENSE,
                                  prunable=False,
                                  can_be_leaf=True))

                node.children.extend(new_nodes)

        if current_depth < self.depth:
            for node in node.children:
                if node.fact_type == LogicNodeFactType.COMMONSENSE:
                    continue
                self.populate(node, current_depth + 1)

    def prune(self, node: LogicNode, current_depth: int = 1):
        to_prune = []

        if current_depth > 1 and node.can_be_leaf:
            if random.random() < self.chance_to_prune_all:
                node.children = []
                return

        prunable = [x for x in node.children if x.prunable]
        if (len(prunable) > 1 and node.operator == LogicNodeOperatorType.OR or\
                len(prunable) > 2 and node.operator == LogicNodeOperatorType.AND) and\
                current_depth <= self.depth:

            if node.prunable:
                for n in random.sample(
                        prunable,
                        len(prunable) -
                    (1 if node.operator == LogicNodeOperatorType.OR else 2)):
                    roll_to_prune = random.random()
                    if roll_to_prune < self.chance_to_prune:
                        to_prune.append(n)

        node.children = [x for x in node.children if x not in to_prune]
        for n in node.children:
            self.prune(n, current_depth + 1)

    def to_json(self):
        args = {
            'chance_of_or': self.chance_of_or,
            'depth': self.depth,
            'chance_to_prune': self.chance_to_prune,
            'chance_to_prune_all': self.chance_to_prune_all,
            'bf_factor': self.bf_factor,
            'deduction_type_sample_rate': self.deduction_type_sample_rate,
            'root_structure': [x.to_json() for x in self.root_structure],
            'nodes': [x.to_json() for x in self.nodes]
        }
        return args

    @classmethod
    def from_json(cls, _js):
        js = deepcopy(_js)
        js['nodes'] = [LogicNode.from_json(x) for x in js['nodes']]
        js['root_structure'] = [
            LogicNode.from_json(x) for x in js['root_structure']
        ]
        return cls(**js)


if __name__ == '__main__':
    """EXAMPLE USES."""

    def tv_scene_ex():
        root_structure = [
            LogicNode('A good drama tv scene',
                      operator=LogicNodeOperatorType.OR,
                      prunable=False,
                      can_be_leaf=False,
                      frozen=True)
        ]

        root_structure[0].children = [
            LogicNode('Bob is sad.',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=True,
                      can_be_leaf=False),
            LogicNode('John now hates Bob.',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=True,
                      can_be_leaf=False),
            LogicNode('Bob bought a car.',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=True,
                      can_be_leaf=False),
            LogicNode('Bob wanted to be happy.',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=True,
                      can_be_leaf=False),
        ]

        tree = LogicTree(depth=4,
                         root_structure=root_structure,
                         bf_factor={
                             1: 0.5,
                             2: 0.5
                         },
                         chance_of_or=0.0,
                         chance_of_cs_fact=0.0,
                         chance_to_prune_all=0.5,
                         chance_to_prune=0.5,
                         enforce_cs_fact_per_level=True)

        rep = tree.print_for_gpt(pad_space=1, pad_char='- ')
        print(rep)

    def eb_ex():
        root_structure = [
            LogicNode('',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False,
                      can_be_leaf=False)
        ]

        n = LogicNode('Eruptions block sunlight.',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False,
                      can_be_leaf=False,
                      frozen=True)
        n.children = [
            LogicNode('Eruptions produce ash clouds.',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False,
                      can_be_leaf=True,
                      frozen=True),
            LogicNode('Ash blocks sunlight.',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False,
                      can_be_leaf=True,
                      frozen=True),
        ]

        g = LogicNode('Eruptions can cause plants to die.',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=True,
                      can_be_leaf=False,
                      frozen=True)

        g.children = [
            n,
            LogicNode('Producers will die without sunlight.',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False,
                      can_be_leaf=True,
                      frozen=True)
        ]

        l = LogicNode('',
                      operator=LogicNodeOperatorType.AND,
                      prunable=False,
                      can_be_leaf=False)
        l.children = [g]

        root_structure[0].children = [l]

        tree = LogicTree(depth=5,
                         root_structure=root_structure,
                         bf_factor={
                             1: 0.3,
                             2: 0.7
                         },
                         chance_of_or=0.0,
                         chance_of_cs_fact=0.0,
                         chance_to_prune_all=0.0,
                         chance_to_prune=0.0,
                         enforce_cs_fact_per_level=True)

        rep = tree.print_for_gpt(pad_space=1, pad_char='- ')
        print(rep)

    def murder_mystery_ex():
        root_structure = [
            LogicNode('Killer',
                      operator=LogicNodeOperatorType.OR,
                      constraints=[LogicNodeConstraints.ONLY_ONE_CAN_BE_TRUE],
                      prunable=False,
                      can_be_leaf=False,
                      frozen=True)
        ]

        suspect_nodes = [
            LogicNode(f'Murderer Suspect {idx + 1}',
                      operator=LogicNodeOperatorType.AND,
                      prunable=False,
                      can_be_leaf=False,
                      frozen=True) for idx in range(1)
        ]
        for s in suspect_nodes:
            s.children = [
                LogicNode('Suspect has means',
                          operator=LogicNodeOperatorType.CHOOSE,
                          prunable=True,
                          can_be_leaf=False),
                LogicNode('Suspect has motive',
                          operator=LogicNodeOperatorType.CHOOSE,
                          prunable=True,
                          can_be_leaf=False),
                LogicNode('Suspect has opportunity',
                          operator=LogicNodeOperatorType.CHOOSE,
                          prunable=True,
                          can_be_leaf=False)
            ]
        root_structure[0].children = suspect_nodes

        tree = LogicTree(depth=4,
                         root_structure=root_structure,
                         bf_factor={
                             1: 0.5,
                             2: 0.5
                         },
                         chance_of_or=0.0,
                         chance_of_cs_fact=0.0,
                         chance_to_prune_all=0.5,
                         chance_to_prune=0.5,
                         enforce_cs_fact_per_level=True)

        rep = tree.print_for_gpt(pad_space=1, pad_char='> ')
        print(rep)

    def action_ex():
        root_structure = [
            LogicNode('Take an action',
                      operator=LogicNodeOperatorType.OR,
                      prunable=False,
                      can_be_leaf=False,
                      frozen=True)
        ]

        root_structure[0].children = [
            LogicNode('Run away',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False,
                      can_be_leaf=False,
                      frozen=True),
            LogicNode('Fight back',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False,
                      can_be_leaf=False,
                      frozen=True),
            LogicNode('Hide',
                      operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False,
                      can_be_leaf=False,
                      frozen=True),
        ]

        for cidx, c in enumerate(root_structure[0].children):
            nfacts = random.randint(2, 4)

            for n in range(nfacts):
                fact = LogicNode('',
                                 operator=LogicNodeOperatorType.CHOOSE,
                                 prunable=False,
                                 can_be_leaf=False,
                                 frozen=True)
                fact.children = [
                    LogicNode('Pro (supporting the parent action)',
                              operator=LogicNodeOperatorType.CHOOSE,
                              prunable=True,
                              can_be_leaf=False,
                              frozen=False),
                    LogicNode('Con (counters the sibling Pro only)',
                              operator=LogicNodeOperatorType.CHOOSE,
                              prunable=True,
                              can_be_leaf=False,
                              frozen=False)
                ]
                root_structure[0].children[cidx].children.append(fact)

        tree = LogicTree(depth=4,
                         root_structure=root_structure,
                         bf_factor={
                             1: 0.25,
                             2: 0.5,
                             3: 0.25
                         },
                         chance_of_or=0.0,
                         chance_of_cs_fact=0.0,
                         chance_to_prune_all=0.5,
                         chance_to_prune=0.75,
                         enforce_cs_fact_per_level=True)

        rep = tree.print_for_gpt(pad_space=1, pad_char='- ')
        print(rep)

    tv_scene_ex()
    eb_ex()
    action_ex()
