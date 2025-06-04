#!/usr/bin/env python
# flake8: noqa
#Original Authors: Tim Henderson and Steve Johnson
#Email: tim.tadh@gmail.com, steve@steveasleep.com
#For licensing see the LICENSE file in the top level directory.

# This is a modified version of zss package.

import collections

import numpy as np
from numpy import ones, zeros


class Node(object):

    def __init__(self, label, children=None):
        self.label = label
        self.children = children or list()

    @staticmethod
    def get_children(node):
        return node.children

    @staticmethod
    def get_label(node):
        return node.label

    def addkid(self, node, before=False):

        if before: self.children.insert(0, node)
        else: self.children.append(node)
        return self

    def get(self, label):

        if self.label == label: return self
        for c in self.children:
            if label in c: return c.get(label)


class AnnotatedTree(object):

    def __init__(self, root, get_children):
        self.get_children = get_children

        self.root = root
        self.nodes = list(
        )  # a post-order enumeration of the nodes in the tree
        self.ids = list()  # a matching list of ids
        self.lmds = list()  # left most descendents of each nodes
        self.keyroots = None
        # the keyroots in the original paper

        stack = list()
        pstack = list()
        stack.append((root, collections.deque()))
        j = 0
        while len(stack) > 0:
            n, anc = stack.pop()
            nid = j
            for c in self.get_children(n):
                a = collections.deque(anc)
                a.appendleft(nid)
                stack.append((c, a))
            pstack.append(((n, nid), anc))
            j += 1
        lmds = dict()
        keyroots = dict()
        i = 0
        while len(pstack) > 0:
            (n, nid), anc = pstack.pop()
            self.nodes.append(n)
            self.ids.append(nid)
            if not self.get_children(n):
                lmd = i
                for a in anc:
                    if a not in lmds: lmds[a] = i
                    else: break
            else:
                try:
                    lmd = lmds[nid]
                except:
                    import pdb
                    pdb.set_trace()
            self.lmds.append(lmd)
            keyroots[lmd] = i
            i += 1
        self.keyroots = sorted(keyroots.values())


def ext_distance(A, B, get_children, single_insert_cost, insert_cost,
                 single_remove_cost, remove_cost, update_cost):
    '''Computes the extended tree edit distance between trees A and B with extended-zss algorithm
    Args:
        A(Node): Root node of tree 1
        B(Node): Root node of tree 2
        get_children(Func): the get_children method of tree
        single_insert_cost(Func): cost of inserting single node
        insert_cost(Func): cost of inserting a subtree
        update_cost(Func): cost of updating A to B


    Return:
        Distance(float):the tree editing distance
    '''
    A, B = AnnotatedTree(A, get_children), AnnotatedTree(B, get_children)
    size_a = len(A.nodes)
    size_b = len(B.nodes)
    treedists = zeros((size_a, size_b), float)
    fd = 1000 * ones((size_a + 1, size_b + 1), float)
    operations = [[[] for _ in range(size_b)] for _ in range(size_a)]

    def treedist(x, y):
        Al = A.lmds
        Bl = B.lmds
        An = A.nodes
        Bn = B.nodes

        m = size_a
        n = size_b

        fd[Al[x]][Bl[y]] = 0
        for i in range(Al[x], x + 1):
            node = An[i]
            fd[i + 1][Bl[y]] = fd[Al[i]][Bl[y]] + remove_cost(node)

        for j in range(Bl[y], y + 1):
            node = Bn[j]

            fd[Al[x]][j + 1] = fd[Al[x]][Bl[j]] + insert_cost(node)

        for i in range(Al[x], x + 1):
            for j in range(Bl[y], y + 1):

                node1 = An[i]
                node2 = Bn[j]
                costs = [
                    fd[i][j + 1] + single_remove_cost(node1),
                    fd[i + 1][j] + single_insert_cost(node2),
                    fd[Al[i]][j + 1] + remove_cost(node1),
                    fd[i + 1][Bl[j]] + insert_cost(node2)
                ]
                m = min(costs)

                if Al[x] == Al[i] and Bl[y] == Bl[j]:
                    treedists[i][j] = min(m,
                                          fd[i][j] + update_cost(node1, node2))
                    fd[i + 1][j + 1] = treedists[i][j]
                else:
                    fd[i + 1][j + 1] = min(m,
                                           fd[Al[i]][Bl[j]] + treedists[i][j])

    for x in A.keyroots:
        for y in B.keyroots:
            treedist(x, y)

    return treedists[-1][-1]
