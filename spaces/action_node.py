from __future__ import annotations
from collections import OrderedDict
from enum import Enum
from typing import *

from .discrete import Discrete


class DiscreteNode(Discrete):
    def __init__(self, n, name: Union[Enum, str],
                 is_function_node: bool = False):
        super(DiscreteNode, self).__init__(n)
        self.name = name
        self.is_function_node = is_function_node
        self.edges: Dict[str, DiscreteNode] = OrderedDict()
        self.edge_data: List[str] = []
        self.edge_nodes: List[DiscreteNode] = []

    def add_branch(self, data: str, node: DiscreteNode = None):
        self.edges[data] = node
        self.edge_data.append(data)
        self.edge_nodes.append(node)

    def add_branches(self, data: List[str], nodes: [DiscreteNode] = None):
        assert len(data) == self.n, 'please add all branches'
        if isinstance(nodes, Iterable):
            [self.add_branch(datum, node) for datum, node in zip(data, nodes)]
        else:
            [self.add_branch(datum, nodes) for datum in data]

    def __repr__(self):
        return "DiscreteNode(%s %d)" % (self.name, self.n)

    def __hash__(self):
        return hash(str(self))

    # def __eq__(self, other):
    #     return self.name == other.name
