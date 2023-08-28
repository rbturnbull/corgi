from pathlib import Path
from attrs import define, field
from hierarchicalsoftmax import SoftmaxNode
from collections import UserDict
import pickle
from enum import Enum


class DNAType(Enum):
    NUCLEAR = 0
    MITOCHONDRION = 1
    PLASTID = 2
    PLASMID = 3


def to_dna_type(value) -> DNAType:
    if isinstance(value, int):
        return DNAType(value)
    elif isinstance(value, str):
        return DNAType[value.upper()]

    return value


@define
class SeqDetail:
    partition:int
    type:DNAType = field(converter=to_dna_type)
    node:SoftmaxNode = field(default=None, eq=False)
    node_id:int = None

    def __getstate__(self):
        return (self.partition, self.type, self.node_id)

    def __setstate__(self, state):
        self.partition, self.type, self.node_id = state
        self.node = None


class SeqDict(UserDict):
    def __init__(self):
        super().__init__()
        self.classification_tree = SoftmaxNode("root")

    def add(self, accession:str, node:SoftmaxNode, partition:int, type:DNAType):
        assert node.root == self.classification_tree
        assert accession not in self, f"Accession {accession} already exists in SeqDict"
        detail = SeqDetail(
            partition=partition,
            type=type,
            node=node,
        )
        self[accession] = detail
        return detail

    def set_indexes(self):
        self.classification_tree.set_indexes_if_unset()
        for detail in self.values():
            detail.node_id = self.classification_tree.node_to_id[detail.node]

    def save(self, path:Path):
        self.set_indexes()
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(self, path:Path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)

    def node(self, accession:str):
        detail = self[accession]
        if detail.node is not None:
            return detail.node
        return self.classification_tree.node_list[detail.node_id]




