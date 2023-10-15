from pathlib import Path
from attrs import define, field
from hierarchicalsoftmax import SoftmaxNode
from collections import UserDict
import pickle


@define
class SeqDetail:
    partition:int
    node:SoftmaxNode = field(default=None, eq=False)
    node_id:int = None

    def __getstate__(self):
        return (self.partition, self.node_id)

    def __setstate__(self, state):
        self.partition, self.node_id = state
        self.node = None

class AlreadyExists(Exception):
    pass


class SeqTree(UserDict):
    def __init__(self, classification_tree=None):
        super().__init__()
        self.classification_tree = classification_tree or SoftmaxNode("root")

    def add(self, accession:str, node:SoftmaxNode, partition:int):
        assert node.root == self.classification_tree
        if accession in self:
            old_node = self.node(accession)
            if not node == old_node:
                raise AlreadyExists(f"Accession {accession} already exists in SeqTree at node {self.node(accession)}. Cannot change to {node}")

        detail = SeqDetail(
            partition=partition,
            node=node,
        )
        self[accession] = detail
        return detail

    def set_indexes(self):
        self.classification_tree.set_indexes_if_unset()
        for detail in self.values():
            if detail.node:
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