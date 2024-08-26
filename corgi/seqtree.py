from typing import List, Optional
from pathlib import Path
from attrs import define, field
from hierarchicalsoftmax import SoftmaxNode
from collections import UserDict
import pickle
from Bio.Seq import Seq
from seqbank import SeqBank
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from seqbank.io import get_file_format
from seqbank.transform import bytes_to_str
from rich.progress import track
import numpy as np
import hashlib
import typer
from plotly import graph_objects as go


def str_to_int_hash(s:str)->int:
    hash_object = hashlib.md5(s.encode())
    hash_digest = hash_object.digest()

    # Convert the hash to an integer
    seed_number = int.from_bytes(hash_digest, 'big')
    seed_number = seed_number % (2**32) # The maximum value is 2**32-1 for a numpy seed

    return seed_number


def node_to_str(node:SoftmaxNode) -> str:
    """ 
    Converts the node to a string
    """
    # return "/".join([str(n) for n in node.ancestors[1:]] + [str(node)])
    result = str(node)
    ancestors = node.ancestors[1:]
    if ancestors:
        result = f"{ancestors[0]}/{result}"
    return result


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
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

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
    
    def accessions_in_partition(self, partition:int):
        for accession, detail in self.items():
            if detail.partition == partition:
                yield accession

    def accessions(self, partition:Optional[int] = None):
        return self.keys() if partition is None else self.accessions_in_partition(partition)
    
    def truncate(self, max_depth:int) -> "SeqTree":
        """
        Truncates the tree to a maximum depth.
        """
        new_tree = SeqTree(self.classification_tree)
        for accession in track(self.keys()):
            node = self.node(accession)
            ancestors = node.ancestors
            if len(ancestors) >= max_depth:
                node = ancestors[max_depth-1]
            new_tree.add(accession, node, self[accession].partition)

        # Remove any nodes that beyond the max depth
        for node in new_tree.classification_tree.pre_order_iter():
            node.readonly = False
            node.softmax_start_index = None
            if len(node.ancestors) >= max_depth:
                node.parent = None

        new_tree.set_indexes()

        return new_tree
            
    def export(self, seqbank:SeqBank, output:Path|str, length:int=0, format:str="", seed:int=0):
        """
        Outputs sequences from a seqbank into a file. 

        If the partition is given then only the accessions in the partition are exported.
        The format of the exported file. If not given, then it will be inferred from the file extension of the output.

        The slice is taken from a random position in the sequence that is at least `length` long.

        Random seed can be set with `seed`.
        """
        format = format or get_file_format(output)
        with open(output, "w") as f:
            for accession, detail in self.items():
                data = seqbank[accession]
                
                # If the sequence is long enough, take a random slice
                if length and length > len(data):
                    start_max = len(data) - length
                    my_seed = seed + str_to_int_hash(accession)
                    np.random.seed(my_seed % 2**32)
                    start = np.random.randint(0, start_max)
                    data = data[start:start+length]

                seq_string = bytes_to_str(data)
                node_str = node_to_str(detail.node)
                record = SeqRecord(
                    Seq(seq_string),
                    id=f"{accession}#{node_str}",
                    description="",
                )

                SeqIO.write(record, f, format)
    
    def add_counts(self):
        """ Adds a count to each node in the tree. """
        for node in self.classification_tree.post_order_iter():
            node.count = 0

        for key in self.keys():
            node = self.node(key)
            node.count += 1

    def render(self, count:bool=False, **kwargs):
        """ Renders the SeqTree. """
        if count:
            self.add_counts()
            for node in self.classification_tree.post_order_iter():
                node.render_str = f"{node.name} ({node.count})" if getattr(node, "count", 0) else node.name

            kwargs['attr'] = "render_str"

        return self.classification_tree.render(**kwargs)
    
    def sunburst(self, **kwargs) -> go.Figure:
        """
        Renders the SeqTree as a sunburst plot.

        The count of accessions at each node is used as the value.

        Returns a plotly figure.
        """
        self.add_counts()
        labels = []
        parents = []
        values = []

        for node in self.classification_tree.pre_order_iter():
            labels.append(node.name)
            parents.append(node.parent.name if node.parent else "")
            values.append(node.count)

        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="remainder",
        ))
        
        fig.update_layout(margin=dict(t=10, l=10, r=10, b=10), **kwargs)
        return fig

    def accessions_to_file(self, file:Path) -> None:
        """ Writes all the accessions of this SeqTree to a file. """
        with open(file, "w") as f:
            for accession in self.keys():
                print(accession, file=f)

    def intersection_seqbank(self, seqbank:SeqBank) -> List[str]:
        """ 
        Removes any accession that is not found in a SeqBank. 
        
        Returns a list of missing accessions.
        """
        my_accessions = self.keys()
        missing = seqbank.missing(my_accessions)
        for accession in missing:
            del self[accession]
        return missing
    


app = typer.Typer()

@app.command()
def intersection_seqbank(
    seqtree:Path = typer.Argument(...,help="The path to the SeqTree."), 
    seqbank:Path = typer.Argument(...,help="The path to the SeqBank."), 
    output:Path = typer.Argument(...,help="The path to the SeqBank."),
):
    """ 
    Takes a SeqTree and removes any accession that is not found in a SeqBank.

    Saves the output SeqTree to `output`.
    """
    seqtree = SeqTree.load(seqtree)
    seqbank = SeqBank(seqbank)
    seqtree.intersection_seqbank(seqbank)
    seqtree.save(output)


@app.command()
def accessions(
    seqtree:Path = typer.Argument(...,help="The path to the SeqTree."), 
    partition:Optional[int] = typer.Option(None,help="The index of the partition to list."), 
):
    """ 
    Prints a list of accessions in a SeqTree. 
    
    If a partition is given, then only the accessions for that partition are given.
    """
    seqtree = SeqTree.load(seqtree)
    for accession in seqtree.accessions(partition=partition):
        print(accession)
    

@app.command()
def export(
    seqtree:Path = typer.Argument(...,help="The path to the SeqTree."), 
    seqbank:Path = typer.Argument(...,help="The path to the SeqBank."), 
    output:Path = typer.Argument(...,help="The path to the output file."), 
    length:int = typer.Option(None,help="If used then a subsequence this maximum length is selected."),
    seed:int = typer.Option(42,help="The random seed to use when slicing the length."),
    partition:Optional[int] = typer.Option(None,help="The index of the partition to include in the export. If not given then all accesions will be exported."), 
    format:str = typer.Option("",help="The format of the exported file. If not given, then it will be inferred from the file extension of the output."), 
):
    seqtree = SeqTree.load(seqtree)
    seqbank = SeqBank(seqbank)
    seqtree.export(seqbank, output, accessions=seqtree.accessions(partition), format=format, length=length, seed=seed)


@app.command()
def render(
    seqtree:Path = typer.Argument(...,help="The path to the SeqTree."), 
    output:Optional[Path] = typer.Option(None, help="The path to save the rendered tree."),
    print:bool = typer.Option(True, help="Whether or not to print the tree to the screen."),
    count:bool = typer.Option(False, help="Whether or not to print the count of accessions at each node."),
):
    seqtree = SeqTree.load(seqtree)
    seqtree.render(filepath=output, print=print, count=count)


@app.command()
def count(
    seqtree:Path = typer.Argument(...,help="The path to the SeqTree."), 
):
    seqtree = SeqTree.load(seqtree)
    print(len(seqtree))


@app.command()
def sunburst(
    seqtree:Path = typer.Argument(...,help="The path to the SeqTree."), 
    show:bool = typer.Option(False, help="Whether or not to show the plot."),
    output:Path = typer.Option(None, help="The path to save the rendered tree."),
    width:int = typer.Option(1000, help="The width of the plot."),
    height:int = typer.Option(0, help="The height of the plot. If 0 then it will be calculated based on the width."),
):
    seqtree = SeqTree.load(seqtree)
    height = height or width

    fig = seqtree.sunburst(width=width, height=height)
    if show:
        fig.show()
    
    if output:
        output = Path(output)
        output.parent.mkdir(exist_ok=True, parents=True)

        # https://github.com/plotly/plotly.py/issues/3469
        import plotly.io as pio   
        pio.kaleido.scope.mathjax = None

        output_func = fig.write_html if output.suffix.lower() == ".html" else fig.write_image
        output_func(output)


@app.command()
def truncate(
    seqtree:Path = typer.Argument(...,help="The path to the SeqTree."), 
    max_depth:int = typer.Argument(...,help="The maximum depth to truncate the tree."),
    output:Path = typer.Argument(...,help="The path to the output file."),
):
    seqtree = SeqTree.load(seqtree)
    new_tree = seqtree.truncate(max_depth)
    new_tree.save(output)


@app.command()
def layer_size(
    seqtree:Path = typer.Argument(...,help="The path to the SeqTree."),         
):
    seqtree = SeqTree.load(seqtree)
    print(seqtree.classification_tree.layer_size)


if __name__ == "__main__":
    app()
