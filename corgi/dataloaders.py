import random
import random
from typing import List

import gzip
import pandas as pd
from pathlib import Path

from Bio import SeqIO
from fastcore.foundation import mask2idxs
from fastai.data.transforms import IndexSplitter 

from fastcore.foundation import L
from fastcore.dispatch import typedispatch
from fastai.data.core import TfmdDL, DataLoaders, get_empty_df
from fastai.torch_core import display_df
from hierarchicalsoftmax import SoftmaxNode

from seqbank import SeqBank

from .tensor import TensorDNA, dna_seq_to_tensor
from .transforms import PadBatchX, DeterministicSliceBatch, GetXY, RandomSliceBatch, PadBatch

from .seqtree import SeqTree



def open_path(path:Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")



@typedispatch
def show_batch(x: TensorDNA, y, samples, ctxs=None, max_n=20, trunc_at=150, **kwargs):
    if ctxs is None:
        ctxs = get_empty_df(min(len(samples), max_n))
    if trunc_at is not None:
        samples = L((s[0], *s[1:]) for s in samples)
    ctxs = [(sample[0].show(), str(sample[1])) for sample in samples]
    df = pd.DataFrame(ctxs, columns=["x", "y"])
    display_df(df)
    return ctxs


class SeqIODataloader:
    def __init__(self, files, device, batch_size:int=1, min_length:int=128, max_length:int=5_000, max_seqs:int=None, format:str=""):
        self.files = list(files)
        self.device = device
        self.format = format
        self.chunk_details = []
        self.max_length = max_length
        self.batch_size = batch_size
        self.min_length = min_length
        self.pad = PadBatchX()
        self.count = 0
        self.max_seqs = max_seqs
        seqs = 0
        for file in self.files:
            for record in self.parse(file):
                if len(record.seq) < self.min_length:
                    continue

                if self.max_seqs and seqs >= self.max_seqs:
                    break

                chunks = len(record.seq)//self.max_length + 1
                self.count += chunks
                seqs += 1


    def get_file_format(self, file):
        if self.format:
            return self.format
        
        file = Path(file)
        suffix = file.suffix.lower()

        if suffix in [".fa", ".fna", ".fasta"]:
            return "fasta"

        if suffix in [".genbank", ".gb", ".gbk"]:
            return "genbank"

        if suffix in [".tab", ".tsv"]:
            return "tsv"

        if suffix in [".fastq", ".fq"]:
            return "fastq"

        raise ValueError(f"Cannot determine file format of {file}.")
    
    def __len__(self):
        return self.count

    def parse(self, file):
        return SeqIO.parse(file, self.get_file_format(file))

    def iter_records(self):
        for file in self.files:
            for record in self.parse(file):
                yield file, record

    def __iter__(self):
        batch = []
        seqs = 0

        for file in self.files:
            for record in self.parse(file):
                if len(record.seq) < self.min_length:
                    continue

                if self.max_seqs and seqs >= self.max_seqs:
                    break

                seqs += 1
                t = dna_seq_to_tensor(record.seq)
                chunks = len(t)//self.max_length + 1

                for chunk_index, chunk in enumerate(t.chunk(chunks)):
                    self.chunk_details.append( (file, record.id, chunk_index) )
                    batch.append(chunk)
                    if len(batch) >= self.batch_size:
                        batch = self.pad(batch)
                        yield batch
                        batch = []

        if batch:
            batch = self.pad(batch)
            yield batch


class HierarchicalDataloader(TfmdDL):
    def __init__(self, seqtree:SeqTree, exclude_partition:int=None, seed:int=42, batch_size=None, **kwargs):
        # Later put in seqbank:SeqBank
        classification_tree = seqtree.classification_tree
        self.seqtree = seqtree
        self.exclude_partition = exclude_partition
        self.seed = seed

        dataset = []
        count = 0

        # Add items to nodes
        for node in classification_tree.post_order_iter():
            node.idxs = set()
    
        # Loop Through Tree and add accessions
        for accession, detail in seqtree.items(): # should this be self.items
            if exclude_partition is not None and exclude_partition == detail.partition:
                continue
            
            node = seqtree.node(accession)

            # Only allow leaf/tip nodes to have items
            if not node.is_leaf:
                continue

            node.idxs.add(count)
            dataset.append(accession)
            count += 1

        # Get epoch size from the minimum number of items before it could repeat and create initial queues
        random.seed(seed)
        for node in classification_tree.post_order_iter():
            if node.children:
                child_min_items_before_repeat = min([
                    child.min_items_before_repeat 
                    for child in node.children 
                    if child.min_items_before_repeat
                ])
                node.min_items_before_repeat = child_min_items_before_repeat * len(node.children)
            elif node.idxs:
                node.min_items_before_repeat = len(node.idxs)  
            else:
                node.min_items_before_repeat = None

            # Initialize queue            
            node.queue = None

        super().__init__(dataset=dataset, batch_size=batch_size, **kwargs)
        self.n = classification_tree.min_items_before_repeat

    def node_queue(self, node):
        if not node.queue:
            node.queue = list(node.idxs) if node.idxs else list(node.children)
            random.shuffle(node.queue)  

        return node.queue

    def next_idx(self, node=None):
        """
        Return the next index to reference the dataset.
        """
        node = node or self.seqtree.classification_tree
        queue = self.node_queue(node)
        # print(node, node.queue)

        if not queue:
            return None
        
        item = queue.pop()
        if isinstance(item, SoftmaxNode):
            return self.next_idx(item)
        
        return item

    def get_idxs(self) -> List[int]:
        """
        Return a list of indices to reference the dataset.
        """
        indexes = []
        while len(indexes) < self.n:
            next_index = self.next_idx()
            if next_index:
                indexes.append(next_index)
        return indexes
        

def create_hierarchical_training_dataloader(
    seqtree:SeqTree, 
    seqbank:SeqBank, 
    batch_size:int, 
    validation_partition:int, 
    minimum: int = 150, 
    maximum: int = 3_000,
) -> HierarchicalDataloader:
    return HierarchicalDataloader(
        seqtree=seqtree, 
        batch_size=batch_size, 
        exclude_partition=validation_partition,
        after_item=GetXY(seqbank=seqbank, seqtree=seqtree),
        before_batch=RandomSliceBatch(maximum=maximum, minimum=minimum),
    )   


class CorgiDataloaders(DataLoaders):
    def __init__(
        self, 
        *loaders, # `DataLoader` objects to wrap
        path:str='.', # Path to store export objects
        device=None, # Device to put `DataLoaders`
        classification_tree:SoftmaxNode=None,
    ):
        super().__init__(*loaders, path=path, device=device)
        self.classification_tree = classification_tree

    def new_empty(self):
        loaders = [dl.new([]) for dl in self.loaders]
        return type(self)(*loaders, path=self.path, device=self.device, classification_tree=self.classification_tree)


def create_training_dataloader(
    seqtree:SeqTree, 
    seqbank:SeqBank, 
    batch_size:int, 
    validation_partition:int, 
    minimum: int = 150, 
    maximum: int = 3_000,
    max_seqs: int = 0,
) -> TfmdDL:
    accessions = []
    for accession, details in seqtree.items():
        if details.partition != validation_partition:
            accessions.append(accession)

    if max_seqs:
        accessions = accessions[:max_seqs]

    return TfmdDL(
        dataset=accessions,
        batch_size=batch_size, 
        shuffle=True,
        after_item=GetXY(seqbank=seqbank, seqtree=seqtree, maximum=maximum, deterministic=False),
        before_batch=RandomSliceBatch(maximum=maximum, minimum=minimum),
    )   


def create_validation_dataloader(
    seqtree:SeqTree, 
    seqbank:SeqBank, 
    batch_size:int, 
    validation_partition:int, 
    validation_length:int,
    max_seqs: int = 0,
) -> TfmdDL:
    accessions = []
    for accession, details in seqtree.items():
        if details.partition == validation_partition:
            accessions.append(accession)

    if max_seqs:
        accessions = accessions[:max_seqs]

    return TfmdDL(
        dataset=accessions,
        batch_size=batch_size, 
        shuffle=False,
        after_item=GetXY(seqbank=seqbank, seqtree=seqtree, maximum=validation_length, deterministic=True),
        before_batch=PadBatch(),
    )   


def create_dataloaders(
    seqtree:SeqTree, 
    seqbank:SeqBank, 
    batch_size:int, 
    validation_partition:int,
    validation_length:int=1_000,    
    minimum: int = 150, 
    maximum: int = 3_000, 
    hierarchical: bool = False,
    max_seqs:int = 0,
) -> CorgiDataloaders:    
    training_dataloader_func = create_hierarchical_training_dataloader if hierarchical else create_training_dataloader
    train_dl = training_dataloader_func(
        seqtree=seqtree, 
        seqbank=seqbank, 
        batch_size=batch_size,
        validation_partition=validation_partition,
        minimum=minimum, 
        maximum=maximum, 
        max_seqs=max_seqs,
    )

    valid_dl = create_validation_dataloader(
        seqtree=seqtree, 
        seqbank=seqbank, 
        batch_size=batch_size,
        validation_length=validation_length, 
        validation_partition=validation_partition,
        max_seqs=max_seqs,
    )
    dls = CorgiDataloaders(train_dl, valid_dl, classification_tree=seqtree.classification_tree)
    return dls

