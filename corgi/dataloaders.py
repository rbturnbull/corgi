import torch
from enum import Enum
import random
from itertools import chain
from attrs import define

import gzip
import pandas as pd
from pathlib import Path
import numpy as np
from rich.progress import track

from Bio import SeqIO
from fastcore.foundation import mask2idxs
from fastai.data.transforms import IndexSplitter 

from fastcore.foundation import L
from fastcore.dispatch import typedispatch
from fastcore.meta import delegates
from rich.progress import track
from fastai.data.core import TfmdDL, DataLoaders, get_empty_df
from fastai.callback.data import WeightedDL
from fastai.data.block import DataBlock, TransformBlock, CategoryBlock
from fastai.torch_core import display_df
from fastai.data.transforms import ColSplitter, ColReader, RandomSplitter
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode

from .tensor import TensorDNA, dna_seq_to_tensor
from .transforms import RandomSliceBatch, SliceTransform, GetTensorDNA, PadBatchX, DeterministicSliceBatch, DeformBatch
from .hierarchy import create_hierarchy
from seqbank import SeqBank
from .seqtree import SeqTree


def open_path(path:Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


class SeqTreeSplitter:
    def __init__(self, seqtree:SeqTree, partition:int=1):
        self.seqtree = seqtree
        self.partition = partition

    def __call__(self, objects):
        validation_indexes = mask2idxs(self.seqtree[object].partition == self.partition for object in objects)
        return IndexSplitter(validation_indexes)(objects)


class SeqTreeNodeIdGetter:
    def __init__(self, seqtree:SeqTree):
        self.seqtree = seqtree

    def __call__(self, accession:str):
        return self.seqtree[accession].node_id


@delegates()
class StratifiedDL(TfmdDL):
    def __init__(self, dataset=None, bs=None, groups=None, **kwargs):
        super().__init__(dataset=dataset, bs=bs, **kwargs)
        self.groups = [list(group) for group in groups] if groups else None
        self.min_length = None
        if not self.groups or not self.shuffle:
            return

        for group in self.groups:
            if self.min_length is None:
                self.min_length = len(group)
                continue
            self.min_length = min(self.min_length, len(group))
        self.queues = [self.shuffle_fn(group) for group in self.groups]
        self.n = self.min_length * len(self.queues)

    def get_idxs(self):
        if not self.groups or not self.shuffle:
            return super().get_idxs()

        epoch_indexes = []
        for i, queue in enumerate(self.queues):
            if len(queue) < self.min_length:
                queue += self.shuffle_fn(self.groups[i])

            epoch_indexes.append(queue[: self.min_length])
            self.queues[i] = queue[self.min_length :]

        return list(chain(*zip(*epoch_indexes)))


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


class DataloaderType(str, Enum):
    PLAIN = "PLAIN"
    WEIGHTED = "WEIGHTED"
    STRATIFIED = "STRATIFIED"


def create_seqtree_dataloaders(
    seqtree:SeqTree, 
    seqbank:SeqBank, 
    batch_size:int=64, 
    validation_partition:int=1,
    deform_lambda: float = None,
    validation_seq_length:int=1_000, 
    verbose:bool=False,
    label_smoothing:float=0.0,
    gamma:float=0.0,
    **kwargs
):    
    # Set up batch transforms
    before_batch = [
        RandomSliceBatch(only_split_index=0), 
        DeterministicSliceBatch(seq_length=validation_partition, only_split_index=1),
    ]
    if deform_lambda is not None:
        before_batch.append(DeformBatch(deform_lambda=deform_lambda))

    dataloaders_kwargs = dict(bs=batch_size, drop_last=False, before_batch=before_batch)

    getters = [
        GetTensorDNA(seqbank),
        SeqTreeNodeIdGetter(seqtree),
    ]

    blocks = (
        TransformBlock, 
        TransformBlock, 
    )
    datablock = DataBlock(
        blocks=blocks,
        splitter=SeqTreeSplitter(seqtree, validation_partition),
        getters=getters,
        n_inp=1,
    )
    dls = datablock.dataloaders(set(seqtree.keys()), verbose=verbose, **dataloaders_kwargs)

    dls.classification_tree = seqtree.classification_tree

    return dls


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
