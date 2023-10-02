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
from .seqdict import SeqDict

@define
class AccessionDetail:
    validation:bool
    node_id:int
    type:int


def open_path(path:Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


class SeqDictSplitter:
    def __init__(self, seqdict:SeqDict, partition:int=1):
        self.seqdict = seqdict
        self.partition = partition

    def __call__(self, objects):
        validation_indexes = mask2idxs(self.seqdict[object].partition == self.partition for object in objects)
        return IndexSplitter(validation_indexes)(objects)


class SeqDictNodeIdGetter:
    def __init__(self, seqdict:SeqDict):
        self.seqdict = seqdict

    def __call__(self, accession:str):
        return self.seqdict[accession].node_id


class SeqDictTypeGetter:
    def __init__(self, seqdict:SeqDict):
        self.seqdict = seqdict

    def __call__(self, accession:str):
        return self.seqdict[accession].type.value




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


def create_datablock_refseq(categories, validation_column="validation", validation_prob=0.2, vocab=None) -> DataBlock:

    # Check if there is a validation column in the dataset otherwise use a random splitter
    if validation_column:
        splitter = ColSplitter(validation_column)
    else:
        splitter = RandomSplitter(valid_pct=validation_prob, seed=42)

    return DataBlock(
        blocks=(TransformBlock, CategoryBlock(vocab=vocab)),
        splitter=splitter,
        get_y=ColReader("category"),
        item_tfms=RowToTensorDNA(categories),
    )


def create_datablock(seq_length=None, validation_column="validation", validation_prob=0.2, vocab=None) -> DataBlock:

    # Check if we need to slice to a specific sequence length
    if seq_length:
        item_tfms = SliceTransform(seq_length)
    else:
        item_tfms = None

    # Check if there is a validation column in the dataset otherwise use a random splitter
    if validation_column:
        splitter = ColSplitter(validation_column)
    else:
        splitter = RandomSplitter(valid_pct=validation_prob, seed=42)

    return DataBlock(
        blocks=(TransformBlock, CategoryBlock(vocab=vocab)),
        splitter=splitter,
        get_x=get_sequence_as_tensor,
        get_y=ColReader("category"),
        item_tfms=item_tfms,
    )


class DataloaderType(str, Enum):
    PLAIN = "PLAIN"
    WEIGHTED = "WEIGHTED"
    STRATIFIED = "STRATIFIED"


class AccessionSplitter:
    def __init__(self, accession_details:dict):
        self.accession_details = accession_details

    def __call__(self, objects):
        validation_indexes = mask2idxs(self.accession_details[object].validation for object in objects)
        return IndexSplitter(validation_indexes)(objects)


class AccessionGetter():
    def __init__(self, accession_details:dict, attribute:str):
        self.accession_details = accession_details
        self.attribute = attribute

    def __call__(self, accession:str):
        return getattr(self.accession_details[accession], self.attribute)


def create_seqbank_dataloaders(
    csv:Path, 
    seqbank:Path, 
    batch_size:int=64, 
    validation_partition:int=1,
    deform_lambda: float = None,
    validation_seq_length:int=1_000, 
    verbose:bool=False,
    label_smoothing:float=0.0,
    gamma:float=0.0,
    **kwargs
):
    seqbank = SeqBank(seqbank)
    csv = Path(csv)
    df = pd.read_csv(csv)

    # Build Hiearchy Tree
    assert 'hierarchy' in df.columns, f"Cannot find 'hierarchy' column in {csv}."
    classification_tree, classification_to_node, classification_to_node_id = create_hierarchy(
        df['hierarchy'].unique(), 
        label_smoothing=label_smoothing, 
        gamma=gamma,
    )

    accession_details = {}
    assert 'partition' in df.columns, f"Cannot find 'partition' column in {csv}."
    assert 'type' in df.columns, f"Cannot find 'type' column in {csv}."
    missing = set()
    for _, row in track(df.iterrows(), description="Reading CSV", total=len(df)):
        accession_details[row['accession']] = AccessionDetail(
            validation=(row['partition'] == validation_partition),
            node_id=classification_to_node_id[row['hierarchy']],
            type=row['type'],
        )
        # if row['accession'] not in seqbank:
        #     missing.add(row['accession'])

    if missing:
        with open("MISSING.txt", "w") as f:
            for accession in missing:
                print(accession, file=f)
        raise ValueError(f"WARNING: {len(missing)} accessions in {csv} are missing from {seqbank}. Written to MISSING.txt")
    
    del df
    
    # Set up batch transforms
    before_batch = [
        RandomSliceBatch(only_split_index=0), 
        DeterministicSliceBatch(seq_length=validation_seq_length, only_split_index=1),
    ]
    if deform_lambda is not None:
        before_batch.append(DeformBatch(deform_lambda=deform_lambda))

    dataloaders_kwargs = dict(bs=batch_size, drop_last=False, before_batch=before_batch)

    getters = [
        GetTensorDNA(seqbank),
        AccessionGetter(accession_details, 'node_id'),
        AccessionGetter(accession_details, 'type'),
    ]

    blocks = (
        TransformBlock, 
        TransformBlock, 
        TransformBlock, 
        # CategoryBlock(vocab=["nuclear", "mitochondrion", "plastid", "plasmid"], sort=False),
    )
    datablock = DataBlock(
        blocks=blocks,
        splitter=AccessionSplitter(accession_details),
        getters=getters,
        n_inp=1,
    )
    dls = datablock.dataloaders(set(accession_details.keys()), verbose=verbose, **dataloaders_kwargs)

    dls.classification_tree = classification_tree

    return dls


def create_seqdict_dataloaders(
    seqdict:SeqDict, 
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
        DeterministicSliceBatch(seq_length=validation_seq_length, only_split_index=1),
    ]
    if deform_lambda is not None:
        before_batch.append(DeformBatch(deform_lambda=deform_lambda))

    dataloaders_kwargs = dict(bs=batch_size, drop_last=False, before_batch=before_batch)

    getters = [
        GetTensorDNA(seqbank),
        SeqDictNodeIdGetter(seqdict),
        SeqDictTypeGetter(seqdict),
    ]

    blocks = (
        TransformBlock, 
        TransformBlock, 
        TransformBlock, 
    )
    datablock = DataBlock(
        blocks=blocks,
        splitter=SeqDictSplitter(seqdict, validation_partition),
        getters=getters,
        n_inp=1,
    )
    dls = datablock.dataloaders(set(seqdict.keys()), verbose=verbose, **dataloaders_kwargs)

    dls.classification_tree = seqdict.classification_tree

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
