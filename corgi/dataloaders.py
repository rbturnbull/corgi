from enum import Enum
import random
from itertools import chain

import gzip
import pandas as pd
from pathlib import Path
import numpy as np

from Bio import SeqIO

from fastcore.foundation import L
from fastcore.dispatch import typedispatch
from fastcore.meta import delegates

from fastai.data.core import TfmdDL, DataLoaders, get_empty_df
from fastai.callback.data import WeightedDL
from fastai.data.block import DataBlock, TransformBlock, CategoryBlock
from fastai.torch_core import display_df
from fastai.data.transforms import ColSplitter, ColReader, RandomSplitter

from .tensor import TensorDNA, dna_seq_to_numpy, dna_seq_to_tensor
from .transforms import RandomSliceBatch, SliceTransform, RowToTensorDNA
from .refseq import RefSeqCategory


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


def get_sequence_as_tensor(row):
    return TensorDNA(row["sequence"])


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


class DataloaderType(Enum):
    PLAIN = 0
    WEIGHTED = 1
    STRATIFIED = 2


def create_dataloaders_refseq_path(dataframe_path: Path, base_dir: Path, batch_size=64, **kwargs):
    dataframe_path = Path(dataframe_path)

    print('Training using:\t', dataframe_path)
    if dataframe_path.suffix == ".parquet":
        df = pd.read_parquet(str(dataframe_path), engine="pyarrow")
    else:
        df = pd.read_csv(str(dataframe_path))

    print(f'Dataframe has {len(df)} sequences.')
    dls = create_dataloaders_refseq(df, batch_size=batch_size, base_dir=base_dir, **kwargs)
    return dls


def create_dataloaders_refseq(
    df: pd.DataFrame,
    base_dir: Path,
    batch_size=64,
    dataloader_type: DataloaderType = DataloaderType.PLAIN,
    verbose: bool = True,
    **kwargs,
) -> DataLoaders:
    categories = [RefSeqCategory(name, base_dir=base_dir) for name in df.category.unique()]

    dataloaders_kwargs = dict(bs=batch_size, drop_last=False, before_batch=RandomSliceBatch)

    validation_column = "validation"
    if validation_column not in df:
        validation_column = None

    print("Creating Datablock")
    vocab = df['category'].unique()
    datablock = create_datablock_refseq(categories, validation_column=validation_column, vocab=vocab, **kwargs)

    if validation_column in df:
        training_df = df[df[validation_column] == 0].reset_index()

        if dataloader_type == DataloaderType.STRATIFIED:
            print("Creating groups for balancing dataset")
            groups = [training_df.index[training_df['category'] == name] for name in vocab]

            dataloaders_kwargs['dl_type'] = StratifiedDL
            dataloaders_kwargs['dl_kwargs'] = [dict(groups=groups), dict()]
        elif dataloader_type == DataloaderType.WEIGHTED:
            print("Creating weights for balancing dataset")
            weights = np.zeros((len(training_df),))
            value_counts = training_df['category'].value_counts()
            for name in df.category.unique():
                weight = value_counts.max() / value_counts[name]
                print(f"\tWeight for {name}: {weight}")
                weights[training_df['category'] == name] = weight

            dataloaders_kwargs['dl_type'] = WeightedDL
            dataloaders_kwargs['dl_kwargs'] = [dict(wgts=weights), dict()]

    print("Creating Dataloaders")
    return datablock.dataloaders(df, verbose=verbose, **dataloaders_kwargs)


def create_dataloaders(df: pd.DataFrame, batch_size=64, **kwargs) -> DataLoaders:
    datablock = create_datablock(**kwargs)
    return datablock.dataloaders(df, bs=batch_size, drop_last=False)


def fasta_to_dataframe(
    fasta_path,
    max_seqs=None,
    validation_from_filename=True,
    validation_prob=0.2,
):
    """
    Creates a pandas dataframe from a fasta file.

    If validation_from_filename is True then it checks if 'valid' or 'train' is in the filename,
    otherwise it falls back to using the validation_prob.
    If 'valid' or 'train' is in the filename and validation_from_filename is True then validation_prob is ignored.
    """
    fasta_path = Path(fasta_path)
    print(f"Processing:\t{fasta_path}")

    if not fasta_path.exists():
        raise FileNotFoundError(f"Cannot find fasta file {fasta_path}.")

    data = []
    if fasta_path.suffix == ".gz":
        fasta = gzip.open(fasta_path, "rt")
    else:
        fasta = open(fasta_path, "rt")

    if validation_from_filename:
        if "valid" in str(fasta_path):
            validation = 1
        elif "train" in str(fasta_path):
            validation = 0
        else:
            validation_from_filename = False

    seqs = SeqIO.parse(fasta, "fasta")
    for seq_index, seq in enumerate(seqs):
        if max_seqs and seq_index >= max_seqs:
            break

        if not validation_from_filename:
            validation = int(random.random() < validation_prob)

        seq_as_numpy = dna_seq_to_tensor(seq.seq)
        data.append([seq.id, seq.description, seq_as_numpy, validation])

    fasta.close()

    df = pd.DataFrame(data, columns=["id", "description", "sequence", "validation"])
    df["file"] = str(fasta_path)
    df["category"] = fasta_path.name.split(".")[0]
    return df


def fastas_to_dataframe(fasta_paths, **kwargs):
    dfs = [fasta_to_dataframe(fasta_path, **kwargs) for fasta_path in fasta_paths]
    return pd.concat(dfs)


def create_dataloaders_from_fastas(fasta_paths, batch_size=64, seq_length=None, **kwargs) -> DataLoaders:
    """
    Creates a DataLoaders object from a list of fasta paths.
    """
    df = fastas_to_dataframe(fasta_paths, **kwargs)
    return create_dataloaders(df, batch_size=batch_size, seq_length=seq_length)


class FastaDataloader:
    def __init__(self, fasta_files, device):
        self.fasta_files = list(fasta_files)
        self.device = device

    def __iter__(self):
        self.randomize()
        self.before_iter()
        self.__idxs = self.get_idxs()  # called in context of main process (not workers/subprocesses)
        for b in _loaders[self.fake_l.num_workers == 0](self.fake_l):
            if self.device is not None:
                b = to_device(b, self.device)
            yield self.after_batch(b)
        self.after_iter()
        if hasattr(self, 'it'):
            del self.it
