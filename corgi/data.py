import os
import gzip
import random
from dataclasses import dataclass
from zlib import adler32
import numpy as np
from pathlib import Path
from scipy.stats import rv_continuous, skewnorm
import torch
from Bio import SeqIO
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import lightning as L
from seqbank import SeqBank
from seqbank.transform import seq_to_numpy

from .seqtree import SeqTree

from rich.console import Console

console = Console()

def generate_overlapping_intervals(total: int, interval_size: int, min_overlap: int, check:bool=True, variable_size:bool=False) -> list[tuple[int, int]]:
    """
    Creates a list of overlapping intervals within a specified range, adjusting the interval size to ensure
    that the overlap is approximately the same across all intervals.

    Args:
        total (int): The total range within which intervals are to be created.
        max_interval_size (int): The maximum size of each interval.
        min_overlap (int): The minimum number of units by which consecutive intervals overlap.
        check (bool): If True, checks are performed to ensure that the intervals meet the specified conditions.

    Returns:
        list[tuple[int, int]]: A list of tuples where each tuple represents the start (inclusive) 
        and end (exclusive) of an interval.

    Example:
        >>> generate_overlapping_intervals(20, 5, 2)
        [(0, 5), (3, 8), (6, 11), (9, 14), (12, 17), (15, 20)]
    """
    if total <= interval_size:
        return [(0, total)]
    
    intervals = []
    start = 0

    if total == 0:
        return intervals
    
    max_interval_size = interval_size
    assert interval_size
    assert min_overlap is not None
    assert interval_size > min_overlap, f"Max interval size of {interval_size} must be greater than min overlap of {min_overlap}"

    # Calculate the number of intervals needed to cover the range
    num_intervals, remainder = divmod(total - min_overlap, interval_size - min_overlap)
    if remainder > 0:
        num_intervals += 1

    # Calculate the exact interval size to ensure consistent overlap
    overlap = min_overlap
    if variable_size:
        if num_intervals > 1:
            interval_size, remainder = divmod(total + (num_intervals - 1) * overlap, num_intervals)
            if remainder > 0:
                interval_size += 1
    else:
        # If the size is fixed, then vary the overlap to keep it even
        if num_intervals > 1:
            overlap, remainder = divmod( num_intervals * interval_size - total, num_intervals - 1)
            if overlap < min_overlap:
                overlap = min_overlap

    while True:
        end = start + interval_size
        if end > total:
            end = total
            start = max(end - interval_size,0)
        intervals.append((start, end))
        start += interval_size - overlap
        if end >= total:
            break

    if check:
        assert intervals[0][0] == 0
        assert intervals[-1][1] == total
        assert len(intervals) == num_intervals, f"Expected {num_intervals} intervals, got {len(intervals)}"

        assert interval_size <= max_interval_size, f"Interval size of {interval_size} exceeds max interval size of {max_interval_size}"
        for interval in intervals:
            assert interval[1] - interval[0] == interval_size, f"Interval size of {interval[1] - interval[0]} is not the expected size {interval_size}"

        for i in range(1, len(intervals)):
            overlap = intervals[i - 1][1] - intervals[i][0]
            assert overlap >= min_overlap, f"Min overlap condition of {min_overlap} not met for intervals {intervals[i - 1]} and {intervals[i]} (overlap {overlap})"

    return intervals

def slice_tensor(tensor, size, start_index=None, pad:bool=True):
    original_length = tensor.shape[0]
    if start_index is None:
        if original_length <= size:
            start_index = 0
        else:
            start_index = random.randrange(0, original_length - size)
    end_index = start_index + size
    if end_index > original_length and pad:
        sliced = tensor[start_index:]
        sliced = nn.ConstantPad1d((0, end_index - original_length), 0)(sliced)
    else:
        sliced = tensor[start_index:end_index]
    return sliced


@dataclass
class Collate():
    def get_length(self) -> int:
        raise NotImplementedError

    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)
        length = self.get_length()

        return torch.stack(tuple(slice_tensor(x, length) for x in x_batch)), torch.tensor(y_batch)


@dataclass
class CollateFixedLength(Collate):
    length:int

    def get_length(self):
        return self.length
    

@dataclass
class CollateRandomLength(Collate):
    minimum_length:int
    maximum_length:int
    skewness:float=5
    loc:float=600
    scale:float=1000
    seed:int=42

    def __post_init__(self):
        assert self.minimum_length > 0
        assert self.minimum_length <= self.maximum_length
        self.random_state = np.random.default_rng(self.seed)
        self.distribution = skewnorm(self.skewness, loc=self.loc, scale=self.scale)

    def get_length(self) -> int:
        while True:
            seq_len = int(self.distribution.rvs(random_state=self.random_state))
            if seq_len < self.minimum_length or seq_len > self.maximum_length:
                continue
            return seq_len



@dataclass(kw_only=True)
class CorgiTrainingDataset(Dataset):
    accessions: list[str]
    seqbank:SeqBank
    seqtree:SeqTree
    deterministic:bool=False
    maximum:int=0

    def __len__(self):
        return len(self.accessions)

    def __getitem__(self, idx):
        accession = self.accessions[idx]
        data = self.seqbank[accession]
        length = len(data)

        # Truncate if necessary
        if self.maximum and length > self.maximum:
            rng = np.random.RandomState(adler32(accession.encode("ascii"))) if self.deterministic else np.random
            start_index = rng.randint(0, length - self.maximum)
            data = data[start_index:start_index+self.maximum]

        array = torch.as_tensor(memoryview(data), dtype=torch.uint8)

        return torch.Tensor(array), self.seqtree[accession].node_id


@dataclass
class CorgiDataModule(L.LightningDataModule):
    seqtree:SeqTree 
    seqbank:SeqBank 
    batch_size:int
    max_items:int=0
    minimum_length: int = 150
    maximum_length: int = 3_000
    validation_length:int=1000
    validation_partition:int=1
    test_partition:int=0
    skewness:float=5
    loc:float=600
    scale:float=1000
    num_workers:int|None = None
    train_all:bool=False

    def __post_init__(self):
        super().__init__()

    def setup(self, stage=None):
        if self.num_workers is None:
            self.num_workers = min(os.cpu_count(), 8)

        # Create list of accessions for training and validation depending on validation_partition and test_partition
        training_accessions = []
        validation_accessions = []

        assert self.validation_partition != self.test_partition, f"Validation partition {self.validation_partition} the same as test partition {self.test_partition}"
        
        for accession, details in self.seqtree.items():
            # Ignore test partition so we do not validate or train on it
            if details.partition == self.test_partition:
                continue

            accessions_list = validation_accessions if details.partition == self.validation_partition else training_accessions
            accessions_list.append(accession)

        if self.train_all:
            training_accessions += validation_accessions

        if len(validation_accessions) == 0:
            console.print(f"[bold red]WARNING: No validation accessions found. Check the validation partition {self.validation_partition}. Using all the training examples for validation.")
            validation_accessions = training_accessions
        
        if self.max_items:
            training_accessions = training_accessions[:self.max_items]
            validation_accessions = validation_accessions[:self.max_items]

        self.train_dataset = CorgiTrainingDataset(
            accessions=training_accessions, 
            seqtree=self.seqtree, 
            seqbank=self.seqbank, 
            deterministic=False, 
            maximum=self.maximum_length,
        )
        self.val_dataset = CorgiTrainingDataset(
            accessions=validation_accessions,
            seqtree=self.seqtree,
            seqbank=self.seqbank,
            deterministic=True,
            maximum=self.validation_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True, 
            collate_fn=CollateRandomLength(self.minimum_length, self.maximum_length, skewness=self.skewness, loc=self.loc, scale=self.scale),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            collate_fn=CollateFixedLength(self.validation_length),
        )


class SeqIODataloader:
    def __init__(self, files, batch_size:int=1, min_length:int=128, max_length:int=5_000, max_seqs:int=None, overlap:int=256, format:str=""):
        self.format = format
        self.chunk_details = []
        self.max_length = max_length
        self.batch_size = batch_size
        self.min_length = min_length
        self.count = 0
        self.max_seqs = max_seqs
        self.overlap = overlap
        seqs = 0

        base_extensions = {".fa", ".fasta", ".fna"}

        # Function to check if a file matches allowed extensions (including .gz)
        def matches_extensions(file: Path):
            return (
                file.suffix in base_extensions or
                (file.suffix == ".gz" and any(file.stem.endswith(ext) for ext in base_extensions))
            )

        # Expand the list
        self.files = []

        # If 'files' is a string or Path, convert it to a list
        if isinstance(files, (str,Path)):
            files = [Path(files)]

        for path in files:
            path = Path(path)
            if path.is_dir():
                # If it's a directory, find all files with the specified extensions
                self.files.extend([file for file in path.rglob("*") if matches_extensions(file)])
            else:
                # If it's not a directory, add the file to the list
                if matches_extensions(path):
                    self.files.append(path)

        for file in self.files:
            for record in self.parse(file):
                if len(record.seq) < self.min_length:
                    continue

                if self.max_seqs and seqs >= self.max_seqs:
                    break

                intervals = generate_overlapping_intervals(len(record.seq), self.max_length, self.overlap)
                self.count += len(intervals)
                seqs += 1

    def get_file_format(self, file):
        if self.format:
            return self.format
        
        file = Path(file)
        suffixes = [suffix.lower() for suffix in file.suffixes]

        # Check if the last suffix is '.gz' and adjust accordingly
        if suffixes[-1] == '.gz' and len(suffixes) > 1:
            suffix = suffixes[-2]
        else:
            suffix = suffixes[-1]

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

    def parse(self, file:Path):
        file_format = self.get_file_format(file)
        
        if file.name.endswith('.gz'):
            with gzip.open(file, "rt") as f:
                yield from SeqIO.parse(f, file_format)
        else:
            yield from SeqIO.parse(file, file_format)

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
                t = torch.tensor(seq_to_numpy(str(record.seq)))

                intervals = generate_overlapping_intervals(len(t), self.max_length, self.overlap)

                for chunk_index, interval in enumerate(intervals):
                    self.chunk_details.append( (file, record.id, record.description, chunk_index) )
                    batch.append(t[interval[0]:interval[1]])
                    if len(batch) >= self.batch_size:
                        batch = self.pad(batch)
                        yield batch
                        batch = []

        if batch:
            batch = self.pad(batch)
            yield batch

    def pad(self, batch):
        max_len = 0
        for item in batch:
            max_len = max(item.shape[0], max_len)

        def pad(tensor):
            return slice_tensor(tensor, max_len).unsqueeze(dim=0)

        return torch.cat(list(map(pad, batch))),
