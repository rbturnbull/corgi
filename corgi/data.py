"""Data loading for Corgi: sequence slicing, collation, datasets and dataloaders."""

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


def generate_overlapping_intervals(
    total: int, interval_size: int, min_overlap: int, check: bool = True, variable_size: bool = False
) -> list[tuple[int, int]]:
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
    assert (
        interval_size > min_overlap
    ), f"Max interval size of {interval_size} must be greater than min overlap of {min_overlap}"

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
            overlap, remainder = divmod(num_intervals * interval_size - total, num_intervals - 1)
            if overlap < min_overlap:
                overlap = min_overlap

    while True:
        end = start + interval_size
        if end > total:
            end = total
            start = max(end - interval_size, 0)
        intervals.append((start, end))
        start += interval_size - overlap
        if end >= total:
            break

    if check:
        assert intervals[0][0] == 0
        assert intervals[-1][1] == total
        assert len(intervals) == num_intervals, f"Expected {num_intervals} intervals, got {len(intervals)}"

        assert (
            interval_size <= max_interval_size
        ), f"Interval size of {interval_size} exceeds max interval size of {max_interval_size}"
        for interval in intervals:
            assert (
                interval[1] - interval[0] == interval_size
            ), f"Interval size of {interval[1] - interval[0]} is not the expected size {interval_size}"

        for i in range(1, len(intervals)):
            overlap = intervals[i - 1][1] - intervals[i][0]
            assert (
                overlap >= min_overlap
            ), f"Min overlap condition of {min_overlap} not met for intervals {intervals[i - 1]} and {intervals[i]} (overlap {overlap})"

    return intervals


def slice_tensor(tensor, size, start_index=None, pad: bool = True):
    """Slices a 1D tensor to a given size.

    Args:
        tensor (Tensor): The tensor to slice.
        size (int): The length of the slice.
        start_index (int | None): Where the slice starts. If None, then it is chosen randomly (or 0 if the tensor
            is no longer than `size`).
        pad (bool): Whether or not to pad with zeros at the end if the slice extends beyond the tensor.

    Returns:
        Tensor: The slice.
    """
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
class Collate:
    """Base class for collating a batch of variable length sequences into tensors of the same length."""

    def get_length(self) -> int:
        """Returns the length that the sequences in the next batch will be sliced or padded to."""
        raise NotImplementedError

    def __call__(self, batch):
        """Collates a batch of (sequence, target) pairs.

        Args:
            batch: A list of (sequence tensor, target) pairs.

        Returns:
            tuple[Tensor, Tensor]: The stacked sequences, sliced or padded to the same length, and the targets.
        """
        x_batch, y_batch = zip(*batch)
        length = self.get_length()

        return torch.stack(tuple(slice_tensor(x, length) for x in x_batch)), torch.tensor(y_batch)


@dataclass
class CollateFixedLength(Collate):
    """Collates batches so that every sequence has the same fixed length."""

    length: int

    def get_length(self):
        """Returns the fixed length."""
        return self.length


@dataclass
class CollateRandomLength(Collate):
    """Collates batches to a random length for each batch.

    The lengths are drawn from a skew-normal distribution, truncated to the minimum and maximum lengths.
    """

    minimum_length: int
    maximum_length: int
    skewness: float = 5
    loc: float = 600
    scale: float = 1000
    seed: int = 42

    def __post_init__(self):
        """Checks the lengths and creates the random state and length distribution."""
        assert self.minimum_length > 0
        assert self.minimum_length <= self.maximum_length
        self.random_state = np.random.default_rng(self.seed)
        self.distribution = skewnorm(self.skewness, loc=self.loc, scale=self.scale)

    def get_length(self) -> int:
        """Draws a length from the distribution which is between the minimum and maximum lengths (inclusive)."""
        while True:
            seq_len = int(self.distribution.rvs(random_state=self.random_state))
            if seq_len < self.minimum_length or seq_len > self.maximum_length:
                continue
            return seq_len


@dataclass(kw_only=True)
class CorgiTrainingDataset(Dataset):
    """A dataset of sequences from a SeqBank, with the node in the classification tree as the target.

    Args:
        accessions (list[str]): The accessions in the dataset.
        seqbank (SeqBank): The SeqBank which holds the sequences.
        seqtree (SeqTree): The SeqTree which gives the node for each accession.
        deterministic (bool): Whether or not truncation of long sequences always takes the same slice for an accession.
        maximum (int): The maximum length of a sequence. Longer sequences are truncated. If zero, they are not truncated.
    """

    accessions: list[str]
    seqbank: SeqBank
    seqtree: SeqTree
    deterministic: bool = False
    maximum: int = 0

    def __len__(self):
        """Returns the number of accessions."""
        return len(self.accessions)

    def __getstate__(self):
        # The SeqBank holds an open database handle and a lock which cannot be pickled
        # (needed for DataLoader workers with the 'spawn' and 'forkserver' start methods).
        # Send the path and engine instead and reopen it in the worker.
        """Returns the state for pickling, replacing the SeqBank with its path and engine so it can be reopened in a worker."""
        state = self.__dict__.copy()
        seqbank = state.pop("seqbank")
        state["_seqbank_args"] = (seqbank.path, seqbank.engine)
        return state

    def __setstate__(self, state):
        """Restores the state after pickling and reopens the SeqBank."""
        path, engine = state.pop("_seqbank_args")
        self.__dict__.update(state)
        self.seqbank = SeqBank(path, engine=engine)

    def reopen_seqbank(self):
        """
        Closes the SeqBank and replaces it with a new one on the same path and engine.

        Used in DataLoader workers so that each worker has its own database handle
        rather than one inherited from the parent process by 'fork'.
        """
        seqbank = self.seqbank
        seqbank.close()
        self.seqbank = SeqBank(seqbank.path, engine=seqbank.engine)

    def __getitem__(self, idx):
        """Returns the sequence and target for an accession.

        Sequences longer than `maximum` are truncated to a random slice (or a fixed one if `deterministic`).

        Returns:
            tuple[Tensor, int]: The sequence tensor and the ID of the node in the classification tree.
        """
        accession = self.accessions[idx]
        data = self.seqbank[accession]
        length = len(data)

        # Truncate if necessary
        if self.maximum and length > self.maximum:
            rng = np.random.RandomState(adler32(accession.encode("ascii"))) if self.deterministic else np.random
            start_index = rng.randint(0, length - self.maximum)
            data = data[start_index : start_index + self.maximum]

        array = torch.as_tensor(memoryview(data), dtype=torch.uint8)

        return torch.Tensor(array), self.seqtree[accession].node_id


def seqbank_worker_init(worker_id: int):
    """Gives each DataLoader worker its own SeqBank handle (not shared with the parent process)."""
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.reopen_seqbank()


@dataclass
class CorgiDataModule(L.LightningDataModule):
    """A Lightning data module which loads sequences from a SeqBank for the nodes in a SeqTree.

    The accessions are split into training and validation sets using the partitions in the SeqTree.
    """

    seqtree: SeqTree
    seqbank: SeqBank
    batch_size: int
    max_items: int = 0
    minimum_length: int = 150
    maximum_length: int = 3_000
    validation_length: int = 1000
    validation_partition: int = 1
    test_partition: int = 0
    skewness: float = 5
    loc: float = 600
    scale: float = 1000
    num_workers: int | None = None
    train_all: bool = False

    def __post_init__(self):
        """Initializes the Lightning data module."""
        super().__init__()

    def setup(self, stage=None):
        """Creates the training and validation datasets from the partitions in the SeqTree.

        The test partition is excluded. If there are no validation accessions, then the training set is used for validation.
        """
        if self.num_workers is None:
            self.num_workers = min(os.cpu_count(), 8)

        # Create list of accessions for training and validation depending on validation_partition and test_partition
        training_accessions = []
        validation_accessions = []

        assert (
            self.validation_partition != self.test_partition
        ), f"Validation partition {self.validation_partition} the same as test partition {self.test_partition}"

        for accession, details in self.seqtree.items():
            # Ignore test partition so we do not validate or train on it
            if details.partition == self.test_partition:
                continue

            accessions_list = (
                validation_accessions if details.partition == self.validation_partition else training_accessions
            )
            accessions_list.append(accession)

        if self.train_all:
            training_accessions += validation_accessions

        if len(validation_accessions) == 0:
            console.print(
                f"[bold red]WARNING: No validation accessions found. Check the validation partition {self.validation_partition}. Using all the training examples for validation."
            )
            validation_accessions = training_accessions

        if self.max_items:
            training_accessions = training_accessions[: self.max_items]
            validation_accessions = validation_accessions[: self.max_items]

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
        """Returns the training dataloader, which shuffles the data and collates batches to random lengths."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seqbank_worker_init,
            shuffle=True,
            collate_fn=CollateRandomLength(
                self.minimum_length, self.maximum_length, skewness=self.skewness, loc=self.loc, scale=self.scale
            ),
        )

    def val_dataloader(self):
        """Returns the validation dataloader, which collates batches to a fixed length."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seqbank_worker_init,
            shuffle=False,
            collate_fn=CollateFixedLength(self.validation_length),
        )


class SeqIODataloader:
    """Iterates over sequence files, giving batches of chunks of sequences as tensors for prediction.

    Sequences longer than `max_length` are split into overlapping chunks. Details of each chunk yielded are recorded
    in `chunk_details`.
    """

    def __init__(
        self,
        files,
        batch_size: int = 1,
        min_length: int = 128,
        max_length: int = 5_000,
        max_seqs: int = None,
        overlap: int = 256,
        format: str = "",
    ):
        """Finds the sequence files and counts the chunks.

        Args:
            files: A path or list of paths to sequence files or directories (searched recursively).
            batch_size (int): The number of chunks in each batch.
            min_length (int): Sequences shorter than this are skipped.
            max_length (int): The maximum length of a chunk.
            max_seqs (int | None): The maximum number of sequences to use. If not set, then there is no limit.
            overlap (int): The minimum overlap between consecutive chunks of a sequence.
            format (str): The file format. If not given, then it is inferred from the file extension.
        """
        self.format = format
        self.chunk_details = []
        self.max_length = max_length
        self.batch_size = batch_size
        self.min_length = min_length
        self.count = 0
        self.max_seqs = max_seqs
        self.overlap = overlap
        seqs = 0

        base_extensions = {".fa", ".fasta", ".fna", ".fas"}

        # Function to check if a file matches allowed extensions (including .gz)
        def matches_extensions(file: Path):
            """Returns whether the file has a FASTA extension, with or without a '.gz' suffix."""
            return file.suffix in base_extensions or (
                file.suffix == ".gz" and any(file.stem.endswith(ext) for ext in base_extensions)
            )

        # Expand the list
        self.files = []

        # If 'files' is a string or Path, convert it to a list
        if isinstance(files, (str, Path)):
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
        """Returns the Biopython format name of a file, inferred from its extension unless `format` was given.

        Raises:
            ValueError: If the format cannot be determined.
        """
        if self.format:
            return self.format

        file = Path(file)
        suffixes = [suffix.lower() for suffix in file.suffixes]

        # Check if the last suffix is '.gz' and adjust accordingly
        if suffixes[-1] == '.gz' and len(suffixes) > 1:
            suffix = suffixes[-2]
        else:
            suffix = suffixes[-1]

        if suffix in [".fa", ".fna", ".fasta", ".fas", ".frn", ".ref"]:
            return "fasta"
        if suffix in [".genbank", ".gb", ".gbk"]:
            return "genbank"
        if suffix in [".tab", ".tsv"]:
            return "tsv"
        if suffix in [".fastq", ".fq"]:
            return "fastq"

        raise ValueError(f"Cannot determine file format of {file}.")

    def __len__(self):
        """Returns the total number of chunks."""
        return self.count

    def parse(self, file: Path):
        """Yields the records in a file, which may be gzipped."""
        file_format = self.get_file_format(file)

        if file.name.endswith('.gz'):
            with gzip.open(file, "rt") as f:
                yield from SeqIO.parse(f, file_format)
        else:
            yield from SeqIO.parse(file, file_format)

    def iter_records(self):
        """Yields (file, record) pairs for every record in every file."""
        for file in self.files:
            for record in self.parse(file):
                yield file, record

    def __iter__(self):
        """Yields batches of chunks as a one-element tuple of a padded tensor.

        The details of each chunk are appended to `chunk_details`.
        """
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
                    self.chunk_details.append((file, record.id, record.description, chunk_index))
                    batch.append(t[interval[0] : interval[1]])
                    if len(batch) >= self.batch_size:
                        batch = self.pad(batch)
                        yield batch
                        batch = []

        if batch:
            batch = self.pad(batch)
            yield batch

    def pad(self, batch):
        """Pads the tensors in a batch to the same length and stacks them in a one-element tuple."""
        max_len = 0
        for item in batch:
            max_len = max(item.shape[0], max_len)

        def pad(tensor):
            """Pads a tensor to the maximum length and adds a batch dimension."""
            return slice_tensor(tensor, max_len).unsqueeze(dim=0)

        return (torch.cat(list(map(pad, batch))),)
