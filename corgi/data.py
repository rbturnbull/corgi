import os
import random
from dataclasses import dataclass
from zlib import adler32
import numpy as np
from scipy.stats import rv_continuous, skewnorm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import lightning as L
from seqbank import SeqBank

from .seqtree import SeqTree


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

        array = torch.frombuffer(data, dtype=torch.uint8)
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

    def setup(self, stage=None):
        if self.num_workers is None:
            self.num_workers = min(os.cpu_count(), 8)

        # Create list of accessions for training and validation depending on validation_partition and test_partition
        training_accessions = []
        validation_accessions = []
        for accession, details in self.seqtree.items():
            # Ignore test partition so we do not validate or train on it
            if details.partition == self.test_partition:
                continue

            accessions_list = validation_accessions if details.partition == self.validation_partition else training_accessions
            accessions_list.append(accession)

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

