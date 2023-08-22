from dataclasses import dataclass
import pandas as pd
import random
import torch
from torch import tensor
import torch.nn as nn

from fastcore.transform import Transform
import numpy as np
from Bio.SeqRecord import SeqRecord
from scipy.stats import nbinom

from .tensor import TensorDNA
from .seqbank import SeqBank


class SplitTransform(Transform):
    """ 
    Only performs the transform if the split index matches `only_split_index`
    """
    def __init__(self, only_split_index=None, **kwargs):
        super().__init__(**kwargs)
        self.current_split_idx = None
        self.only_split_index = only_split_index

    def __call__(self, 
        b, 
        split_idx:int=None, # Index of the train/valid dataset
        **kwargs
    ):
        if self.only_split_index == split_idx or self.only_split_index is None:
            return super().__call__(b, split_idx=split_idx, **kwargs)
        return b


def slice_tensor(tensor, size, start_index=None):
    original_length = tensor.shape[0]
    if start_index is None:
        if original_length <= size:
            start_index = 0
        else:
            start_index = random.randrange(0, original_length - size)
    end_index = start_index + size
    if end_index > original_length:
        sliced = tensor[start_index:]
        sliced = nn.ConstantPad1d((0, end_index - original_length), 0)(sliced)
    else:
        sliced = tensor[start_index:end_index]
    return sliced


class SliceTransform(Transform):
    def __init__(self, size):
        self.size = size

    def encodes(self, tensor: TensorDNA):
        return slice_tensor(tensor, self.size)


VOCAB = "NACGT"
CHAR_TO_INT = dict(zip(VOCAB, range(len(VOCAB))))


def char_to_int(c):
    return CHAR_TO_INT.get(c, 0)


class CharsToTensorDNA(Transform):
    def encodes(self, seq: list):
        return TensorDNA([char_to_int(c) for c in seq])

    def encodes(self, seq: SeqRecord):
        seq_as_numpy = np.array(seq, "c")
        seq_as_numpy = seq_as_numpy.view(np.uint8)
        # Ignore any characters in sequence which are below an ascii value of 'A' i.e. 65
        seq_as_numpy = seq_as_numpy[seq_as_numpy >= ord("A")]
        for character, value in CHAR_TO_INT.items():
            seq_as_numpy[seq_as_numpy == ord(character)] = value
        seq_as_numpy = seq_as_numpy[seq_as_numpy < len(CHAR_TO_INT)]

        return TensorDNA(seq_as_numpy)


@dataclass
class GetTensorDNA(Transform):
    seqbank:SeqBank

    def encodes(self, row: pd.Series):
        return TensorDNA(self.seqbank[row['accession']])

    def encodes(self, accession: str):
        return TensorDNA(self.seqbank[accession])


class RandomSliceBatch(SplitTransform):
    rand_generator = None

    def __init__(self, rand_generator=None, distribution=None, minimum: int = 150, maximum: int = 3_000, **kwargs):
        super().__init__(**kwargs)

        self.rand_generator = rand_generator or self.default_rand_generator
        if distribution is None:
            from scipy.stats import skewnorm

            distribution = skewnorm(5, loc=600, scale=1000)
        self.distribution = distribution
        self.minimum = minimum
        self.maximum = maximum

    def default_rand_generator(self):
        # return random.randint(self.minimum, self.maximum)

        seq_len = int(self.distribution.rvs())
        seq_len = max(self.minimum, seq_len)
        seq_len = min(self.maximum, seq_len)
        return seq_len

    def encodes(self, batch):
        seq_len = self.rand_generator()
        # seq_len = 150  # hack

        def slice(tensor):
            return (slice_tensor(tensor[0], seq_len),) + tensor[1:]

        return list(map(slice, batch))


class DeterministicSliceBatch(SplitTransform):
    def __init__(self, seq_length, **kwargs):
        super().__init__(**kwargs)

        self.seq_length = seq_length

    def slice(self, tensor):
        return (slice_tensor(tensor[0], self.seq_length, start_index=0),) + tensor[1:]

    def encodes(self, batch):
        return list(map(self.slice, batch))


class ShortRandomSliceBatch(RandomSliceBatch):
    def __init__(self, rand_generator=None, distribution=None, minimum: int = 80, maximum: int = 150):
        if distribution is None:
            from scipy.stats import uniform

            distribution = uniform(loc=minimum, scale=maximum - minimum)
        super().__init__(rand_generator=rand_generator, distribution=distribution, minimum=minimum, maximum=maximum)


class PadBatch(Transform):
    def encodes(self, batch):
        max_len = 0
        for item in batch:
            max_len = max(item[0].shape[0], max_len)

        def pad(tensor):
            return (slice_tensor(tensor[0], max_len),) + tensor[1:]

        return list(map(pad, batch))


class PadBatchX(Transform):
    def encodes(self, batch):
        max_len = 0
        for item in batch:
            max_len = max(item.shape[0], max_len)

        def pad(tensor):
            return slice_tensor(tensor, max_len).unsqueeze(dim=0)

        return torch.cat(list(map(pad, batch))),


class DeformBatch(SplitTransform):
    def __init__(self, deform_lambda=4, **kwargs):
        super().__init__(**kwargs)
        self.deform_lambda = deform_lambda
        self.distribution = torch.distributions.exponential.Exponential(deform_lambda)
        self.states = len(VOCAB)

    # def encodes_as_tensor(self, batch):
    #     print(type(batch), len(batch), type(batch[0]), len(batch[0]))
    #     assert 0
    #     times = self.distribution.sample()
    #     alt_states = self.states-1
    #     probability_same = 1.0/self.states + alt_states/self.states * torch.exp(-times)
    #     probability_same = probability_same.unsqueeze(1)
    #     weights = torch.cat( [probability_same, (1-probability_same).repeat(1,alt_states)/alt_states], dim=1)
    #     batch += torch.multinomial(weights, batch.shape[1], replacement=True)
    #     batch = batch % self.states
        
    #     print(weights)
    #     return batch

    def deform(self, tensor):
        times = self.distribution.sample()
        alt_states = self.states-1
        probability_same = 1.0/self.states + alt_states/self.states * torch.exp(-times)
        # weights = torch.cat( [probability_same, (1-probability_same).repeat(1,alt_states)/alt_states], dim=1)
        weights = torch.as_tensor([probability_same] + alt_states * [(1-probability_same)/alt_states])
        x = tensor[0] + torch.multinomial(weights, len(tensor[0]), replacement=True)
        x = x % self.states

        return (x,) + tensor[1:]

    def encodes(self, batch):
        batch = list(map(self.deform, batch))
        return batch
