import numpy as np
import torch
from fastai.torch_core import TensorBase

vocab_to_int = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 0}
int_to_vocab = dict(zip(vocab_to_int.values(), vocab_to_int.keys()))


class TensorDNA(TensorBase):
    dtype = torch.uint8

    def __str__(self):
        # return str(self.tolist())[:50]
        items = self.tolist()
        truncate_at = 50
        if type(items) == int:
            items = [items]

        length = len(items)

        if length > truncate_at:
            midpoint = truncate_at // 2
            items = items[:midpoint] + [".."] + items[-midpoint:]
        chars = [int_to_vocab[x] if x in int_to_vocab else str(x) for x in items]
        seq_str = "".join(chars)
        return f"{seq_str} [{length}]"

    def show(self, ctx=None, **kwargs):
        return str(self)


def dna_seq_to_numpy(seq) -> np.ndarray:
    """
    Transforms a sequence from biopython to a numpy array.

    Should this be a transform??
    """
    seq_as_numpy = np.array(str(seq), "c")
    seq_as_numpy = seq_as_numpy.view(np.uint8)
    # Ignore any characters in sequence which are below an ascii value of 'A' i.e. 65
    seq_as_numpy = seq_as_numpy[seq_as_numpy >= ord("A")]
    for character, value in vocab_to_int.items():
        seq_as_numpy[seq_as_numpy == ord(character)] = value
    seq_as_numpy = seq_as_numpy[seq_as_numpy < len(vocab_to_int)]
    seq_as_numpy = np.array(seq_as_numpy, dtype="u1")
    return seq_as_numpy


def dna_seq_to_tensor(seq) -> TensorDNA:
    """
    Transforms a a sequence from biopython to a TensorDNA tensor.

    Should this be a pipeline?
    Can we use the ToTensor transform in fastai?
    """
    return TensorDNA(dna_seq_to_numpy(seq))
