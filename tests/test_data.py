from pathlib import Path
import torch
from torch import Tensor
from seqbank import SeqBank
from corgi.seqtree import SeqTree
from corgi.data import CollateRandomLength, CollateFixedLength, CorgiDataModule


TEST_DATA = Path(__file__).parent / "testdata"


def test_collate_random_lengths():
    collation = CollateRandomLength(minimum_length=150, maximum_length=3_000)
    samples = 2_000
    lengths = torch.tensor([collation.get_length() for _ in range(samples)])
    assert lengths.min() >= 150
    assert lengths.max() <= 3_000
    assert 1300 < lengths.float().mean() < 1400
    
    plot = False
    if plot:
        import plotly.express as px
        fig = px.histogram(lengths, nbins=100)
        fig.show()


def test_collate_random():
    collation = CollateRandomLength(minimum_length=150, maximum_length=3_000)
    batch = [
        (Tensor([1,2,3,4]),0),
        (Tensor([1,2,3,4,1,2,3,4,1,2,3,4,1,1,1]),1),
    ]
    x,y = collation(batch)
    assert x.shape == (2, 694)
    assert y.shape == (2,)
    assert x[0].tolist() == [1,2,3,4] + [0]*690
    assert x[1].tolist() == [1,2,3,4,1,2,3,4,1,2,3,4,1,1,1] + [0]*679


def test_collate_fixed():
    collation = CollateFixedLength(length=20)

    batch = [
        (Tensor([1,2,3,4]),0),
        (Tensor([1,2,3,4,1,2,3,4,1,2,3,4,1,1,1]),1),
    ]

    x,y = collation(batch)

    assert x.shape == (2, 20)
    assert y.shape == (2,)
    assert x[0].tolist() == [1,2,3,4] + [0]*16
    assert x[1].tolist() == [1,2,3,4,1,2,3,4,1,2,3,4,1,1,1] + [0]*5


def test_train_dataloader():
    seqtree = SeqTree.load(TEST_DATA/"seqtree.pkl")
    seqbank = SeqBank(TEST_DATA/"seqbank.sb")
    data = CorgiDataModule(
        seqtree=seqtree, 
        seqbank=seqbank, 
        validation_partition=1, 
        batch_size=4, 
        maximum_length=667,
        num_workers=0,
    )
    data.setup()
    dataloader = data.train_dataloader()
    x,y = next(iter(dataloader))
    assert x.shape == (2, 663)
    assert y.shape == (2,)


def test_val_dataloader():
    seqtree = SeqTree.load(TEST_DATA/"seqtree.pkl")
    seqbank = SeqBank(TEST_DATA/"seqbank.sb")
    data = CorgiDataModule(
        seqtree=seqtree, 
        seqbank=seqbank, 
        validation_partition=1, 
        batch_size=4, 
        maximum_length=667,
        validation_length=999,
        num_workers=1,
    )
    data.setup()
    dataloader = data.val_dataloader()
    x,y = next(iter(dataloader))
    assert x.shape == (2, 999)
    assert y.shape == (2,)
