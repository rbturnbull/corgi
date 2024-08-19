from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from seqbank import SeqBank
from corgi.seqtree import SeqTree
from corgi import data
from corgi.archive.transforms import GetTensorDNA, RandomSliceBatch, GetXY, DeterministicSliceBatch
from corgi.archive.tensor import TensorDNA


test_data = Path(__file__).parent / "testdata"



    

def test_create_training_dataloader():
    seqtree = SeqTree.load(test_data/"seqtree.pkl")
    seqbank = SeqBank(test_data/"seqbank.sb")
    dl = data.create_training_dataloader(seqtree=seqtree, seqbank=seqbank, validation_partition=1, batch_size=4, maximum=567)
    total = 0
    for x,y in dl:
        assert x.shape[0] == 4
        assert y.shape[0] == 4
        
        assert x.shape[1] <= 567
        assert type(x) == TensorDNA
        total += x.shape[0]
    
    assert total == dl.n


def test_validation_dataloader():
    seqtree = SeqTree.load(test_data/"seqtree.pkl")
    seqbank = SeqBank(test_data/"seqbank.sb")
    breakpoint()
    validation_length = 567
    batch_size = 4

    dl = data.create_validation_dataloader(
        seqtree=seqtree, 
        seqbank=seqbank, 
        batch_size=batch_size, 
        validation_length=validation_length,
        validation_partition=1,
    )
    
    total = 0
    for x,y in dl:
        assert x.shape[0] == y.shape[0]
        assert x.shape[0] < batch_size
        
        assert x.shape[1] == validation_length
        assert type(x) == TensorDNA
        total += x.shape[0]

    assert total == dl.n == 2


def test_dataloaders():
    validation_length = 567
    seqtree = SeqTree.load(test_data/"seqtree.pkl")
    seqbank = SeqBank(test_data/"seqbank.sb")
    batch_size = 4

    dls = data.create_dataloaders(seqtree=seqtree, seqbank=seqbank, batch_size=batch_size, validation_partition=0, validation_length=validation_length)

    total = 0
    for x,y in dls.valid:
        assert x.shape[0] == y.shape[0]
        assert x.shape[0] < batch_size
        
        assert x.shape[1] == validation_length
        assert type(x) == TensorDNA
        total += x.shape[0]

    assert total == dls.valid.n == 1
