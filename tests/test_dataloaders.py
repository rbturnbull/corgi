from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from seqbank import SeqBank
from corgi.seqtree import SeqTree
from corgi import dataloaders
from corgi.transforms import GetTensorDNA, RandomSliceBatch, GetXY, DeterministicSliceBatch
from corgi.tensor import TensorDNA
from fastai.data.core import TfmdDL


test_data = Path(__file__).parent / "testdata"



def test_hierarchical_dataloader():
    partitions = 5
    classification_tree = SoftmaxNode("root", count=0)
    bacteria = SoftmaxNode("bacteria", parent=classification_tree, count=0)
    mycobacterium = SoftmaxNode("mycobacterium", parent=bacteria, count=12)
    salmonella = SoftmaxNode("salmonella", parent=bacteria, count=13)

    plant = SoftmaxNode("plant", parent=classification_tree, count=2)
    arabidopsis = SoftmaxNode("arabidopsis", parent=plant, count=4)
    rice = SoftmaxNode("rice", parent=plant, count=5)

    virus = SoftmaxNode("virus", parent=classification_tree, count=4)
    rotavirus = SoftmaxNode("rotavirus", parent=virus, count=8)
    novovirus = SoftmaxNode("novovirus", parent=virus, count=10)

    seqtree = SeqTree(classification_tree)
    for node in classification_tree.post_order_iter():
        for i in range(node.count):
            seqtree.add(f"{node}-{i}", node, i % partitions)
            
    dl = dataloaders.HierarchicalDataloader(seqtree=seqtree, batch_size=4)

    # Check min_items_before_repeat
    assert dl.n == 24
    assert plant.min_items_before_repeat == 8
    assert virus.min_items_before_repeat == 16
    assert bacteria.min_items_before_repeat == 24
    assert salmonella.min_items_before_repeat == 13
    assert rotavirus.min_items_before_repeat == 8
    assert novovirus.min_items_before_repeat == 10
    assert rice.min_items_before_repeat == 5
    assert arabidopsis.min_items_before_repeat == 4
    assert mycobacterium.min_items_before_repeat == 12
    
    indexes = dl.get_idxs()
    assert len(indexes) == 24
    assert len(set(indexes)) == 24

    expected_batches_epoch0 =  [
        ['arabidopsis-2', 'rotavirus-2', 'salmonella-6', 'rice-1'],
        ['novovirus-0', 'mycobacterium-9', 'salmonella-9', 'rotavirus-6'],
        ['rice-1', 'mycobacterium-4', 'novovirus-1', 'arabidopsis-0'],
        ['novovirus-9', 'rice-2', 'salmonella-2', 'mycobacterium-11'],
        ['rotavirus-7', 'arabidopsis-3', 'rotavirus-5', 'arabidopsis-1'],
        ['mycobacterium-0', 'salmonella-8', 'novovirus-8', 'rice-3'],        
    ]
    for batch, expected_batch in zip(dl, expected_batches_epoch0):
        assert batch == expected_batch
        
    expected_batches_epoch1 =  [
        ['rotavirus-6', 'arabidopsis-2', 'mycobacterium-6', 'rice-4'],
        ['novovirus-2', 'salmonella-4', 'mycobacterium-5', 'rice-0'],
        ['rotavirus-3', 'novovirus-5', 'salmonella-11', 'arabidopsis-3'],
        ['rotavirus-4', 'arabidopsis-1', 'salmonella-0', 'mycobacterium-10'],
        ['novovirus-9', 'rice-4', 'mycobacterium-7', 'novovirus-5'],
        ['arabidopsis-0', 'rice-3', 'rotavirus-7', 'salmonella-7'],
    ]
    for batch, expected_batch in zip(dl, expected_batches_epoch1):
        assert batch == expected_batch
    

def test_hierarchical_dataloader_transforms():
    seqtree = SeqTree.load(test_data/"seqtree.pkl")
    seqbank = SeqBank(test_data/"seqbank.sb")
    dl = dataloaders.HierarchicalDataloader(
        seqtree=seqtree, 
        batch_size=4, 
        after_item=GetXY(seqbank=seqbank, seqtree=seqtree),
        before_batch=RandomSliceBatch(maximum=4000),
    )   
    
    total = 0
    for x,y in dl:
        assert x.shape[0] == 4
        assert y.shape[0] == 4
        
        assert x.shape[1] <= 4000
        assert type(x) == TensorDNA
        total += x.shape[0]
    
    assert total == dl.n


def test_validation_dataloader():
    seqtree = SeqTree.load(test_data/"seqtree.pkl")
    seqbank = SeqBank(test_data/"seqbank.sb")
    validation_length = 567

    keys = list(seqtree.keys())

    dl = TfmdDL(
        dataset=keys,
        batch_size=4, 
        after_item=GetXY(seqbank=seqbank, seqtree=seqtree),
        before_batch=DeterministicSliceBatch(seq_length=validation_length),
    )   
    
    total = 0
    for x,y in dl:
        assert x.shape[0] == y.shape[0]
        
        assert x.shape[1] == validation_length
        assert type(x) == TensorDNA
        total += x.shape[0]

    assert total == len(keys)