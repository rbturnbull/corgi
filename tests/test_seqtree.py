
import tempfile
from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from corgi.seqtree import SeqTree, str_to_int_hash


def test_seqtree():
    seqtree = SeqTree()
    assert type(seqtree) == SeqTree
    assert type(seqtree.classification_tree) == SoftmaxNode

    # create node
    bacteria = SoftmaxNode("bacteria", parent=seqtree.classification_tree)
    plant = SoftmaxNode("plant", parent=seqtree.classification_tree)

    detail = seqtree.add("accession1", bacteria, 0)
    assert detail.partition == 0    
    detail = seqtree.add("accession2", plant, 1)
    assert detail.partition == 1

    detail = seqtree.add("accession3", plant, 2)
    assert detail.partition == 2
    assert set(seqtree.keys()) == set("accession1 accession2 accession3".split())

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        filepath = tmpdirname/"seqtree.pkl"
        seqtree.save(filepath)
    
        assert filepath.exists()
        seqtree2 = SeqTree.load(filepath)
        assert type(seqtree2) == SeqTree
        assert type(seqtree2.classification_tree) == SoftmaxNode
        assert len(seqtree2) == len(seqtree)

        for accession in seqtree.keys():
            assert seqtree2[accession] == seqtree[accession]
            assert seqtree2[accession].node_id is not None
                
def test_seqtree_save():
    seqtree = SeqTree()
    assert type(seqtree) == SeqTree
    assert type(seqtree.classification_tree) == SoftmaxNode

    # create node
    bacteria = SoftmaxNode("Bacteria", parent=seqtree.classification_tree)
    virus = SoftmaxNode("Virus", parent=seqtree.classification_tree)
    eukaryota = SoftmaxNode("Eukaryota", parent=seqtree.classification_tree)
    plant = SoftmaxNode("Plant", parent=seqtree.classification_tree)

    detail = seqtree.add("NZ_JAJNFP010000161.1", bacteria, 0)
    assert detail.partition == 0    
    
    detail = seqtree.add("NC_024664.1", eukaryota, 1)
    assert detail.partition == 1

    detail = seqtree.add("NC_010663.1", virus, 1)
    assert detail.partition == 1

    detail = seqtree.add("NC_036112.1", plant, 2)
    assert detail.partition == 2

    detail = seqtree.add("NC_036113.1", plant, 2)
    assert detail.partition == 2

    assert set(seqtree.keys()) == set("NZ_JAJNFP010000161.1 NC_024664.1 NC_010663.1 NC_036112.1 NC_036113.1".split())

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        filepath = tmpdirname/"seqtree.pkl"
        seqtree.save(filepath)
    
        assert filepath.exists()
        seqtree2 = SeqTree.load(filepath)
        assert type(seqtree2) == SeqTree
        assert type(seqtree2.classification_tree) == SoftmaxNode
        assert len(seqtree2) == len(seqtree)

        for accession in seqtree.keys():
            assert seqtree2[accession] == seqtree[accession]
            assert seqtree2[accession].node_id is not None
        
        
def test_seqtree_load():
    seqtree = SeqTree.load(Path(__file__).parent/"testdata/seqtree.pkl")
    assert seqtree.classification_tree.render_equal(
        """
        root
        ├── Bacteria
        ├── Virus
        ├── Eukaryota
        └── Plant
        """        
    )  
    assert len(seqtree) == 5
    assert seqtree["NC_010663.1"].partition == 1
    assert seqtree.node("NC_010663.1").name == "Virus"

    assert seqtree["NC_024664.1"].partition == 1
    assert seqtree.node("NC_024664.1").name == "Eukaryota"

    assert seqtree["NC_036112.1"].partition == 2
    assert seqtree.node("NC_036112.1").name == "Plant"

    assert seqtree["NC_036113.1"].partition == 2
    assert seqtree.node("NC_036113.1").name == "Plant"

    assert seqtree["NZ_JAJNFP010000161.1"].partition == 0
    assert seqtree.node("NZ_JAJNFP010000161.1").name == "Bacteria"


def test_str_to_int_hash():
    assert str_to_int_hash("hello") == 269993362
    assert str_to_int_hash("This is a test string!3289470#") == 989461991
