from corgi.seqtree import SeqTree
import tempfile
from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode


def test_seqtree():
    seqtree = SeqTree()
    assert type(seqtree) == SeqTree
    assert type(seqtree.classification_tree) == SoftmaxNode

    # create node
    bacteria = SoftmaxNode("bacteria", parent=seqtree.classification_tree)
    plant = SoftmaxNode("plant", parent=seqtree.classification_tree)

    detail = seqtree.add("accession1", bacteria, 0, "NUCLEAR")
    assert detail.partition == 0
    assert detail.type == DNAType.NUCLEAR
    
    detail = seqtree.add("accession2", plant, 1, DNAType.PLASTID)
    assert detail.partition == 1
    assert detail.type == DNAType.PLASTID

    detail = seqtree.add("accession3", plant, 2, 1)
    assert detail.partition == 2
    assert detail.type == DNAType.MITOCHONDRION

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
        
        
def test_seqtree_load():
    seqtree = SeqTree.load(Path(__file__).parent/"testdata/seqtree.pkl")
    assert seqtree.classification_tree.render_equal(
        """
        root
        ├── Virus
        ├── Eukaryota
        ├── Plant
        └── Bacteria
        """        
    )  
    assert len(seqtree) == 5
    assert seqtree["NC_010663.1"].partition == 0
    assert seqtree["NC_010663.1"].type == DNAType.NUCLEAR
    assert seqtree.node("NC_010663.1").name == "Virus"

    assert seqtree["NC_024664.1"].partition == 1
    assert seqtree["NC_024664.1"].type == DNAType.MITOCHONDRION
    assert seqtree.node("NC_024664.1").name == "Eukaryota"

    assert seqtree["NC_036112.1"].partition == 0
    assert seqtree["NC_036112.1"].type == DNAType.PLASTID
    assert seqtree.node("NC_036112.1").name == "Plant"

    assert seqtree["NC_036113.1"].partition == 1
    assert seqtree["NC_036113.1"].type == DNAType.PLASTID
    assert seqtree.node("NC_036113.1").name == "Plant"

    assert seqtree["NZ_JAJNFP010000161.1"].partition == 0
    assert seqtree["NZ_JAJNFP010000161.1"].type == DNAType.NUCLEAR
    assert seqtree.node("NZ_JAJNFP010000161.1").name == "Bacteria"


