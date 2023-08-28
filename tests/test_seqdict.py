from corgi.seqdict import SeqDict, DNAType
import tempfile
from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode


def test_seqdict():
    seqdict = SeqDict()
    assert type(seqdict) == SeqDict
    assert type(seqdict.classification_tree) == SoftmaxNode

    # create node
    bacteria = SoftmaxNode("bacteria", parent=seqdict.classification_tree)
    plant = SoftmaxNode("plant", parent=seqdict.classification_tree)

    detail = seqdict.add("accession1", bacteria, 0, "NUCLEAR")
    assert detail.partition == 0
    assert detail.type == DNAType.NUCLEAR
    
    detail = seqdict.add("accession2", plant, 1, DNAType.PLASTID)
    assert detail.partition == 1
    assert detail.type == DNAType.PLASTID

    detail = seqdict.add("accession3", plant, 2, 1)
    assert detail.partition == 2
    assert detail.type == DNAType.MITOCHONDRION

    assert set(seqdict.keys()) == set("accession1 accession2 accession3".split())

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        filepath = tmpdirname/"seqdict.pkl"
        seqdict.save(filepath)
    
        assert filepath.exists()
        seqdict2 = SeqDict.load(filepath)
        assert type(seqdict2) == SeqDict
        assert type(seqdict2.classification_tree) == SoftmaxNode
        assert len(seqdict2) == len(seqdict)

        for accession in seqdict.keys():
            assert seqdict2[accession] == seqdict[accession]
            assert seqdict2[accession].node_id is not None
        
        
def test_seqdict_load():
    seqdict = SeqDict.load(Path(__file__).parent/"testdata/seqdict.pkl")
    assert seqdict.classification_tree.render_equal(
        """
        root
        ├── Virus
        ├── Eukaryota
        ├── Plant
        └── Bacteria
        """        
    )  
    assert len(seqdict) == 5
    assert seqdict["NC_010663.1"].partition == 0
    assert seqdict["NC_010663.1"].type == DNAType.NUCLEAR
    assert seqdict.node("NC_010663.1").name == "Virus"

    assert seqdict["NC_024664.1"].partition == 1
    assert seqdict["NC_024664.1"].type == DNAType.MITOCHONDRION
    assert seqdict.node("NC_024664.1").name == "Eukaryota"

    assert seqdict["NC_036112.1"].partition == 0
    assert seqdict["NC_036112.1"].type == DNAType.PLASTID
    assert seqdict.node("NC_036112.1").name == "Plant"

    assert seqdict["NC_036113.1"].partition == 1
    assert seqdict["NC_036113.1"].type == DNAType.PLASTID
    assert seqdict.node("NC_036113.1").name == "Plant"

    assert seqdict["NZ_JAJNFP010000161.1"].partition == 0
    assert seqdict["NZ_JAJNFP010000161.1"].type == DNAType.NUCLEAR
    assert seqdict.node("NZ_JAJNFP010000161.1").name == "Bacteria"


