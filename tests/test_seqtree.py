
import tempfile
import pytest
from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from typer.testing import CliRunner
from corgi.seqtree import SeqTree, app, str_to_int_hash


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


def test_seqtree_merge(tmp_path):
    seqtree = SeqTree()
    bacteria = SoftmaxNode("Bacteria", parent=seqtree.classification_tree)
    shared = SoftmaxNode("Shared", parent=bacteria)
    seqtree.add("existing", shared, 0)
    seqtree.add("duplicate", bacteria, 0)

    other = SeqTree()
    other_bacteria = SoftmaxNode("Bacteria", parent=other.classification_tree)
    other_shared = SoftmaxNode("Shared", parent=other_bacteria)
    new_child = SoftmaxNode("New", parent=other_bacteria)
    virus = SoftmaxNode("Virus", parent=other.classification_tree)
    different_parent = SoftmaxNode("Shared", parent=virus)
    SoftmaxNode("Unused", parent=virus)
    other.add("matching", other_shared, 1)
    other.add("new", new_child, 2)
    other.add("different-parent", different_parent, 3)
    other.add("duplicate", virus, 4)

    # Exercise read-only indexed trees and details containing only node IDs.
    seqtree.save(tmp_path / "self.st")
    other.save(tmp_path / "other.st")
    seqtree = SeqTree.load(tmp_path / "self.st")
    other = SeqTree.load(tmp_path / "other.st")
    existing_node = seqtree.node("existing")
    seqtree.merge(other)

    assert set(seqtree) == {"existing", "duplicate", "matching", "new", "different-parent"}
    assert seqtree.node("existing") is existing_node
    assert seqtree.node("matching") is existing_node
    assert seqtree.node("different-parent") is not existing_node
    assert seqtree.node("different-parent").parent.name == "Virus"
    assert seqtree.node("new").parent is existing_node.parent
    assert seqtree.node("duplicate").name == "Virus"
    assert [seqtree[key].partition for key in
            ("existing", "matching", "new", "different-parent", "duplicate")] == [0, 1, 2, 3, 4]
    assert seqtree.classification_tree.render_equal("""
        root
        ├── Bacteria
        │   ├── Shared
        │   └── New
        └── Virus
            ├── Shared
            └── Unused
    """)
    assert other.node("matching").root is other.classification_tree
    assert other["matching"].node is None
    assert len(other.classification_tree.descendants) == 6

    seqtree.save(tmp_path / "merged.st")
    restored = SeqTree.load(tmp_path / "merged.st")
    for accession in seqtree:
        assert restored[accession].partition == seqtree[accession].partition
        assert [node.name for node in restored.node(accession).path] == [
            node.name for node in seqtree.node(accession).path]


def test_seqtree_merge_different_roots():
    seqtree = SeqTree()
    other = SeqTree(SoftmaxNode("other-root"))
    with pytest.raises(ValueError, match="different root names"):
        seqtree.merge(other)


def test_merge_command(tmp_path):
    seqtree = SeqTree()
    bacteria = SoftmaxNode("Bacteria", parent=seqtree.classification_tree)
    seqtree.add("existing", bacteria, 0)
    seqtree.add("duplicate", bacteria, 0)

    other = SeqTree()
    other_bacteria = SoftmaxNode("Bacteria", parent=other.classification_tree)
    virus = SoftmaxNode("Virus", parent=other.classification_tree)
    other.add("matching", other_bacteria, 1)
    other.add("new", virus, 2)
    other.add("duplicate", virus, 3)

    seqtree_path = tmp_path / "self.st"
    other_path = tmp_path / "other.st"
    output = tmp_path / "output" / "merged.st"
    seqtree.save(seqtree_path)
    other.save(other_path)
    input_bytes = [path.read_bytes() for path in (seqtree_path, other_path)]

    result = CliRunner().invoke(app, ["merge", str(seqtree_path), str(other_path), str(output)])

    assert result.exit_code == 0, (result.output, result.exception)
    merged = SeqTree.load(output)
    assert set(merged) == {"existing", "matching", "new", "duplicate"}
    assert merged.node("matching") is merged.node("existing")
    assert merged.node("new").name == "Virus"
    assert merged.node("duplicate") is merged.node("new")
    assert [merged[key].partition for key in
            ("existing", "matching", "new", "duplicate")] == [0, 1, 2, 3]
    assert merged.classification_tree.render_equal("""
        root
        ├── Bacteria
        └── Virus
    """)
    assert [path.read_bytes() for path in (seqtree_path, other_path)] == input_bytes
