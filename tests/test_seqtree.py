
import tempfile
import pytest
from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from typer.testing import CliRunner
from unittest.mock import patch
from Bio import SeqIO
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


def test_seqtree_render_alphabetical():
    root = SoftmaxNode("root")
    SoftmaxNode("b", parent=root)
    SoftmaxNode("a", parent=root)
    seqtree = SeqTree(root)

    unsorted_lines = str(seqtree.render()).splitlines()
    assert [line.split()[-1] for line in unsorted_lines[1:]] == ["b", "a"]

    sorted_lines = str(seqtree.render(alphabetical=True)).splitlines()
    assert [line.split()[-1] for line in sorted_lines[1:]] == ["a", "b"]


def make_small_seqtree() -> SeqTree:
    root = SoftmaxNode("root")
    animal = SoftmaxNode("Animalia", parent=root)
    plant = SoftmaxNode("Plantae", parent=root)
    seqtree = SeqTree(root)
    seqtree.add("NC_1", animal, 0)
    seqtree.add("NC_2", plant, 1)
    return seqtree


def test_sunburst_command_html(tmp_path):
    seqtree_path = tmp_path / "tree.st"
    make_small_seqtree().save(seqtree_path)
    output = tmp_path / "subdir" / "sunburst.html"

    result = CliRunner().invoke(app, ["sunburst", str(seqtree_path), "--output", str(output)])

    assert result.exit_code == 0, result.output
    html = output.read_text()
    assert "Animalia" in html
    assert "Plantae" in html


def test_sunburst_command_image(tmp_path):
    pytest.importorskip("kaleido")
    seqtree_path = tmp_path / "tree.st"
    make_small_seqtree().save(seqtree_path)
    output = tmp_path / "sunburst.svg"

    result = CliRunner().invoke(app, ["sunburst", str(seqtree_path), "--output", str(output)])

    if result.exception is not None and "Chrome" in str(result.exception):
        pytest.skip("Chrome is not available for Kaleido image export")
    assert result.exit_code == 0, result.output
    assert output.exists()


import pickle
from corgi.seqtree import AlreadyExists, SeqDetail, node_to_str
from seqbank import SeqBank

TEST_DATA = Path(__file__).parent / "testdata"
runner = CliRunner()


@pytest.fixture
def seqtree_path():
    return TEST_DATA / "seqtree.pkl"


@pytest.fixture
def seqbank():
    seqbank = SeqBank(TEST_DATA / "seqbank.rocksdict.sb", engine="rocksdict")
    yield seqbank
    seqbank.close()


def invoke(*args):
    result = runner.invoke(app, [str(arg) for arg in args])
    assert result.exit_code == 0, result.output
    return result


def test_node_to_str():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    b = SoftmaxNode("b", parent=a)
    c = SoftmaxNode("c", parent=b)
    assert node_to_str(root) == "root"
    assert node_to_str(a) == "a"
    assert node_to_str(b) == "a/b"
    assert node_to_str(c) == "a/c"


def test_seq_detail_pickle():
    detail = SeqDetail(partition=3, node=SoftmaxNode("x"), node_id=7)
    clone = pickle.loads(pickle.dumps(detail))
    assert clone.partition == 3
    assert clone.node_id == 7
    assert clone.node is None


def test_add_already_exists():
    seqtree = SeqTree()
    a = SoftmaxNode("a", parent=seqtree.classification_tree)
    b = SoftmaxNode("b", parent=seqtree.classification_tree)
    seqtree.add("acc", a, 0)
    seqtree.add("acc", a, 1)  # Same node is allowed
    assert seqtree["acc"].partition == 1
    with pytest.raises(AlreadyExists):
        seqtree.add("acc", b, 0)


def test_accessions_and_partitions(seqtree_path):
    seqtree = SeqTree.load(seqtree_path)
    assert set(seqtree.accessions()) == set(seqtree.keys())
    assert set(seqtree.accessions(partition=2)) == {"NC_036112.1", "NC_036113.1"}
    assert set(seqtree.accessions_in_partition(0)) == {"NZ_JAJNFP010000161.1"}

    seqtree.set_partition("NC_03611*", 5)
    assert set(seqtree.accessions(partition=5)) == {"NC_036112.1", "NC_036113.1"}
    assert seqtree["NC_024664.1"].partition == 1


def test_node_resolves_from_node_id(seqtree_path):
    seqtree = SeqTree.load(seqtree_path)
    assert seqtree["NC_036112.1"].node is None
    assert seqtree.node("NC_036112.1").name == "Plant"


def test_counts(seqtree_path):
    seqtree = SeqTree.load(seqtree_path)
    seqtree.add_counts()
    counts = {node.name: node.count for node in seqtree.classification_tree.pre_order_iter()}
    assert counts == {"root": 0, "Bacteria": 1, "Eukaryota": 1, "Virus": 1, "Plant": 2}

    seqtree.add_partition_counts()
    plant = next(n for n in seqtree.classification_tree.pre_order_iter() if n.name == "Plant")
    assert dict(plant.partition_counts) == {2: 2}


def test_render_options(seqtree_path):
    seqtree = SeqTree.load(seqtree_path)
    assert "Plant (2)" in str(seqtree.render(count=True))
    assert "Plant 2->2" in str(seqtree.render(partition_counts=True))


def test_pickle_tree(seqtree_path, tmp_path):
    seqtree = SeqTree.load(seqtree_path)
    output = tmp_path / "tree.pkl"
    seqtree.pickle_tree(output)
    with open(output, "rb") as f:
        tree = pickle.load(f)
    assert tree.name == "root"


def test_prune():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    b = SoftmaxNode("b", parent=a)
    c = SoftmaxNode("c", parent=b)
    seqtree = SeqTree(root)
    seqtree.add("deep", c, 1)
    seqtree.add("shallow", a, 2)

    pruned = seqtree.prune(max_depth=2)
    # The root counts as the first level so nothing deeper than the first level below it is kept
    assert pruned.node("deep").name == "a"
    assert pruned.node("shallow").name == "a"
    assert pruned["deep"].partition == 1
    assert [n.name for n in pruned.classification_tree.pre_order_iter()] == ["root", "a"]


def test_intersection_seqbank(seqtree_path, seqbank):
    seqtree = SeqTree.load(seqtree_path)
    seqtree.add("NOT_IN_SEQBANK", seqtree.node("NC_036112.1"), 0)
    missing = seqtree.intersection_seqbank(seqbank)
    assert missing == {"NOT_IN_SEQBANK"}
    assert "NOT_IN_SEQBANK" not in seqtree


def test_export(seqtree_path, seqbank, tmp_path):
    seqtree = SeqTree.load(seqtree_path)
    output = tmp_path / "all.fasta"
    seqtree.export(seqbank, output)
    records = list(SeqIO.parse(output, "fasta"))
    assert len(records) == 5
    assert {r.id for r in records} == {
        "NZ_JAJNFP010000161.1#Bacteria",
        "NC_024664.1#Eukaryota",
        "NC_010663.1#Virus",
        "NC_036112.1#Plant",
        "NC_036113.1#Plant",
    }


def test_export_partition_and_length(seqtree_path, seqbank, tmp_path):
    seqtree = SeqTree.load(seqtree_path)
    output = tmp_path / "part.fa"
    seqtree.export(seqbank, output, length=100, seed=1, partition=2)
    records = list(SeqIO.parse(output, "fasta"))
    assert {r.id for r in records} == {"NC_036112.1#Plant", "NC_036113.1#Plant"}
    assert all(len(r.seq) == 100 for r in records)

    # The slice is reproducible given the seed
    again = tmp_path / "again.fa"
    seqtree.export(seqbank, again, length=100, seed=1, partition=2)
    assert [str(r.seq) for r in SeqIO.parse(again, "fasta")] == [str(r.seq) for r in records]


def test_export_length_longer_than_sequence(seqtree_path, seqbank, tmp_path):
    seqtree = SeqTree.load(seqtree_path)
    output = tmp_path / "long.fasta"
    seqtree.export(seqbank, output, length=10_000_000, partition=2)
    assert all(len(r.seq) == len(seqbank[r.id.split("#")[0]]) for r in SeqIO.parse(output, "fasta"))


def test_sunburst_figure(seqtree_path):
    fig = SeqTree.load(seqtree_path).sunburst(width=300)
    assert set(fig.data[0].labels) == {"root", "Bacteria", "Eukaryota", "Virus", "Plant"}
    assert fig.layout.width == 300


def test_accessions_to_file(seqtree_path, tmp_path):
    output = tmp_path / "accessions.txt"
    SeqTree.load(seqtree_path).accessions_to_file(output)
    assert set(output.read_text().split()) == set(SeqTree.load(seqtree_path).keys())


def test_cli_intersection_seqbank(seqtree_path, tmp_path):
    seqtree = SeqTree.load(seqtree_path)
    seqtree.add("NOT_IN_SEQBANK", seqtree.node("NC_036112.1"), 0)
    modified = tmp_path / "modified.st"
    seqtree.save(modified)
    output = tmp_path / "out.st"
    invoke("intersection-seqbank", modified, TEST_DATA / "seqbank.rocksdict.sb", output)
    result = SeqTree.load(output)
    assert "NOT_IN_SEQBANK" not in result
    assert len(result) == 5


def test_cli_accessions(seqtree_path):
    result = invoke("accessions", seqtree_path)
    assert set(result.output.split()) == set(SeqTree.load(seqtree_path).keys())

    result = invoke("accessions", seqtree_path, "--partition", 2)
    assert set(result.output.split()) == {"NC_036112.1", "NC_036113.1"}


def test_cli_export(seqtree_path, tmp_path):
    output = tmp_path / "export.fasta"
    invoke(
        "export", seqtree_path, TEST_DATA / "seqbank.rocksdict.sb", output, "--partition", 2, "--length", 50, "--seed", 3
    )
    assert len(list(SeqIO.parse(output, "fasta"))) == 2


def test_cli_render(seqtree_path, tmp_path):
    assert "Plant" in invoke("render", seqtree_path).output
    assert "Plant (2)" in invoke("render", seqtree_path, "--count").output
    assert "Plant 2->2" in invoke("render", seqtree_path, "--partition-counts").output

    output = tmp_path / "tree.txt"
    result = invoke("render", seqtree_path, "--no-print", "--output", output, "--alphabetical")
    assert "Plant" not in result.output
    text = output.read_text()
    assert text.index("Bacteria") < text.index("Eukaryota") < text.index("Plant") < text.index("Virus")


def test_cli_count(seqtree_path):
    assert invoke("count", seqtree_path).output.strip() == "5"


def test_cli_sunburst_show_and_size(seqtree_path, tmp_path):
    output = tmp_path / "plot.html"
    with patch("plotly.graph_objects.Figure.show") as show:
        invoke("sunburst", seqtree_path, "--show", "--output", output, "--width", 400, "--height", 0)
    show.assert_called_once()
    assert output.exists()


def test_cli_prune(seqtree_path, tmp_path):
    output = tmp_path / "pruned.st"
    invoke("prune", seqtree_path, 2, output)
    pruned = SeqTree.load(output)
    assert len(pruned) == 5
    assert pruned.node("NC_036112.1").name == "Plant"


def test_cli_layer_size(seqtree_path):
    expected = SeqTree.load(seqtree_path).classification_tree.layer_size
    assert invoke("layer-size", seqtree_path).output.strip() == str(expected)


def test_cli_pickle_tree(seqtree_path, tmp_path):
    output = tmp_path / "tree.pkl"
    invoke("pickle-tree", seqtree_path, output)
    assert output.exists()


def test_cli_set_partition(seqtree_path, tmp_path):
    output = tmp_path / "partitioned.st"
    invoke("set-partition", seqtree_path, "NC_*", 9, output)
    seqtree = SeqTree.load(output)
    assert set(seqtree.accessions(partition=9)) == {"NC_024664.1", "NC_010663.1", "NC_036112.1", "NC_036113.1"}
    assert seqtree["NZ_JAJNFP010000161.1"].partition == 0
