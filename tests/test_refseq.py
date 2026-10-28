import gzip
from pathlib import Path
from unittest.mock import patch

import pytest
from hierarchicalsoftmax import SoftmaxNode
from seqbank import SeqBank

from corgi import refseq
from corgi.refseq import DNAType, get_node, get_node_at_rank, line_count, my_open, refseq_to_seqtree
from corgi.seqtree import SeqTree

TEST_DATA = Path(__file__).parent / "testdata"


class FakeTaxonomy:
    """A stand-in for multitax.NcbiTx with a lookup of rank -> taxon ID for each taxon."""

    lineages = {
        "t1": {"superkingdom": "2", "phylum": "p1", "class": "c1", "order": "o1"},
        "t2": {"superkingdom": "2", "phylum": "p1", "class": "c1", "order": "o2"},
        "t3": {"superkingdom": "2759", "phylum": "p2", "class": "c2", "order": "o4"},
        "t6": {"superkingdom": "2759", "phylum": "p2"},
        "t4": {"superkingdom": "2759", "phylum": "p2", "class": "c2", "order": "o3"},
        "t5": {"superkingdom": "2", "phylum": "p1", "class": "c3"},
    }
    names = {"2": "Bacteria", "2759": "Eukaryota", "p1": "Phylum1", "p2": "Phylum2"}
    names.update(
        {"c1": "Class1", "c2": "Class2", "o1": "Order1", "o2": "Order2", "o3": "Order3", "o4": "Order4", "c3": "Class3"}
    )

    def parent_rank(self, taxon, rank):
        return self.lineages[taxon].get(rank)

    def name(self, taxon_id):
        return self.names[taxon_id]


def test_dna_type_str():
    assert [str(t) for t in DNAType] == ["Nuclear", "Mitochondrion", "Plastid", "Plasmid"]


@pytest.mark.parametrize(
    "description,expected",
    [
        ("Homo sapiens mitochondrion, complete genome", DNAType.MITOCHONDRION),
        ("Zea mays chloroplast plastid", DNAType.PLASTID),
        ("Escherichia coli plasmid pXYZ", DNAType.PLASMID),
        ("Escherichia coli chromosome", DNAType.NUCLEAR),
        ("MITOCHONDRION", DNAType.MITOCHONDRION),
    ],
)
def test_dna_type_from_description(description, expected):
    assert DNAType.from_description(description) == expected


def test_my_open_plain_and_gzip(tmp_path):
    plain = tmp_path / "a.txt"
    plain.write_text("hello\n")
    with my_open(plain) as f:
        assert f.read() == "hello\n"

    zipped = tmp_path / "a.txt.GZ"
    with gzip.open(zipped, "wt") as f:
        f.write("hello\n")
    with my_open(zipped) as f:
        assert f.read() == b"hello\n"


def test_line_count(tmp_path):
    plain = tmp_path / "a.txt"
    plain.write_text("a\nb\nc\n")
    assert line_count(plain) == 3

    zipped = tmp_path / "a.txt.gz"
    with gzip.open(zipped, "wt") as f:
        f.write("a\nb\n")
    assert line_count(zipped) == 2


def test_get_refseq_release():
    response = type("Response", (), {"text": "225\n"})()
    with patch("corgi.refseq.requests.get", return_value=response) as get:
        assert refseq.get_refseq_release() == 225
    assert get.call_args.args[0].endswith("RELEASE_NUMBER")


def test_get_catalogue(tmp_path):
    with (
        patch("corgi.refseq.get_refseq_release", return_value=225),
        patch("corgi.refseq.cached_download") as download,
        patch("corgi.refseq.user_cache_dir", return_value=str(tmp_path)),
    ):
        path = refseq.get_catalogue()
    assert path == tmp_path / "Corgi" / "RefSeq-release225.catalog.gz"
    assert download.call_args.args[0].endswith("RefSeq-release225.catalog.gz")
    assert download.call_args.kwargs["local_path"] == path


def test_get_node_at_rank():
    taxonomy = FakeTaxonomy()
    root = SoftmaxNode("root")
    taxon_to_node = {}

    node = get_node_at_rank("t1", taxonomy, taxon_to_node, root, "superkingdom")
    assert node.name == "Bacteria"
    assert node.parent is root
    assert node.rank == "superkingdom"
    assert taxon_to_node["2"] is node

    # Reuses an existing node
    assert get_node_at_rank("t2", taxonomy, taxon_to_node, root, "superkingdom") is node

    # Missing ranks leave the current node unchanged
    assert get_node_at_rank("t6", taxonomy, taxon_to_node, node, "class") is node


def test_get_node():
    taxonomy = FakeTaxonomy()
    root = SoftmaxNode("root")
    taxon_to_node = {}
    ranks = ["superkingdom", "phylum", "class", "order"]

    node = get_node("t1", taxonomy, taxon_to_node, root, ranks)
    assert [n.name for n in node.ancestors[1:]] == ["Bacteria", "Phylum1", "Class1"]
    assert node.name == "Order1"
    assert node.rank == "order"

    # Shares the ancestors with a sibling
    sibling = get_node("t2", taxonomy, taxon_to_node, root, ranks)
    assert sibling.parent is node.parent

    # Stops at the last available rank
    shallow = get_node("t6", taxonomy, taxon_to_node, root, ranks)
    assert shallow.name == "Phylum2"


CATALOG = "\n".join(
    [
        "t1\tBacterium one\tNC_010663.1\tbacteria.1.genomic.gbff.gz\tna",
        "t2\tBacterium two\tNC_024664.1\tbacteria.2.genomic.gbff.gz\tna",
        "t3\tPlant\tNC_036112.1\tplastid.1.genomic.gbff.gz\tna",
        "t4\tFungus\tNC_036113.1\tmitochondrion.1.genomic.gbff.gz\tna",
        "t5\tFungus\tNC_044840.1\tplasmid.1.genomic.gbff.gz\tna",
        "t1\tProtein\tNP_000001.1\tbacteria.1.protein.faa.gz\tna",  # not DNA
        "t1\tNot in the seqbank\tNC_999999.1\tbacteria.3.genomic.gbff.gz\tna",
        "t1\tNot DNA prefix\tXX_010663.1\tbacteria.3.genomic.gbff.gz\tna",
        "",
    ]
)


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / "catalog.txt"
    path.write_text(CATALOG)
    return path


@pytest.fixture
def input_seqbank():
    seqbank = SeqBank(TEST_DATA / "seqbank.rocksdict.sb", engine="rocksdict")
    yield TEST_DATA / "seqbank.rocksdict.sb"
    seqbank.close()


def run_refseq_to_seqtree(tmp_path, catalog, input_seqbank, **kwargs):
    output = tmp_path / "out.st"
    with patch("corgi.refseq.NcbiTx", return_value=FakeTaxonomy()):
        refseq_to_seqtree(output=output, seqbank=input_seqbank, catalog=catalog, **kwargs)
    return SeqTree.load(output)


def test_refseq_to_seqtree(tmp_path, catalog, input_seqbank):
    accessions = tmp_path / "accessions.txt"
    render = tmp_path / "tree.txt"
    seqtree = run_refseq_to_seqtree(
        tmp_path,
        catalog,
        input_seqbank,
        accessions=accessions,
        render=render,
        print_tree=True,
        partitions=2,
    )
    # Only DNA accessions in the SeqBank are included
    assert set(seqtree.keys()) == {"NC_010663.1", "NC_024664.1", "NC_036112.1", "NC_036113.1", "NC_044840.1"}
    assert accessions.read_text().split() == list(seqtree.keys())
    assert "Nuclear" in render.read_text()

    # Types are determined by the file the accession is found in
    types = {accession: seqtree.node(accession).ancestors[1].name for accession in seqtree}
    assert types == {
        "NC_010663.1": "Nuclear",
        "NC_024664.1": "Nuclear",
        "NC_036112.1": "Plastid",
        "NC_036113.1": "Mitochondrion",
        "NC_044840.1": "Plasmid",
    }
    # Sequences are assigned to the order and partitioned in turn
    assert seqtree.node("NC_010663.1").rank == "order"
    assert {seqtree["NC_010663.1"].partition, seqtree["NC_024664.1"].partition} == {0}


def test_refseq_to_seqtree_partitions_in_turn(tmp_path, input_seqbank):
    catalog = tmp_path / "catalog.txt"
    catalog.write_text(
        "t1\tA\tNC_010663.1\tbacteria.genomic.gbff.gz\tna\n"
        "t1\tB\tNC_024664.1\tbacteria.genomic.gbff.gz\tna\n"
        "t1\tC\tNC_036112.1\tbacteria.genomic.gbff.gz\tna\n"
    )
    seqtree = run_refseq_to_seqtree(tmp_path, catalog, input_seqbank, partitions=2, restrict_ranks="order")
    assert [seqtree[a].partition for a in ["NC_010663.1", "NC_024664.1", "NC_036112.1"]] == [0, 1, 0]


def test_refseq_to_seqtree_max_seqs(tmp_path, catalog, input_seqbank):
    seqtree = run_refseq_to_seqtree(tmp_path, catalog, input_seqbank, max_seqs=1)
    # Every sequence in the catalog is in a different leaf node
    assert len(seqtree) == 5

    single_catalog = tmp_path / "single.txt"
    single_catalog.write_text(
        "t1\tA\tNC_010663.1\tbacteria.genomic.gbff.gz\tna\n" "t1\tB\tNC_024664.1\tbacteria.genomic.gbff.gz\tna\n"
    )
    seqtree = run_refseq_to_seqtree(tmp_path, single_catalog, input_seqbank, max_seqs=1)
    assert list(seqtree.keys()) == ["NC_010663.1"]


def test_refseq_to_seqtree_restrict_ranks(tmp_path, catalog, input_seqbank):
    seqtree = run_refseq_to_seqtree(tmp_path, catalog, input_seqbank, restrict_ranks="class")
    # Only sequences whose deepest known rank is a class are kept (here, the plasmid with no order)
    assert list(seqtree.keys()) == ["NC_044840.1"]
    assert seqtree.node("NC_044840.1").rank == "class"


def test_refseq_to_seqtree_rank_overrides(tmp_path, catalog, input_seqbank):
    seqtree = run_refseq_to_seqtree(
        tmp_path,
        catalog,
        input_seqbank,
        plastid_ranks="superkingdom",
        plastid_restrict_ranks="superkingdom",
        mitochondrion_ranks="phylum",
        mitochondrion_restrict_ranks="phylum",
        restrict_ranks="class,order",
    )
    assert seqtree.node("NC_036112.1").rank == "superkingdom"
    assert seqtree.node("NC_036113.1").rank == "phylum"
    assert seqtree.node("NC_010663.1").rank == "order"


@pytest.mark.parametrize("engine", ["rocksdict", "lmdb"])
def test_refseq_to_seqtree_seqbank_output(tmp_path, catalog, input_seqbank, engine):
    seqbank_output = tmp_path / f"output.{engine}.sb"
    seqtree = run_refseq_to_seqtree(tmp_path, catalog, input_seqbank, seqbank_output=seqbank_output, engine=engine)
    output = SeqBank(seqbank_output)
    try:
        assert output.get_accessions() == set(seqtree.keys())
        assert output.engine == engine
    finally:
        output.close()


def test_refseq_to_seqtree_gzip_catalog_and_non_dna_prefix(tmp_path):
    seqbank_path = tmp_path / "input.sb"
    seqbank = SeqBank(seqbank_path, write=True, engine="rocksdict")
    seqbank.add("ACGTACGTAC", "NC_000001.1")
    seqbank.add("ACGTACGTAC", "NP_000001.1")  # In the SeqBank but not a DNA accession
    seqbank.close()

    catalog = tmp_path / "catalog.txt.gz"
    with gzip.open(catalog, "wt") as f:
        f.write("t1\tA\tNC_000001.1\tbacteria.genomic.gbff.gz\tna\n")
        f.write("t1\tB\tNP_000001.1\tbacteria.protein.faa.gz\tna\n")

    seqtree = run_refseq_to_seqtree(tmp_path, catalog, seqbank_path)
    assert list(seqtree.keys()) == ["NC_000001.1"]
