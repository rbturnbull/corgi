"""Tools for building a SeqTree from the RefSeq catalog and taxonomy."""

from multitax import NcbiTx
from hierarchicalsoftmax import SoftmaxNode
from pathlib import Path
import requests
from enum import Enum
from appdirs import user_cache_dir
from torchapp.download import cached_download
from rich.progress import track
from seqbank import SeqBank
import gzip


from .seqtree import SeqTree

# https://academic.oup.com/nar/article/44/D1/D733/2502674
DNA_PREFIXES = {"NC", "AC", "NZ", "NT", "NW", "NG"}

RANKS = ("superkingdom", "kingdom", "phylum", "class", "order")


class DNAType(Enum):
    """The types of DNA sequence that are classified."""

    NUCLEAR = "Nuclear"
    MITOCHONDRION = "Mitochondrion"
    PLASTID = "Plastid"
    PLASMID = "Plasmid"

    def __str__(self):
        """Returns the name of the type."""
        return str(self.value)

    @classmethod
    def from_description(cls, description: str):
        """Convert a description to a DNAType."""
        description = description.lower()
        if "mitochondrion" in description:
            return cls.MITOCHONDRION
        elif "plastid" in description:
            return cls.PLASTID
        elif "plasmid" in description:
            return cls.PLASMID
        else:
            return cls.NUCLEAR


def my_open(file):
    """Opens a text file for reading, or a gzipped file in binary mode."""
    file = Path(file)
    if file.suffix.lower() == ".gz":
        return gzip.open(file, 'rb')
    return open(file, "r")


def line_count(file: Path):
    """Adapted from https://stackoverflow.com/a/9631635"""

    def blocks(stream, size=65536):
        """Yields blocks of the stream until it is exhausted."""
        while True:
            b = stream.read(size)
            if not b:
                break
            yield b

    with my_open(file) as f:
        if isinstance(f, gzip.GzipFile):
            count = sum(bl.count(b"\n") for bl in track(blocks(f), description="Counting lines:"))
        else:
            count = sum(bl.count("\n") for bl in track(blocks(f), description="Counting lines:"))

    return count


def get_refseq_release() -> int:
    """Returns the number of the current RefSeq release."""
    url = "https://ftp.ncbi.nlm.nih.gov/refseq/release/RELEASE_NUMBER"
    response = requests.get(url)
    data = response.text.strip()
    return int(data)


def get_catalogue() -> Path:
    """Downloads (or reuses from the cache) the catalog for the current RefSeq release and returns its path."""
    version = get_refseq_release()
    filename = f"RefSeq-release{version}.catalog.gz"
    url = f"https://ftp.ncbi.nlm.nih.gov/refseq/release/release-catalog/{filename}"

    local_path = Path(user_cache_dir("torchapps")) / "Corgi" / filename
    cached_download(url, local_path=local_path)
    return local_path


def get_node_at_rank(taxon, taxonomy, taxon_to_node, current_node, rank):
    """Returns the node for the taxon's ancestor at a rank, creating it under `current_node` if needed.

    If the taxon has no ancestor at the rank, then `current_node` is returned.

    Args:
        taxon: The taxon ID.
        taxonomy: The taxonomy (e.g. multitax.NcbiTx).
        taxon_to_node (dict): The nodes already created, by taxon ID.
        current_node (SoftmaxNode): The parent node for any new node.
        rank (str): The taxonomic rank.
    """
    current_id = taxonomy.parent_rank(taxon, rank)
    if not current_id:
        return current_node

    if current_id in taxon_to_node:
        return taxon_to_node[current_id]

    name = taxonomy.name(current_id)
    current_node = SoftmaxNode(name, parent=current_node, id=current_id, rank=rank, nseq=0)
    taxon_to_node[current_id] = current_node
    return current_node


def get_node(taxon, taxonomy, taxon_to_node, root, ranks):
    """Returns the node in the tree for a taxon, creating nodes for its ancestors at the given ranks as needed.

    Args:
        taxon: The taxon ID.
        taxonomy: The taxonomy (e.g. multitax.NcbiTx).
        taxon_to_node (dict): The nodes already created, by taxon ID.
        root (SoftmaxNode): The node to start from.
        ranks (list[str]): The ranks to include, from the highest to the lowest.
    """
    current_node = root
    for rank in ranks:
        current_node = get_node_at_rank(taxon, taxonomy, taxon_to_node, current_node, rank)

    return current_node


def refseq_to_seqtree(
    output: Path,
    seqbank: Path,
    seqbank_output: Path | None = None,
    engine: str | None = None,
    render: Path | None = None,
    accessions: Path | None = None,
    partitions: int = 6,
    max_seqs: int = 100,
    catalog: Path | None = None,
    ranks: str = "superkingdom,kingdom,phylum,class,order",
    restrict_ranks: str = "class,order",
    mitochondrion_ranks: str = "",
    plastid_ranks: str = "",
    plasmid_ranks: str = "",
    nuclear_ranks: str = "",
    mitochondrion_restrict_ranks: str = "",
    plastid_restrict_ranks: str = "",
    plasmid_restrict_ranks: str = "",
    nuclear_restrict_ranks: str = "",
    print_tree: bool = False,
):
    """Builds a SeqTree from the RefSeq catalog for the accessions found in a SeqBank.

    Each accession is placed in the taxonomic tree and assigned to a cross-validation partition.
    This is available on the command line as `corgi-tools refseq-to-seqtree`.

    Args:
        output (Path): The path to save the SeqTree.
        seqbank (Path): The SeqBank with the sequences for the RefSeq accessions.
        seqbank_output (Path | None): Where to save a new SeqBank containing only the sequences selected for
            the SeqTree.
        engine (str | None): The storage engine for the output SeqBank: 'rocksdict', 'lmdb' or 'speedict'.
            If not given, then RocksDict is used.
        render (Path | None): The path to render an image of the SeqTree.
        accessions (Path | None): The path to write a list of the accessions in the SeqTree.
        partitions (int): The number of cross-validation partitions.
        max_seqs (int): The maximum number of sequences for each leaf node. Use 0 for no limit.
        catalog (Path | None): The path to a RefSeq catalog. If not given, then the latest release is downloaded.
        ranks (str): The comma-separated taxonomic ranks to include in the tree.
        restrict_ranks (str): The comma-separated ranks of the nodes which sequences can be assigned to.
        mitochondrion_ranks (str): Overrides `ranks` for mitochondrion sequences.
        plastid_ranks (str): Overrides `ranks` for plastid sequences.
        plasmid_ranks (str): Overrides `ranks` for plasmid sequences.
        nuclear_ranks (str): Overrides `ranks` for nuclear sequences.
        mitochondrion_restrict_ranks (str): Overrides `restrict_ranks` for mitochondrion sequences.
        plastid_restrict_ranks (str): Overrides `restrict_ranks` for plastid sequences.
        plasmid_restrict_ranks (str): Overrides `restrict_ranks` for plasmid sequences.
        nuclear_restrict_ranks (str): Overrides `restrict_ranks` for nuclear sequences.
        print_tree (bool): Whether or not to print the tree to the screen.
    """
    catalog = catalog or get_catalogue()
    print(f"Using catalog: {catalog}")
    count = line_count(catalog)
    print(f"{count} lines in catalog")

    seqbank = SeqBank(seqbank)
    seqbank_accessions = seqbank.get_accessions()

    if seqbank_output is not None:
        print(f"Saving seqbank with accessions corresponding to {seqbank_output}")
        seqbank_output = SeqBank(seqbank_output, write=True, engine=engine)

    root = SoftmaxNode("root", rank="root", nseq=0)
    seqtree = SeqTree(root)

    # Setup Nodes for each DNAType
    type_nodes = {}
    type_taxon_to_node = {}
    for type_obj in DNAType:
        type_str = str(type_obj)
        type_nodes[type_str] = SoftmaxNode(type_str, parent=root, rank="type", nseq=0)
        type_taxon_to_node[type_str] = {}

    rank_strings = dict(
        mitochondrion=mitochondrion_ranks or ranks,
        plastid=plastid_ranks or ranks,
        plasmid=plasmid_ranks or ranks,
        nuclear=nuclear_ranks or ranks,
    )

    restrict_rank_strings = dict(
        mitochondrion=mitochondrion_restrict_ranks or restrict_ranks,
        plastid=plastid_restrict_ranks or restrict_ranks,
        plasmid=plasmid_restrict_ranks or restrict_ranks,
        nuclear=nuclear_restrict_ranks or restrict_ranks,
    )

    ranks_by_type = {k.title(): v.split(",") for k, v in rank_strings.items()}
    restrict_ranks_by_type = {k.title(): v.split(",") for k, v in restrict_rank_strings.items()}

    print('Loading taxonomy...')
    taxonomy = NcbiTx()

    with my_open(catalog) as f:
        for line in track(f, description="Processing RefSeq: ", total=count):
            if isinstance(line, bytes):
                line = line.decode()
            components = line.split("\t")
            accession = components[2]

            if accession not in seqbank_accessions:
                continue

            prefix = accession[:2]

            # Exclude non-DNA sequences
            if prefix not in DNA_PREFIXES:
                continue

            taxon = components[0]
            files = components[3]
            # status = components[4]

            # Get the type of DNA by seeing what files it is in
            if "mitochondrion" in files:
                type_obj = DNAType.MITOCHONDRION
            elif "plastid" in files:
                type_obj = DNAType.PLASTID
            elif "plasmid" in files:
                type_obj = DNAType.PLASMID
            else:
                type_obj = DNAType.NUCLEAR
            type_str = str(type_obj)

            # Get the node for this taxon
            node = get_node(
                taxon, taxonomy, type_taxon_to_node[type_str], type_nodes[type_str], ranks=ranks_by_type[type_str]
            )

            # Skip if not in one of the allowed ranks (e.g. class, order)
            if node.rank not in restrict_ranks_by_type[type_str]:
                continue

            # Skip if we have too many sequences for this leaf node
            if max_seqs and node.nseq >= max_seqs:
                continue

            if seqbank_output is not None:
                seqbank_output.add(seqbank[accession], accession)

            partition = node.nseq % partitions

            node.nseq += 1

            seqtree.add(accession, node, partition)

    print(f'Saving seqtree with {len(seqtree)} sequences to {output}')
    seqtree.save(output)

    if render:
        print(f"Rendering seqtree to {render}")

    for node in root.post_order_iter():
        node.render_str = f"{node.name} ({node.nseq})" if node.nseq else node.name

    seqtree.render(filepath=render, attr="render_str", print=print_tree)

    if accessions:
        print(f"Writing accessions to file {accessions}")
        seqtree.accessions_to_file(accessions)

    if seqbank_output is not None:
        seqbank_output.close()
    seqbank.close()
