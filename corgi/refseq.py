from multitax import NcbiTx
from hierarchicalsoftmax import SoftmaxNode
from pathlib import Path
import requests
from typing import List
from enum import Enum
from appdirs import user_cache_dir
from torchapp.download import cached_download
from rich.progress import track
from seqbank import SeqBank
import gzip

import typer


from .seqtree import SeqTree

app = typer.Typer()

# https://academic.oup.com/nar/article/44/D1/D733/2502674
DNA_PREFIXES = {"NC", "AC", "NZ", "NT", "NW", "NG"}

RANKS = ("superkingdom", "kingdom", "phylum", "class", "order")


class DNAType(Enum):
    NUCLEAR = "Nuclear"
    MITOCHONDRION = "Mitochondrion"
    PLASTID = "Plastid"
    PLASMID = "Plasmid"

    def __str__(self):
        return str(self.value)


def my_open(file):
    file = Path(file)
    if file.suffix.lower() == ".gz":
        return gzip.open(file, 'rb')
    return open(file, "r")


def line_count(file:Path):
    """ Adapted from https://stackoverflow.com/a/9631635 """
    def blocks(stream, size=65536):
        while True:
            b = stream.read(size)
            if not b: break
            yield b

    with my_open(file) as f:
        if isinstance(f, gzip.GzipFile):
            count = (sum(bl.count(b"\n") for bl in track(blocks(f), description="Counting lines:")))
        else:
            count = (sum(bl.count("\n") for bl in track(blocks(f), description="Counting lines:")))
    
    return count


def get_refseq_release() -> int:
    url = "https://ftp.ncbi.nlm.nih.gov/refseq/release/RELEASE_NUMBER"
    response = requests.get(url)
    data = response.text.strip()
    return int(data)


def get_catalogue() -> Path:
    version = get_refseq_release()
    filename = f"RefSeq-release{version}.catalog.gz"
    url = f"https://ftp.ncbi.nlm.nih.gov/refseq/release/release-catalog/{filename}"

    local_path = Path(user_cache_dir("torchapps"))/"Corgi"/filename
    cached_download(url, local_path=local_path)
    return local_path


def get_node_at_rank(taxon, taxonomy, taxon_to_node, current_node, rank):
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
    current_node = root
    for rank in ranks:
        current_node = get_node_at_rank(taxon, taxonomy, taxon_to_node, current_node, rank)

    return current_node


@app.command()
def refseq_to_seqtree(
    output:Path, 
    seqbank,
    seqbank_output:Path=None,
    render:Path=None, 
    accessions:Path=None,
    partitions:int=6, 
    max_seqs:int=100, 
    catalog:Path=None,
    ranks:str="superkingdom,kingdom,phylum,class,order",
    restrict_ranks:str="class,order",
    print_tree:bool=False,
):
    catalog = catalog or get_catalogue()
    print(f"Using catalog: {catalog}")
    count = line_count(catalog)
    print(f"{count} lines in catalog")

    seqbank = SeqBank(seqbank)
    seqbank_accessions = seqbank.get_accessions()
    
    print(f'Saving seqbank with accessions corresponding to {seqbank_output}')
    seqbank_output = SeqBank(seqbank_output, write=True)

    root = SoftmaxNode("root", rank="root", nseq=0)
    seqtree = SeqTree(root)

    # Setup Nodes for each DNAType
    type_nodes = {}
    type_taxon_to_node = {}
    for type_obj in DNAType:
        type_str = str(type_obj)
        type_nodes[type_str] = SoftmaxNode(type_str, parent=root, rank="type", nseq=0)
        type_taxon_to_node[type_str] = {}

    if isinstance(ranks, str):
        ranks = ranks.split(",")

    if isinstance(restrict_ranks, str):
        restrict_ranks = set(restrict_ranks.split(","))

    print('Loading taxonomy...')
    taxonomy = NcbiTx()

    with my_open(catalog) as f:
        for line in track(f, description="Processing RefSeq: ", total=count):
            if isinstance(line, bytes):
                line = line.decode()
            components = line.split("\t")
            accession = components[2]
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
            node = get_node(taxon, taxonomy, type_taxon_to_node[type_str], type_nodes[type_str], ranks=ranks)

            # Skip if not in one of the allowed ranks (e.g. class, order)
            if node.rank not in restrict_ranks:
                continue

            # Skip if we have too many sequences for this leaf node
            if max_seqs and node.nseq >= max_seqs:
                continue

            if accession not in seqbank_accessions:
                continue

            data = seqbank[accession]            
            seqbank_output.add(data, accession)

            partition = node.nseq % partitions

            node.nseq += 1

            seqtree.add(accession, node, partition)

    print(f'Saving seqtree with {len(seqtree)} sequences to {output}')
    seqtree.save(output)

    if render:
        print(f'Rendering seqtree to {render}')

    for node in root.post_order_iter():
        node.render_str = f"{node.name} ({node.nseq})" if node.nseq else node.name

    seqtree.render(filepath=render, attr="render_str", print=print_tree)

    if accessions:
        print(f"Writing accessions to file {accessions}")
        seqtree.accessions_to_file(accessions)


if __name__ == "__main__":
    app()
