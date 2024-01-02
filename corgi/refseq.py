from multitax import NcbiTx
from hierarchicalsoftmax import SoftmaxNode
from pathlib import Path
import requests
from typing import List
from enum import Enum
from appdirs import user_cache_dir
from torchapp.download import cached_download
from .seqtree import SeqTree

import typer

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
        return self.value


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
    render:Path=None, 
    partitions:int=5, 
    max_seqs:int=100, 
    catalog:Path=None,
    ranks:str="superkingdom,kingdom,phylum,class,order",
    restrict_ranks:str="class,order",
):
    catalog = catalog or get_catalogue()
    print(f"Using catalog: {catalog}")

    root = SoftmaxNode("root", rank="root", nseq=0)
    seqtree = SeqTree(root)

    # Setup Nodes for each DNAType
    type_nodes = {}
    type_taxon_to_node = {}
    for type_obj in DNAType:
        type = str(type_obj)
        type_nodes[type] = SoftmaxNode(type, parent=root, rank="type", nseq=0)
        type_taxon_to_node[type] = {}

    if isinstance(ranks, str):
        ranks = ranks.split(",")

    if isinstance(restrict_ranks, str):
        restrict_ranks = set(restrict_ranks.split(","))

    taxonomy = NcbiTx()

    with open(catalog) as f:
        for line in f:
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
                type_str = DNAType.MITOCHONDRION
            elif "plastid" in files:
                type_str = DNAType.PLASTID
            elif "plasmid" in files:
                type_str = DNAType.PLASMID
            else:
                type_str = DNAType.NUCLEAR
            type_str = str(type_str)

            # Get the node for this taxon
            node = get_node(taxon, taxonomy, type_taxon_to_node[type_str], type_nodes[type_str], ranks=ranks)
            
            # Skip if not in one of the allowed ranks (e.g. class, order)
            if node.rank != restrict_ranks:
                continue

            # Skip if we have too many sequences for this leaf node
            if max_seqs and node.nseq >= max_seqs:
                continue

            partition = node.nseq % partitions

            node.nseq += 1

            seqtree.add(accession, node, partition)

    print(f'Saving seqtree with {len(seqtree)} sequences to {output}')
    seqtree.save(output)

    if render:
        print(f'Rendering seqtree to {render}')
        seqtree.render(filepath=render)


if __name__ == "__main__":
    app()
