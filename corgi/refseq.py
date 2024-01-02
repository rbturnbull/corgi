from multitax import NcbiTx
from hierarchicalsoftmax import SoftmaxNode
from pathlib import Path
import requests
from enum import Enum
from appdirs import user_cache_dir
from torchapp.download import cached_download
from .seqtree import SeqTree

import typer

app = typer.Typer()

class DNAType(Enum):
    NUCLEAR = "Nuclear"
    MITOCHONDRION = "Mitochondrion"
    PLASTID = "Plastid"
    PLASMID = "Plasmid"



# https://academic.oup.com/nar/article/44/D1/D733/2502674
DNA_PREFIXES = set(["NC", "AC", "NZ", "NT", "NW", "NG"])


def get_refseq_release() -> int:
    url = "https://ftp.ncbi.nlm.nih.gov/refseq/release/RELEASE_NUMBER"
    response = requests.get("https://ftp.ncbi.nlm.nih.gov/refseq/release/RELEASE_NUMBER")
    data = response.text.strip()
    return int(data)


def get_catalogue():
    version = get_refseq_release()
    filename = f"RefSeq-release{version}.catalog.gz""
    url = f"https://ftp.ncbi.nlm.nih.gov/refseq/release/release-catalog/{filename}"

    local_path = Path(user_cache_dir("torchapps"))/"Corgi"/filename
    cached_download(url, local_path=local_path)
    return local_path


@app.command()
def refseq_to_seqtree(partitions:int=5, max_seqs:int=100)

    tax = NcbiTx()

    root = SoftmaxNode("root", rank="root", nseq=0)
    seqtree = SeqTree(root)


    types = [
        NUCLEAR,
        MITOCHONDRION,
        PLASTID,
        PLASMID,
    ]

    type_nodes = {}
    type_taxon_to_node = {}
    for type in types:
        type_nodes[type] = SoftmaxNode(type, parent=root, rank="type", nseq=0)
        type_taxon_to_node[type] = {}


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


    def get_node(taxon, taxonomy, taxon_to_node, root):
        current_node = root
        #ranks = ["superkingdom", "kingdom", "phylum", "class"]
        ranks = ["superkingdom", "kingdom", "phylum", "class", "order"]
        for rank in ranks:
            current_node = get_node_at_rank(taxon, taxonomy, taxon_to_node, current_node, rank)

        return current_node





    catalog = "RefSeq-release220.catalog"
    # catalog = "RefSeq-release219-dna-100.catalog"
    # catalog = "RefSeq-release219-dna-10000.catalog"

    with open(catalog) as f:
        for line in f:
            components = line.split("\t")
            accession = components[2]
            prefix = accession[:2]
            if prefix not in dna_prefixes:
                continue
            
            files = components[3]
            status = components[4]
            taxon = components[0]

            if "mitochondrion" in files:
                type_str = MITOCHONDRION
            elif "plastid" in files:
                type_str = PLASTID
            elif "plasmid" in files:
                type_str = PLASMID
            else:
                type_str = NUCLEAR


            node = get_node(taxon, tax, type_taxon_to_node[type_str], type_nodes[type_str])
            if node == type_nodes[type_str]:
                continue

            if node.nseq >= MAX_SEQS:
                continue

            partition = node.nseq % partitions

            node.nseq += 1

            seqtree.add(accession, node, partition)

    seqtree.save(f"refseq220-seqtree-easy-order-{MAX_SEQS}.pkl")
    root.render(filepath=f"refseq220-seqtree-easy-order-{MAX_SEQS}.svg")
    print('len seqtree', len(seqtree))