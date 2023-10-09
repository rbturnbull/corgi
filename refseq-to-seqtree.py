from multitax import NcbiTx
from hierarchicalsoftmax import SoftmaxNode
from corgi.seqtree import SeqTree
from seqbank import SeqBank

# tax = NcbiTx(files="multitaxtaxdump.tar.gz")
tax = NcbiTx()


MAX_SEQS = 100
partitions = 5

seqbank = SeqBank("/data1/refseq-opt.sb")

#https://academic.oup.com/nar/article/44/D1/D733/2502674
dna_prefixes = set(["NC", "AC", "NZ", "NT", "NW", "NG"])

root = SoftmaxNode("root", rank="root", nseq=0)
seqtree = SeqTree(root)

MITOCHONDRION = "Mitochondrion"
NUCLEAR = "Nuclear"
PLASTID = "Plastid"
PLASMID = "Plasmid"

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

        if accession not in seqbank:
            continue

        partition = node.nseq % partitions

        node.nseq += 1

        seqtree.add(accession, node, partition)


seqtree.save(f"refseq220-seqtree-easy-order-{MAX_SEQS}b.pkl")
root.render(filepath=f"refseq220-seqtree-easy-order-{MAX_SEQS}b.svg")
print('len seqtree', len(seqtree))