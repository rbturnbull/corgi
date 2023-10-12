import typer
from rich.progress import track

from pathlib import Path
from corgi.seqtree import SeqTree

from seqbank import SeqBank
#  
def main(
    seqtree_path:Path = typer.Argument(...,help="The seqtree which has the sequences to use."), 
    seqbank_path:Path = typer.Argument(...,help="The path to seqbank."), 
    save_removed:Path = None,
    get:bool=False,
):    
    print(f"Loading seqtree {seqtree_path}")
    seqtree = SeqTree.load(seqtree_path)

    print(f"Loading seqbank {seqbank_path}")
    seqbank = SeqBank(seqbank_path)

    accessions_to_add = seqtree.keys()
    missing = seqbank.missing(accessions_to_add)
    print(missing)
    print(len(missing), 'missing')

    if save_removed:
        for accession in missing:
            del seqtree[accession]
        
        seqtree.save(save_removed)


if __name__ == "__main__":
    typer.run(main)
