import typer
from rich.progress import track

from pathlib import Path
from corgi.seqdict import SeqDict

from seqbank import SeqBank
#  
def main(
    seqdict_path:Path = typer.Argument(...,help="The seqdict which has the sequences to use."), 
    seqbank_path:Path = typer.Argument(...,help="The path to seqbank."), 
    save_removed:Path = None,
    get:bool=False,
):    
    print(f"Loading seqdict {seqdict_path}")
    seqdict = SeqDict.load(seqdict_path)

    print(f"Loading seqbank {seqbank_path}")
    seqbank = SeqBank(seqbank_path)

    accessions_to_add = seqdict.keys()
    missing = seqbank.missing(accessions_to_add)
    print(missing)
    print(len(missing), 'missing')

    if save_removed:
        for accession in missing:
            del seqdict[accession]
        
        seqdict.save(save_removed)


if __name__ == "__main__":
    typer.run(main)
