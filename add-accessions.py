import typer
from rich.progress import track

from pathlib import Path
from corgi.seqdict import SeqDict

from corgi.seqbank import SeqBank
#  
def main(
    seqdict_path:Path = typer.Argument(...,help="The seqdict which has the sequences to use."), 
    seqbank_path:Path = typer.Argument(...,help="The path to seqbank."), 
    base_dir:Path = typer.Argument(...,help="The path to download the fasta files."), 
    email:str = typer.Argument(...,help="The email to use for downloading sequences."),
):    
    print(f"Loading seqdict {seqdict_path}")
    seqdict = SeqDict.load(seqdict_path)

    print(f"Loading seqbank {seqbank_path}")
    seqbank = SeqBank(seqbank_path)

    # print("Getting current accessions")
    # current_accessions = seqbank.get_accessions()
    # print("current_accessions", len(current_accessions))

    # print("Getting accessions to add")
    # accessions_to_add = set(seqdict.keys()) - current_accessions
    # print("accessions_to_add", len(accessions_to_add))

    accessions_to_add = seqdict.keys()
    seqbank.add_accessions(accessions_to_add, base_dir=base_dir, email=email, batch_size=1)


if __name__ == "__main__":
    typer.run(main)
