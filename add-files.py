from typing import List
import typer
from rich.progress import track

from pathlib import Path
from corgi.seqdict import SeqDict

from corgi.seqbank import SeqBank
#  
def main(
    seqbank_path:Path = typer.Argument(...,help="The path to seqbank."), 
    files:List[Path] = typer.Argument(...,help="The list of files to add."), 
):    
    print(f"Loading seqbank {seqbank_path}")
    seqbank = SeqBank(seqbank_path)

    for file in files:
        print(f"Adding file: '{file}'")
        seqbank.add_file(file)


if __name__ == "__main__":
    typer.run(main)
