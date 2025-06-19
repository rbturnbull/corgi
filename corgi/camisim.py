from multitax import NcbiTx
import pandas as pd
from pathlib import Path
import typer
from .data import SeqIODataloader
from .refseq import DNAType

app = typer.Typer()


@app.command()
def camisim(
    mapping: Path = typer.Argument(..., help="The path to the GSA mapping file (gsa_mapping.tsv.gz)"),
    genomes_dir: Path = typer.Argument(..., help="The directory containing the genomes"),
    output_csv: Path = typer.Argument(..., help="The output CSV file path"),
):
    """ Convert GSA mapping to a CSV file with Corgi and Tiara labels as columns. """
    df = pd.read_csv(mapping, sep="\t", compression="gzip")
    df['corgi_category'] = ""
    df['tiara_category'] = ""

    print('Loading taxonomy...')
    taxonomy = NcbiTx()

    dataloader = SeqIODataloader(genomes_dir)
    for _, record in dataloader.iter_records():
        dna_type = DNAType.from_description(record.description)
        row = df['contig_id'] == record.accession
        if dna_type == DNAType.MITOCHONDRION:
            df.loc[row, 'corgi_category'] = "Mitochondrion"
            df.loc[row, 'tiara_category'] = "mit"
        elif dna_type == DNAType.PLASTID:
            df.loc[row, 'corgi_category'] = "Plastid"
            df.loc[row, 'tiara_category'] = "pla"
        elif dna_type == DNAType.PLASMID:
            df.loc[row, 'corgi_category'] = "Plasmid"
            df.loc[row, 'tiara_category'] = "unk2"
        else:
            # bac/arc/unk/unk1
            taxon_id = df[row]['taxon_id'].values[0]
            superkingdom = taxonomy.parent_rank(taxon_id, "superkingdom")
            df.loc[row, 'corgi_category'] = f"Nuclear/{superkingdom}"
            tiara_category = superkingdom.lower()[:3]  
            if tiara_category not in ["bac", "arc", "euk"]:
                tiara_category = "unk1"
            df.loc[row, 'tiara_category'] = tiara_category

    print(f"Writing to {output_csv}")        
    df.to_csv(output_csv, index=False)
    
    
if __name__ == "__main__":
    app()