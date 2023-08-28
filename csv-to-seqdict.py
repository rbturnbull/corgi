import typer
import pandas as pd
from pathlib import Path
from corgi.hierarchy import create_hierarchy
from corgi.seqdict import SeqDict, SeqDetail


def main(csv: Path, seqdict_path:Path, gamma:float = 0.0, label_smoothing:float = 0.0):
    df = pd.read_csv(csv)

    # Build Hiearchy Tree
    assert 'hierarchy' in df.columns, f"Cannot find 'hierarchy' column in {csv}."
    classification_tree, classification_to_node, _ = create_hierarchy(
        df['hierarchy'].unique(), 
        label_smoothing=label_smoothing, 
        gamma=gamma,
    )

    seqdict = SeqDict()
    seqdict.classification_tree = classification_tree

    assert 'partition' in df.columns, f"Cannot find 'partition' column in {csv}."
    assert 'type' in df.columns, f"Cannot find 'type' column in {csv}."
    for _, row in df.iterrows():
        seqdict.add(
            accession=row['accession'],
            partition=int(row['partition']),
            node=classification_to_node[row['hierarchy']],
            type=int(row['type']),
        )
    
    del df

    # Save
    seqdict.save(seqdict_path)



if __name__ == "__main__":
    typer.run(main)
