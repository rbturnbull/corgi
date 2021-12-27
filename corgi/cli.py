from os import mkdir
from click.core import batch
import typer
from pathlib import Path
from typing import List
from typing import Optional
import pandas as pd
import time

from fastcore.transform import Pipeline

from fastai.learner import load_learner

from . import training, dataloaders, profiling, preprocessing, optimization
from .transforms import SliceTransform

app = typer.Typer()


def version_callback(value: bool):
    """
    Prints the current version.
    """
    if value:
        import importlib.metadata
        version = importlib.metadata.version("corgi")
        typer.echo(version)
        raise typer.Exit()


@app.command()
def train(
    output_dir: Path,
    dataframe: Path,
    base_dir: Path = None,
    batch_size: int = 64,
    epochs: int = 20,
    lr_max: float = 1e-3,
    fp16: bool = True,
    distributed: bool = False,
    wandb: bool = False,
    wandb_name: str = "",
    # Model parameters (these should not be repeated here)
    # Can I use the delegate class from fastcore?
    embedding_dim: int =16,
    filters: int = 512,
    kernel_size_cnn: int = 9,
    lstm_dims: int = 256,
    final_layer_dims: int = 0,  # If this is zero then it isn't used.
    dropout: float = 0.5,
    kernel_size_maxpool: int = 2,
    residual_blocks: bool = False,
):
    """
    Trains a model from a preprocessed.
    """
    dls = dataloaders.create_dataloaders_refseq_path(dataframe, base_dir=base_dir, batch_size=batch_size)
    print('Outputting to: \t', output_dir)

    if wandb:
        import wandb
        if not wandb_name:
            wandb_name = output_dir.name
        wandb.init(project="corgi", name=wandb_name)

    learner = training.train(
        dls, 
        output_dir=output_dir, 
        epochs=epochs, 
        fp16=fp16, 
        distributed=distributed,
        lr_max=lr_max,
        embedding_dim=embedding_dim,
        filters=filters,
        kernel_size_cnn=kernel_size_cnn,
        lstm_dims=lstm_dims,
        final_layer_dims=final_layer_dims,
        dropout=dropout,
        kernel_size_maxpool=kernel_size_maxpool,
        residual_blocks=residual_blocks,
    )
    profiling.display_profiling()
    return learner


@app.command()
def optimize(
    output_dir: Path,
    study_name: str,
    dataframe: Path,
    n_trials: int,
    base_dir: Path = None,
    batch_size: int = 64,
    storage_name: str="sqlite:///corgi-studies.db",
    epochs: int = 20,
    fp16: bool = True,
    wandb: bool = True,
):
    """
    Optimizes hyperparameters.
    """
    dls = dataloaders.create_dataloaders_refseq_path(dataframe, base_dir=base_dir, batch_size=batch_size)
    print('Outputting to: \t', output_dir)

    study = optimization.optimize(
        dls, 
        output_dir=output_dir,
        n_trials=n_trials,    
        study_name=study_name,
        storage_name=storage_name,
        epochs=epochs,
        fp16=fp16,
        wandb=wandb,
    )

    profiling.display_profiling()
    return study


@app.command()
def export(
    output_dir: str,
    csv: Path,
    filename: str = "model",
    batch_size: int = 64,
    base_dir: Path = None,
    fp16: bool = False,
):
    """
    Exports a checkpointed model so that it is ready for inference.
    """
    print('Outputting to: \t', output_dir)

    df = pd.read_csv(csv)
    dls = dataloaders.create_dataloaders_refseq(df, batch_size=batch_size, base_dir=base_dir )
    learner = training.export(dls, output_dir=output_dir, filename=filename, fp16=fp16)
    return learner


@app.command()
def validate(
    learner_path: Path,
    csv: Path,
    output_dir: Path,
    category: Optional[List[str]] = typer.Option(None),
    length: Optional[List[int]] = typer.Option(None),
    batch_size: int = 64,
    base_dir: Path = None,
):
    """
    Tests classification on validation sequences.
    """
    df = pd.read_csv(csv)
    df = df[ df['validation'] == 1]
    print(f'Validation on {len(df)} sequences.')

    # open learner from pickled file
    learner = load_learner(learner_path)
    after_item_original = Pipeline(learner.dls.after_item)

    # print(learner.show_training_loop())
    # return
    
    # rename variables for clarity
    # variables are singular because of the way typer handles lists
    seq_lengths = length
    categories = category
    
    if not categories:
        categories = df['category'].unique()

    if not seq_lengths:
        seq_lengths = [150]

    output_dir.mkdir(exist_ok=True, parents=True)
    results = []

    print("seq_lengths", seq_lengths)
    for seq_len in seq_lengths:
        print(f"Validating seqs at length {seq_len}")
        # learner.dls.after_item = Pipeline(after_item_original)
        # learner.dls.after_item.add(SliceTransform(seq_len))

        for category in categories:
            print(f"Validating category {category}")

            category_df = df[df['category'] == category].copy()
        
            # Classify results
            start_time = time.time()
            dl = learner.dls.test_dl(category_df)
            dl.before_batch = Pipeline()
            dl.after_item = Pipeline(after_item_original)
            dl.after_item.add(SliceTransform(seq_len))
            print(dl.after_item)
            
            result = learner.get_preds(dl=dl, reorder=False, with_decoded=True)
            end_time = time.time()
            classification_time = end_time - start_time

            # Output results
            category_df['prediction'] = list(map(lambda category_index: learner.dls.vocab[category_index], result[2]))
            category_df['probability'] =  [probs[category].item() for probs, category in zip(result[0], result[2])]    

            total = len(category_df)
            correct = (category_df['prediction'] == category).sum()
            sensitivity = correct/total

            # Output results
            category_df.to_csv(str(output_dir/f"validation-{category}-{seq_len}.csv"))
            row_results = dict(
                category=category,
                seq_len=seq_len,
                total=total,
                correct=correct,
                sensitivity=sensitivity,
                classification_time=classification_time,
            )
            print(row_results)
            results.append(row_results)
    results_df = pd.DataFrame(results)
    results_df.to_csv(str(output_dir/f"results.csv"))
    print(results_df)
    profiling.display_profiling()


@app.command()
def classify(
    learner_path: Path,
    output_csv: Path,
    fasta_paths: List[Path],
    max_seqs: int = None,
):
    """
    Classifies sequences in a Fasta file.
    """
    # Read Fasta file
    df = dataloaders.fastas_to_dataframe(fasta_paths=fasta_paths, max_seqs=max_seqs)

    # open learner from pickled file
    learner = load_learner(learner_path)

    # Classify results
    dl = learner.dls.test_dl(df)
    result = learner.get_preds(dl=dl, reorder=False, with_decoded=True)

    # Output results
    df['prediction'] = list(map(lambda category_index: learner.dls.vocab[category_index], result[2]))
    df['probability'] =  [probs[category].item() for probs, category in zip(result[0], result[2])]    
    df = df[['id','prediction', 'probability', 'file'] ]
    df.to_csv(str(output_csv))

    profiling.display_profiling()


@app.command()
def preprocess(
    output: Path,
    base_dir: Path = None,
    category: Optional[List[str]] = typer.Option(None),
    max_files : int = None,
    file_index: Optional[List[int]] = typer.Option(None),
):

    df = preprocessing.preprocess( category, base_dir, max_files=max_files, file_indexes=file_index )
    output.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output)
    print(df)


@app.command()
def download(
    base_dir: Path = None,
    category: Optional[List[str]] = typer.Option(None),
    max_files : int = None,
):
    preprocessing.download( category, base_dir, max_files=max_files )


@app.command()
def repo():
    """
    Opens the repository in a web browser
    """
    typer.launch("https://gitlab.unimelb.edu.au/mdap/corgi")


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Prints the current version."
    ),    
):
    """
    CORGI - Classifier for ORganelle Genomes.
    """
    pass

