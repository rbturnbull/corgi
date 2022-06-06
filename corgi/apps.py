from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
from fastai.metrics import accuracy, Precision, Recall, RocAuc, F1Score
import fastapp as fa
from rich.console import Console
console = Console()

from . import dataloaders, models, refseq

class Corgi(fa.FastApp):
    """
    corgi - Classifier for ORganelle Genomes
    """
    def __init__(self):
        super().__init__()
        self.categories = refseq.REFSEQ_CATEGORIES # If 

    def dataloaders(
        self,
        csv:Path = fa.Param(help="The CSV which has the sequences to use."),
        base_dir:Path = fa.Param(help="The base directory with the RefSeq HDF5 files."),
        batch_size:int = fa.Param(default=32, help="The batch size."),
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Corgi uses in training and prediction.

        Args:
            inputs (Path): The input file.
            batch_size (int): The number of elements to use in a batch for training and prediction. Defaults to 32.
        """
        if csv is None:
            raise Exception("No CSV given")
        if base_dir is None:
            raise Exception("No base_dir given")
        dls = dataloaders.create_dataloaders_refseq_path(csv, base_dir=base_dir, batch_size=batch_size)
        self.categories = dls.vocab
        return dls

    def model(
        self,
        embedding_dim: int = fa.Param(default=16, help="The size of the embeddings for the nucleotides (N, A, G, C, T)."),
        filters: int = fa.Param(default=256, help="The number of filters in each of the 1D convolution layers. These are concatenated together"),
        cnn_layers: int = fa.Param(default=6, help="The number of 1D convolution layers."),
        kernel_size_maxpool: int = fa.Param(default=2, help="The size of the pooling before going to the LSTM."),
        lstm_dims: int = fa.Param(default=256, help="The size of the hidden layers in the LSTM in both directions."),
        final_layer_dims: int = fa.Param(default=0, help="The size of a dense layer after the LSTM. If this is zero then this layer isn't used."),
        dropout: float = fa.Param(default=0.5, help="The amount of dropout to use. (not currently enabled)"),
    ) -> nn.Module:
        """
        Creates a deep learning model for the Corgi to use.

        Returns:
            nn.Module: The created model.
        """
        num_classes = len(self.categories)
        return models.ConvRecurrantClassifier(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            filters=filters,
            cnn_layers=cnn_layers,
            lstm_dims=lstm_dims,
            final_layer_dims=final_layer_dims,
            dropout=dropout,
            kernel_size_maxpool=kernel_size_maxpool,
        )

    def metrics(self):
        average = "macro"
        return [
            accuracy, 
            F1Score(average=average),
            Precision(average=average),
            Recall(average=average),
            RocAuc(average=average),
        ]

    def monitor(self):
        return "f1_score"