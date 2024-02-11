import time
from pathlib import Path
from typing import List
from torch import nn
import pandas as pd
from fastai.data.core import DataLoaders
from torchapp.util import copy_func, call_func, change_typer_to_defaults, add_kwargs
from fastai.learner import Learner, load_learner
import torchapp as ta
from rich.console import Console
from rich.table import Table
from rich.box import SIMPLE
from Bio.SeqIO import FastaIO
import numpy as np
from polytorch import PolyLoss, HierarchicalData, total_size
from polytorch.metrics import HierarchicalGreedyAccuracy
from seqbank import SeqBank

from . import dataloaders, models, refseq, transforms
from .seqtree import SeqTree

console = Console()


def set_alpha(tree, k=0.5):
    for node in tree.pre_order_iter():
        node.alpha = np.power(k, len(node.ancestors))


class Corgi(ta.TorchApp):
    """
    corgi - Classifier for ORganelle Genomes Inter alia
    """
    def dataloaders(
        self,
        seqtree: Path = ta.Param(help="The seqtree which has the sequences to use."),
        seqbank:Path = ta.Param(help="The HDF5 file with the sequences."),
        validation_partition:int = ta.Param(default=0, help="The partition to use for validation."),
        batch_size: int = ta.Param(default=32, help="The batch size."),
        validation_length:int = 1_000,
        # deform_lambda:float = ta.Param(default=None, help="The lambda for the deform transform."),
        tips_mode:bool = False,
        max_seqs:int = None,
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Corgi uses in training and prediction.

        Args:
            inputs (Path): The input file.
            batch_size (int): The number of elements to use in a batch for training and prediction. Defaults to 32.
        """
        if seqtree is None:
            raise Exception("No seqtree given")
        if seqbank is None:
            raise Exception("No seqbank given")

        seqtree = Path(seqtree)
        seqbank = Path(seqbank)

        print(f"Loading seqtree {seqtree}")
        seqtree = SeqTree.load(seqtree)

        print(f"Loading seqbank {seqbank}")
        seqbank = SeqBank(seqbank)

        print(f"Creating dataloaders with batch_size {batch_size} and validation partition {validation_partition}.")
        dls = dataloaders.create_dataloaders(
            seqtree=seqtree,
            seqbank=seqbank,
            batch_size=batch_size,
            validation_partition=validation_partition,
            validation_length=validation_length,
            max_seqs=max_seqs,
        )
        dls.classification_tree = seqtree.classification_tree
        self.classification_tree = dls.classification_tree
        self.classification_tree.tips_mode = tips_mode
        if tips_mode:
            self.classification_tree.index_tips_mode()
            set_alpha(self.classification_tree)

        self.output_types = [
            HierarchicalData(root=self.classification_tree),
        ]
        return dls

    def model(
        self,
        pretrained:Path = ta.Param(None, help="A pretrained model to finetune."),
        embedding_dim: int = ta.Param(
            default=8,
            help="The size of the embeddings for the nucleotides (N, A, G, C, T).",
            tune=True,
            tune_min=4,
            tune_max=32,
            log=True,
        ),
        filters: int = ta.Param(
            default=256,
            help="The number of filters in each of the 1D convolution layers. These are concatenated together",
        ),
        cnn_layers: int = ta.Param(
            default=6,
            help="The number of 1D convolution layers.",
            tune=True,
            tune_min=2,
            tune_max=6,
        ),
        kernel_size_maxpool: int = ta.Param(
            default=2,
            help="The size of the pooling before going to the LSTM.",
        ),
        lstm_dims: int = ta.Param(default=256, help="The size of the hidden layers in the LSTM in both directions."),
        final_layer_dims: int = ta.Param(
            default=0, help="The size of a dense layer after the LSTM. If this is zero then this layer isn't used."
        ),
        dropout: float = ta.Param(
            default=0.2,
            help="The amount of dropout to use. (not currently enabled)",
            tune=True,
            tune_min=0.0,
            tune_max=0.3,
        ),
        final_bias: bool = ta.Param(
            default=True,
            help="Whether or not to use bias in the final layer.",
            tune=True,
        ),
        cnn_only: bool = True,
        kernel_size: int = ta.Param(
            default=3, help="The size of the kernels for CNN only classifier.", tune=True, tune_choices=[3, 5, 7, 9]
        ),
        cnn_dims_start: int = ta.Param(
            default=None,
            help="The size of the number of filters in the first CNN layer. If not set then it is derived from the MACC",
        ),
        factor: float = ta.Param(
            default=2.0,
            help="The factor to multiply the number of filters in the CNN layers each time it is downscaled.",
            tune=True,
            log=True,
            tune_min=0.5,
            tune_max=2.5,
        ),
        penultimate_dims: int = ta.Param(
            default=1024,
            help="The factor to multiply the number of filters in the CNN layers each time it is downscaled.",
            tune=True,
            log=True,
            tune_min=512,
            tune_max=2048,
        ),
        include_length: bool = True,
        transformer_heads: int = ta.Param(8, help="The number of heads in the transformer."),
        transformer_layers: int = ta.Param(0, help="The number of layers in the transformer. If zero then no transformer is used."),
        macc:int = ta.Param(
            default=10_000_000,
            help="The approximate number of multiply or accumulate operations in the model. Used to set cnn_dims_start if not provided explicitly.",
        ),
    ) -> nn.Module:
        """
        Creates a deep learning model for the Corgi to use.

        Returns:
            nn.Module: The created model.
        """
        assert self.classification_tree

        num_classes = total_size(self.output_types)

        if pretrained:
            pretrained_learner = load_learner(pretrained)
            model = pretrained_learner.model
            model.replace_output_types(self.output_types, final_bias=final_bias)
            return model

        # if cnn_dims_start not given then calculate it from the MACC
        if not cnn_dims_start:
            assert macc

            cnn_dims_start = models.calc_cnn_dims_start(
                macc=macc,
                seq_len=1024, # arbitary number
                embedding_dim=embedding_dim,
                cnn_layers=cnn_layers,
                kernel_size=kernel_size,
                factor=factor,
                penultimate_dims=penultimate_dims,
                num_classes=num_classes,
            )

        return models.ConvClassifier(
            num_embeddings=5,  # i.e. the size of the vocab which is N, A, C, G, T
            kernel_size=kernel_size,
            factor=factor,
            cnn_layers=cnn_layers,
            output_types=self.output_types,
            kernel_size_maxpool=kernel_size_maxpool,
            final_bias=final_bias,
            dropout=dropout,
            cnn_dims_start=cnn_dims_start,
            penultimate_dims=penultimate_dims,
            include_length=include_length,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
        )

        # return models.ConvRecurrantClassifier(
        #     num_classes=num_classes,
        #     embedding_dim=embedding_dim,
        #     filters=filters,
        #     cnn_layers=cnn_layers,
        #     lstm_dims=lstm_dims,
        #     final_layer_dims=final_layer_dims,
        #     dropout=dropout,
        #     kernel_size_maxpool=kernel_size_maxpool,
        #     final_bias=final_bias,
        # )

    def metrics(self):
        return [
            HierarchicalGreedyAccuracy(root=self.classification_tree, max_depth=1, data_index=0, name="type_accuracy"),
            HierarchicalGreedyAccuracy(root=self.classification_tree, max_depth=2, data_index=0, name="superkingdom_accuracy"),
            HierarchicalGreedyAccuracy(root=self.classification_tree, max_depth=3, data_index=0, name="kingdom_accuracy"),
            HierarchicalGreedyAccuracy(root=self.classification_tree, max_depth=4, data_index=0, name="phylum_accuracy"),
            HierarchicalGreedyAccuracy(root=self.classification_tree, max_depth=5, data_index=0, name="class_accuracy"),
            HierarchicalGreedyAccuracy(root=self.classification_tree, max_depth=6, data_index=0, name="order_accuracy"),
        ]

    def monitor(self):
        return "superkingdom_accuracy"

    def loss_func(self):
        assert self.output_types
        return PolyLoss(data_types=self.output_types, feature_axis=1)

    def inference_dataloader(
        self,
        learner,
        file: List[Path] = ta.Param(None, help="A fasta file with sequences to be classified."),
        max_seqs: int = None,
        batch_size:int = 1,
        max_length:int = 5_000,
        min_length:int = 128,
        **kwargs,
    ):
        self.seqio_dataloader = dataloaders.SeqIODataloader(files=file, device=learner.dls.device, batch_size=batch_size, max_length=max_length, max_seqs=max_seqs, min_length=min_length)
        self.categories = learner.dls.vocab
        return self.seqio_dataloader

    def output_results(
        self,
        results,
        output_dir:Path = ta.Param(default=None, help="A path to output the results as a CSV."),
        csv: Path = ta.Param(default=None, help="A path to output the results as a CSV. If not given then a default name is chosen inside the output directory."),
        save_filtered:bool = ta.Param(default=True, help="Whether or not to save the filtered sequences."),
        threshold: float = ta.Param(
            default=None, 
            help="The threshold to use for filtering. "
                "If not given, then only the most likely category used for filtering.",
        ),
        **kwargs,
    ):
        if not output_dir:
            time_string = time.strftime("%Y_%m_%d-%I_%M_%S_%p")
            output_dir = f"corgi-output-{time_string}"

        output_dir = Path(output_dir)

        chunk_details = pd.DataFrame(self.seqio_dataloader.chunk_details, columns=["file", "accession", "chunk"])
        predictions_df = pd.DataFrame(results[0].numpy(), columns=self.categories)
        results_df = pd.concat(
            [chunk_details.drop(columns=['chunk']), predictions_df],
            axis=1,
        )

        # Average over chunks
        results_df = results_df.groupby(["file", "accession"]).mean().reset_index()

        columns = set(predictions_df.columns)

        results_df['prediction'] = results_df[self.categories].idxmax(axis=1)
        results_df['eukaryotic'] = predictions_df[list(columns & set(refseq.EUKARYOTIC))].sum(axis=1)
        results_df['prokaryotic'] = predictions_df[list(columns & set(refseq.PROKARYOTIC))].sum(axis=1)
        results_df['organellar'] = predictions_df[list(columns & set(refseq.ORGANELLAR))].sum(axis=1)

        if not csv:
            output_dir.mkdir(parents=True, exist_ok=True)
            csv = output_dir / f"corgi-output.csv"

        console.print(f"Writing results for {len(results_df)} sequences to: {csv}")
        results_df.to_csv(csv, index=False)

        # Write all the sequences to fasta files
        if save_filtered:
            record_to_string = FastaIO.as_fasta

            output_dir.mkdir(parents=True, exist_ok=True)
            
            file_handles = {}

            for file, record in self.seqio_dataloader.iter_records():
                row = results_df[ (results_df.accession == record.id) & (results_df.file == file) ]
                if len(row) == 0:
                    categories = ["unclassified"]
                else:
                    # Get the categories to write to
                    if not threshold:
                        # if no threshold then just use the most likely category
                        categories = [row['prediction'].item()]
                    else:
                        # otherwise use all categories above or equal to the threshold
                        category_predictions = row.iloc[0][self.categories]
                        categories = [category_predictions[category_predictions >= threshold].index.item()]

                for category in categories:
                    if category not in file_handles:
                        file_path = output_dir / f"{category}.fasta"
                        file_handles[category] = open(file_path, "w")

                    file_handle = file_handles[category]
                    file_handle.write(record_to_string(record))

            for file_handle in file_handles.values():
                file_handle.close()

        # Output bar chart
        from termgraph.module import Data, BarChart, Args

        value_counts = results_df['prediction'].value_counts()
        data = Data([[count] for count in value_counts], value_counts.index)
        chart = BarChart(
            data,
            Args(
                space_between=False,
            ),
        )

        chart.draw()

    def category_counts_dataloader(self, dataloader, description):
        from collections import Counter

        counter = Counter()
        for batch in dataloader:
            counter.update(batch[1].cpu().numpy())
        total = sum(counter.values())

        table = Table(title=f"{description}: Categories in epoch", box=SIMPLE)

        table.add_column("Category", justify="right", style="cyan", no_wrap=True)
        table.add_column("Count", justify="center")
        table.add_column("Percentage")

        for category_id, category in enumerate(self.categories):
            count = counter[category_id]
            table.add_row(category, str(count), f"{count/total*100:.1f}%")

        table.add_row("Total", str(total), "")

        console.print(table)

    def category_counts(self, **kwargs):
        dataloaders = call_func(self.dataloaders, **kwargs)
        self.category_counts_dataloader(dataloaders.train, "Training")
        self.category_counts_dataloader(dataloaders.valid, "Validation")

    def pretrained_location(self) -> str:
        return "https://github.com/rbturnbull/corgi/releases/download/v0.3.1-alpha/corgi-0.3.pkl"
