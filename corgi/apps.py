from typing import TYPE_CHECKING
from pathlib import Path
import typer
import torchapp as ta
from rich.console import Console


if TYPE_CHECKING:
    from torch import nn
    import pandas as pd
    from hierarchicalsoftmax.nodes import SoftmaxNode
    from torchmetrics import Metric
    from .data import CorgiDataModule

console = Console()

def output_results_bar_chart(predictions: "pd.Series", top_k: int = 10, bar_size: int = 80):
    from rich.table import Table
    from rich.box import SIMPLE

    value_counts = predictions.value_counts()
    total_categories = len(value_counts)
    if top_k:
        original_value_counts = value_counts
        value_counts = value_counts.iloc[:top_k]
    table = Table(box=SIMPLE)
    table.add_column("Prediction", justify="left", style="bold")
    table.add_column("Proportion", justify="left", style="blue")
    table.add_column("Count", justify="right")
    for prediction, count in value_counts.items():
        proportion = count / len(predictions)
        bar = "█" * int(proportion * bar_size)
        table.add_row(prediction, f"{bar} {proportion:.1%}", f"{count}")
    if total_categories > top_k:
        count = sum(original_value_counts.iloc[top_k:])
        proportion = count / len(predictions)
        bar = "█" * int(proportion * bar_size)
        table.add_row(
            "Other Categories",
            f"{bar} {proportion:.1%}",
            f"{count}"
        )
    # Add line
    table.add_row('────────────────────', '─' * bar_size, '─' * 10, style="red")
    table.add_row(
        "Total",
        "",
        f"{len(predictions)}"
    )

    console.print(table)



def set_alphas_with_phi(tree, phi=1.0):
    import numpy as np

    for node in tree.pre_order_iter():
        node.alpha = np.power(phi, len(node.ancestors))


class Corgi(ta.TorchApp):
    """
    corgi - Classifier for ORganelle Genomes (Inter alia)
    """
    @ta.method
    def data(
        self,
        seqtree: Path = ta.Param(..., help="The seqtree which has the sequences to use."),
        seqbank: Path = ta.Param(..., help="The seqbank file with the sequences."),
        validation_partition:int = ta.Param(default=0, help="The partition to use for validation."),
        batch_size: int = ta.Param(default=32, help="The batch size."),
        validation_length:int = ta.Param(default=1_000, help="The standard length of sequences to use for validation."),
        phi:float=ta.Param(default=1.0, tune=True, tune_min=0.8, tune_max=1.2, help="A multiplication factor for the loss at each level of the tree."),
        test_partition:int = ta.Param(default=0, help="The partition to retain for testing."),
        minimum_length: int = ta.Param(default=150, help="The minimum length to truncate sequences in a training batch."),
        maximum_length: int = ta.Param(default=3_000, help="The maximum length to truncate sequences in a training batch."),
        skewness:float = ta.Param(default=5, help="The skewness of the distribution of sequence lengths in a batch."),
        loc:float = ta.Param(default=600, help="A parameter to shift the centre of the distribution of sequence lengths."),
        scale:float = ta.Param(default=1000, help="A parameter to scale the distribution of sequence lengths."),
        # tips_mode:bool = False,
        max_items:int = ta.Param(default=0, help="The maximum number of items to use for training. If zero then all items are used."),
        train_all:bool = ta.Param(default=False, help="Whether or not to use the validation partition for training."),
    ) -> "CorgiDataModule":
        """
        Creates a Pytorch Lightning object which Corgi uses in training and prediction.
        """
        from seqbank import SeqBank
        from polytorch import HierarchicalData
        from .data import CorgiDataModule
        from .seqtree import SeqTree
        

        assert seqtree is not None, "No seqtree given"
        assert seqbank is not None, "No seqbank given"
        
        seqtree = Path(seqtree)
        seqbank = Path(seqbank)

        print(f"Loading seqtree {seqtree}")
        seqtree = SeqTree.load(seqtree)
        self.classification_tree = seqtree.classification_tree

        print(f"Loading seqbank {seqbank}")
        seqbank = SeqBank(seqbank)

        self.classification_tree = seqtree.classification_tree

        set_alphas_with_phi(self.classification_tree, phi=phi)

        self.output_types = [
            HierarchicalData(root=self.classification_tree),
        ]

        print(f"Creating dataloaders with batch_size {batch_size} and validation partition {validation_partition}.")
        data = CorgiDataModule(
            seqtree=seqtree,
            seqbank=seqbank,
            batch_size=batch_size,
            validation_partition=validation_partition,
            max_items=max_items,
            minimum_length=minimum_length,
            maximum_length=maximum_length,
            validation_length=validation_length,
            test_partition=test_partition,
            train_all=train_all,
            skewness=skewness,
            loc=loc,
            scale=scale,
        )

        return data

    @ta.method("module_class")
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
        **kwargs,
    ) -> 'nn.Module':
        """
        Creates a deep learning model for the Corgi to use.

        Returns:
            nn.Module: The created model.
        """
        from polytorch import total_size
        from .models import calc_cnn_dims_start, ConvClassifier

        assert self.classification_tree

        num_classes = total_size(self.output_types)

        if pretrained:
            module_class = self.module_class(**kwargs)
            module = module_class.load_from_checkpoint(pretrained)
            model = module.model
            model.replace_output_types(self.output_types, final_bias=final_bias)
            return model

        # if cnn_dims_start not given then calculate it from the MACC
        if not cnn_dims_start:
            assert macc

            cnn_dims_start = calc_cnn_dims_start(
                macc=macc,
                seq_len=1024, # arbitary number
                embedding_dim=embedding_dim,
                cnn_layers=cnn_layers,
                kernel_size=kernel_size,
                factor=factor,
                penultimate_dims=penultimate_dims,
                num_classes=num_classes,
            )

        return ConvClassifier(
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

    @ta.method    
    def metrics(self) -> list[tuple[str,'Metric']]:
        from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric

        rank_accuracy = RankAccuracyTorchMetric(
            root=self.classification_tree, 
            ranks={
                1:"type_accuracy", 
                2:"superkingdom_accuracy",
                3:"kingdom_accuracy",
                4:"phylum_accuracy",
                5:"class_accuracy",
                6:"order_accuracy",
            },
        )
                
        return [('rank_accuracy', rank_accuracy)]

    @ta.method
    def monitor(self):
        return "superkingdom_accuracy"

    @ta.method
    def loss_function(self):
        from polytorch import PolyLoss

        assert self.output_types
        return PolyLoss(data_types=self.output_types, feature_axis=1)

    def package_name(self) -> str:
        """
        Returns the name of the package.
        """
        return "bio-corgi"

    @ta.method
    def prediction_dataloader(
        self,
        module,
        input: list[Path] = ta.Param(None, help="A fasta file with sequences to be classified."),
        file: list[Path] = ta.Param(None, help="A fasta file with sequences to be classified (DEPRECATED. Use `input`)."),
        seqtree: Path = ta.Param(None, help="The seqtree with the classification tree to use. DEPRECATED."),
        max_seqs: int = None,
        batch_size:int = 1,
        max_length:int = 5_000,
        min_length:int = 128,
        **kwargs,
    ):
        from .data import SeqIODataloader
        from .seqtree import SeqTree

        files = []
        if input:
            if isinstance(input, (str, Path)):
                input = [input]
            files.extend(input)
        if file:
            if isinstance(file, (str, Path)):
                file = [file]
            files.extend(file)

        if not files:
            raise typer.BadParameter("No files given to classify.")

        if seqtree and Path(seqtree).exists():
            seqtree = SeqTree.load(seqtree)
            self.classification_tree = seqtree.classification_tree
        else:
            self.classification_tree = module.hparams['classification_tree']
        self.dataloader = SeqIODataloader(files=files, batch_size=batch_size, max_length=max_length, max_seqs=max_seqs, min_length=min_length)
        return self.dataloader

    def node_to_str(self, node:'SoftmaxNode') -> str:
        """ 
        Converts the node to a string
        """
        from .seqtree import node_to_str
        return node_to_str(node)

    @ta.method
    def output_results(
        self,
        results,
        output_csv: Path = ta.Param(default=None, help="A path to output the results as a CSV."),
        output_tips_csv: Path = ta.Param(default=None, help="A path to output the results as a CSV which only stores the probabilities at the tips."),
        # output_fasta: Path = ta.Param(default=None, help="A path to output the results in FASTA format."),
        image_dir: Path = ta.Param(default=None, help="A directory to output the results as images."),
        image_format:str = "svg",
        image_threshold:float = 0.005,
        prediction_threshold:float = ta.Param(default=0.5, help="The threshold value for making hierarchical predictions."),
        **kwargs,
    ):
        import torch
        import pandas as pd
        from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions, render_probabilities
        
        assert self.classification_tree # This should be saved on the checkpoint
        
        classification_probabilities = node_probabilities(results[0], root=self.classification_tree)
        category_names = [self.node_to_str(node) for node in self.classification_tree.node_list if not node.is_root]
        chunk_details = pd.DataFrame(self.dataloader.chunk_details, columns=["file", "original_id", "description", "chunk"])
        predictions_df = pd.DataFrame(classification_probabilities.numpy(), columns=category_names)

        results_df = pd.concat(
            [chunk_details.drop(columns=['chunk']), predictions_df],
            axis=1,
        )

        # Average over chunks
        results_df["chunk_index"] = results_df.index
        results_df = results_df.groupby(["file", "original_id", "description"]).mean().reset_index()

        # sort to get original order
        results_df = results_df.sort_values(by="chunk_index").drop(columns=["chunk_index"])
        
        # Get new tensors now that we've averaged over chunks
        classification_probabilities = torch.as_tensor(results_df[category_names].to_numpy()) 

        # get greedy predictions which can use the raw activation or the softmax probabilities
        predictions = greedy_predictions(
            classification_probabilities, 
            root=self.classification_tree, 
            threshold=prediction_threshold,
        )

        results_df['prediction'] = [
            self.node_to_str(node)
            for node in predictions
        ]

        results_df['accession'] = results_df['original_id'].apply(lambda x: x.split("#")[0])
        
        def get_prediction_probability(row):
            prediction = row["prediction"]
            if prediction in row:
                return row[prediction]
            return 1.0
        
        results_df['probability'] = results_df.apply(get_prediction_probability, axis=1)

        # Reorder columns
        results_df = results_df[["file", "accession", "prediction", "probability", "original_id", "description" ] + category_names]

        # Output images
        if image_dir:
            console.print(f"Writing inference probability renders to: {image_dir}")
            image_dir = Path(image_dir)
            image_paths = []
            for _, row in results_df.iterrows():
                filepath = row['file']
                accession = row['accession']
                image_path = image_dir / Path(filepath).name / f"{accession}.{image_format}"
                image_paths.append(image_path)

            render_probabilities(
                root=self.classification_tree, 
                filepaths=image_paths,
                probabilities=classification_probabilities,
                predictions=predictions,
                threshold=image_threshold,
            )

        # Output Bar Chart
        output_results_bar_chart(results_df["prediction"], top_k=10)

        if not (image_dir or output_csv or output_tips_csv):
            print("No output files requested.")
            
        # if output_fasta:
        #     console.print(f"Writing results for {len(results_df)} repeats to: {output_fasta}")
        #     with open(output_fasta, "w") as fasta_out:
        #         for file in self.dataloader.files:
        #             for record in SeqIO.parse(file, "fasta"):
        #                 original_id = record.id
        #                 row = results_df.loc[results_df.original_id == original_id]
        #                 if len(row) == 0:
        #                     SeqIO.write(record, fasta_out, "fasta")
        #                     continue

        #                 accession = row['accession'].item()
        #                 prediction = row["prediction"].item()
                        
        #                 new_id = f"{accession}#{prediction}"
        #                 record.id = new_id
                        
        #                 # Adapt description
        #                 record.description = record.description.replace(original_id, "")
        #                 last_bracket = record.description.rfind(")")
        #                 if last_bracket == -1:
        #                     record.description = f"{record.description} ( "
        #                 else:
        #                     record.description = record.description[:last_bracket].rstrip() + ", "

        #                 if prediction in row:
        #                     new_probability = row[prediction].values[0]
        #                 else:
        #                     new_probability = 1.0 # i.e.root
        #                 record.description = f"{record.description} classification probability = {new_probability:.2f} )"

        #                 SeqIO.write(record, fasta_out, "fasta")

        if output_tips_csv:
            output_tips_csv = Path(output_tips_csv)
            output_tips_csv.parent.mkdir(exist_ok=True, parents=True)
            non_tips = [self.node_to_str(node) for node in self.classification_tree.node_list if not node.is_leaf]
            tips_df = results_df.drop(columns=non_tips)
            tips_df.to_csv(output_tips_csv, index=False)

        if output_csv:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(exist_ok=True, parents=True)
            console.print(f"Writing results for {len(results_df)} sequences to: {output_csv}")
            results_df.to_csv(output_csv, index=False)

        # if self.vector:
        #     # x = results_df.to_xarray()
        #     # breakpoint()
        #     # embeddings = xr.DataArray(results[0][1], dims=("accession", "embedding"))
        #     # embeddings.to_netcdf("embeddings.nc")
        #     torch.save(results[0][1], "embeddings.pkl")

        return results_df

    @ta.method
    def extra_hyperparameters(self) -> dict:
        """ Extra hyperparameters to save with the module. """
        return dict(
            classification_tree=self.classification_tree,
        )

    def checkpoint(
        self, 
        checkpoint:Path=ta.Param(default=None, help="A path to a checkpoint to load. If not given then a default checkpoint is used."),
        large: bool = ta.Param(default=False, help="Whether or not to use a large checkpoint (overridden by `checkpoint` parameter)."),
    ) -> str:
        if checkpoint:
            return checkpoint
        if large:
            return "https://figshare.unimelb.edu.au/ndownloader/files/55166705"
            raise NotImplementedError("Large checkpoint not yet available.")
        
        return "https://figshare.unimelb.edu.au/ndownloader/files/55166714"
