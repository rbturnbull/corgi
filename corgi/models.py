"""Neural network models for classifying DNA sequences."""

from typing import Callable, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from polytorch import split_tensor, total_size

from rich.console import Console

console = Console()


class PositionalEncoding(nn.Module):
    """
    Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Creates the sinusoidal positional encodings.

        Args:
            d_model (int): The size of the embeddings.
            dropout (float): The dropout probability.
            max_len (int): The maximum sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def conv3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """convolution of width 3 with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class ResidualBlock1D(nn.Module):
    """Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """Creates the layers of the block.

        Args:
            inplanes (int): The number of input channels.
            planes (int): The number of output channels.
            stride (int): The stride of the first convolution. If not 1, then the identity is downsampled to match.
            downsample (nn.Module | None): A module to downsample the identity.
            norm_layer: The normalization layer class. Defaults to BatchNorm1d.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Applies two convolutions and adds the (possibly downsampled) input."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ConvRecurrantClassifier(nn.Module):
    """A CNN followed by an optional bidirectional LSTM which classifies a sequence into a number of classes.

    The input is a tensor of nucleotide indices (N, A, C, G, T) of shape (batch, sequence length).
    """

    def __init__(
        self,
        num_classes,
        embedding_dim: int = 8,
        filters: int = 256,
        cnn_layers: int = 6,
        kernel_size_cnn: int = 9,
        lstm_dims: int = 256,
        final_layer_dims: int = 0,  # If this is zero then it isn't used.
        dropout: float = 0.5,
        kernel_size_maxpool: int = 2,
        residual_blocks: bool = False,
        final_bias: bool = True,
        multi_kernel_sizes: bool = True,
    ):
        """Creates the layers of the classifier.

        Args:
            num_classes (int): The number of output classes.
            embedding_dim (int): The size of the nucleotide embeddings.
            filters (int): The number of filters in each convolution.
            cnn_layers (int): The number of convolutions when `multi_kernel_sizes` is True.
            kernel_size_cnn (int): The kernel size of the convolution when `multi_kernel_sizes` is False.
            lstm_dims (int): The hidden size of the LSTM. If zero, then no LSTM is used.
            final_layer_dims (int): The size of an extra linear layer before the output. If zero, it is not used.
            dropout (float): The dropout probability.
            kernel_size_maxpool (int): The size of the max pooling.
            residual_blocks (bool): Whether to use residual blocks instead of a single convolution.
            final_bias (bool): Whether the output layer has a bias.
            multi_kernel_sizes (bool): Whether to use parallel convolutions with increasing kernel sizes.
        """
        super().__init__()

        num_embeddings = 5  # i.e. the size of the vocab which is N, A, C, G, T

        self.num_classes = num_classes
        self.num_embeddings = num_embeddings

        ########################
        ## Embedding
        ########################
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.dropout = nn.Dropout(dropout)

        ########################
        ## Convolutional Layer
        ########################

        self.multi_kernel_sizes = multi_kernel_sizes
        if multi_kernel_sizes:
            kernel_size = 5
            convolutions = []
            for _ in range(cnn_layers):
                convolutions.append(
                    nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=kernel_size, padding='same')
                )
                kernel_size += 2

            self.convolutions = nn.ModuleList(convolutions)
            self.pool = nn.MaxPool1d(kernel_size=kernel_size_maxpool)
            current_dims = filters * cnn_layers
        else:
            self.filters = filters
            self.residual_blocks = residual_blocks
            self.intermediate_filters = 128
            if residual_blocks:
                self.cnn_layers = nn.Sequential(
                    ResidualBlock1D(embedding_dim, embedding_dim),
                    ResidualBlock1D(embedding_dim, self.intermediate_filters, 2),
                    ResidualBlock1D(self.intermediate_filters, self.intermediate_filters),
                    ResidualBlock1D(self.intermediate_filters, filters, 2),
                    ResidualBlock1D(filters, filters),
                )
            else:
                self.kernel_size_cnn = kernel_size_cnn
                self.cnn_layers = nn.Sequential(
                    nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=kernel_size_cnn),
                    nn.MaxPool1d(kernel_size=kernel_size_maxpool),
                )
            current_dims = filters

        ########################
        ## Recurrent Layer
        ########################
        self.lstm_dims = lstm_dims
        if lstm_dims:
            self.bi_lstm = nn.LSTM(
                input_size=current_dims,  # Is this dimension? - this should receive output from maxpool
                hidden_size=lstm_dims,
                bidirectional=True,
                bias=True,
                batch_first=True,
                dropout=dropout,
            )
            current_dims = lstm_dims * 2

        if final_layer_dims:
            self.fc1 = nn.Linear(
                in_features=current_dims,
                out_features=final_layer_dims,
            )
            current_dims = final_layer_dims

        #################################
        ## Linear Layer(s) to Predictions
        #################################
        self.final_layer_dims = final_layer_dims
        self.logits = nn.Linear(
            in_features=current_dims,
            out_features=self.num_classes,
            bias=final_bias,
        )

    def forward(self, x):
        ########################
        ## Embedding
        ########################
        # Cast as pytorch tensor
        # x = Tensor(x)

        # Convert to int because it may be simply a byte
        """Returns the logits for a batch of sequences.

        Args:
            x (Tensor): Nucleotide indices of shape (batch, sequence length).

        Returns:
            Tensor: Logits of shape (batch, num_classes).
        """
        x = x.int()
        x = self.embed(x)

        ########################
        ## Convolutional Layer
        ########################
        # Transpose seq_len with embedding dims to suit convention of pytorch CNNs (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)

        if self.multi_kernel_sizes:
            conv_results = [conv(x) for conv in self.convolutions]
            x = torch.cat(conv_results, dim=-2)

            x = self.pool(x)
        else:
            x = self.cnn_layers(x)

        # Current shape: batch, filters, seq_len
        # With batch_first=True, LSTM expects shape: batch, seq, feature
        x = x.transpose(2, 1)

        ########################
        ## Recurrent Layer
        ########################

        # BiLSTM
        if self.lstm_dims:
            output, (h_n, c_n) = self.bi_lstm(x)
            # h_n of shape (num_layers * num_directions, batch, hidden_size)
            # We are using a single layer with 2 directions so the two output vectors are
            # [0,:,:] and [1,:,:]
            # [0,:,:] -> considers the first index from the first dimension
            x = torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=-1)
        else:
            # if there is no recurrent layer then simply sum over sequence dimension
            x = torch.sum(x, dim=1)

        #################################
        ## Linear Layer(s) to Predictions
        #################################
        # Ignore if the final_layer_dims is empty
        if self.final_layer_dims:
            x = F.relu(self.fc1(x))
        # Get logits. The cross-entropy loss optimisation function just takes in the logits and automatically does a softmax
        out = self.logits(x)

        return out


class ConvClassifier(nn.Module):
    """A convolutional classifier with an optional transformer encoder or bidirectional LSTM.

    The output is split according to `output_types` (e.g. for hierarchical classification).
    """

    def __init__(
        self,
        embedding_dim=8,
        cnn_layers=6,
        output_types=None,
        cnn_dims_start=64,
        kernel_size_maxpool=2,
        return_embeddings: bool = False,
        num_embeddings=5,  # i.e. the size of the vocab which is N, A, C, G, T
        kernel_size=3,
        factor=2,
        padding="same",
        padding_mode="zeros",
        dropout=0.5,
        final_bias=True,
        lstm_dims: int = 0,
        penultimate_dims: int = 1028,
        include_length: bool = False,
        length_scaling: float = 3_000.0,
        transformer_heads: int = 8,
        transformer_layers: int = 6,
    ):
        """Creates the layers of the classifier.

        Args:
            embedding_dim (int): The size of the nucleotide embeddings.
            cnn_layers (int): The number of convolutional layers.
            output_types: The output data types, which determine the size and splitting of the output.
            cnn_dims_start (int): The number of filters in the first convolution.
            kernel_size_maxpool (int): The size of the max pooling after each convolution.
            return_embeddings (bool): Whether the forward pass also returns the penultimate layer's activations.
            num_embeddings (int): The vocabulary size (N, A, C, G, T).
            kernel_size (int): The kernel size of the convolutions.
            factor (float): The factor by which the number of filters grows with each layer.
            padding (str): The padding of the convolutions.
            padding_mode (str): The padding mode of the convolutions.
            dropout (float): The dropout probability.
            final_bias (bool): Whether the output layer has a bias.
            lstm_dims (int): The hidden size of the LSTM. If zero, then the sequence is averaged instead.
            penultimate_dims (int): The size of the penultimate linear layer.
            include_length (bool): Whether to give the (scaled) sequence length as an extra feature.
            length_scaling (float): The divisor used to scale the sequence length.
            transformer_heads (int): The number of attention heads in the transformer.
            transformer_layers (int): The number of transformer layers. If zero, then no transformer is used.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.cnn_layers = cnn_layers
        self.output_types = output_types
        self.kernel_size_maxpool = kernel_size_maxpool
        self.return_embeddings = return_embeddings

        self.num_embeddings = num_embeddings
        self.kernel_size = kernel_size
        self.factor = factor
        self.dropout = dropout
        self.include_length = include_length
        self.length_scaling = length_scaling
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        in_channels = embedding_dim
        out_channels = cnn_dims_start
        conv_layers = []
        for layer_index in range(cnn_layers):
            conv_layers += [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size_maxpool),
                # nn.Conv1d(
                #     in_channels=out_channels,
                #     out_channels=out_channels,
                #     kernel_size=kernel_size,
                #     stride=kernel_size_maxpool,
                # ),
            ]
            in_channels = out_channels
            out_channels = int(out_channels * factor)

        self.conv = nn.Sequential(*conv_layers)

        if self.transformer_layers:
            self.positional_encoding = PositionalEncoding(d_model=in_channels)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=in_channels, nhead=self.transformer_heads, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
        else:
            self.transformer_encoder = None

        self.lstm_dims = lstm_dims
        if lstm_dims:
            self.bi_lstm = nn.LSTM(
                input_size=in_channels,  # Is this dimension? - this should receive output from maxpool
                hidden_size=lstm_dims,
                bidirectional=True,
                bias=True,
                batch_first=True,
                dropout=dropout,
            )
            current_dims = lstm_dims * 2
        else:
            current_dims = in_channels

        # self.average_pool = nn.AdaptiveAvgPool1d(1)

        current_dims += int(include_length)
        self.penultimate = nn.Linear(in_features=current_dims, out_features=penultimate_dims, bias=True)
        self.final = nn.Linear(
            in_features=penultimate_dims, out_features=total_size(self.output_types), bias=final_bias
        )

    def forward(self, x):
        # Convert to int because it may be simply a byte
        """Returns the predictions for a batch of sequences.

        Args:
            x (Tensor): Nucleotide indices of shape (batch, sequence length).

        Returns:
            The predictions split by output type. If `return_embeddings` is set, then the penultimate layer's
            activations are also returned.
        """
        x = x.int()
        length = x.shape[-1]
        x = self.embedding(x)

        # Transpose seq_len with embedding dims to suit convention of pytorch CNNs (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)

        if hasattr(self, 'transformer_encoder') and self.transformer_encoder:
            x = x.transpose(2, 1)
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            x = x.transpose(1, 2)

        if self.lstm_dims:
            x = x.transpose(2, 1)
            output, (h_n, c_n) = self.bi_lstm(x)
            # h_n of shape (num_layers * num_directions, batch, hidden_size)
            # We are using a single layer with 2 directions so the two output vectors are
            # [0,:,:] and [1,:,:]
            # [0,:,:] -> considers the first index from the first dimension
            x = torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=-1)
        # elif hasattr(x, 'average_pool'):
        #     x = self.average_pool(x)
        #     x = torch.flatten(x, 1)
        else:
            x = torch.mean(x, axis=-1)
            # x = torch.sum(x, axis=-1)

        if getattr(self, 'include_length', False):
            length_tensor = torch.full((x.shape[0], 1), length / self.length_scaling, device=x.device)
            x = torch.cat([x, length_tensor], dim=1)

        penultimate_result = self.penultimate(x)
        predictions = self.final(F.relu(penultimate_result))

        split_predictions = split_tensor(predictions, self.output_types, feature_axis=1)

        if getattr(self, "return_embeddings", False):
            return split_predictions, penultimate_result

        return split_predictions

    def output_types_match(self, output_types) -> bool:
        """Returns whether the final layer of this model is already suitable for the given output types.

        This is the case when the number of outputs is the same and each output type is equivalent to the
        corresponding current one. Hierarchical outputs are equivalent if their classification trees have
        the same nodes (by name and ancestry) in the same order, because the final layer's weights are
        specific to each node.

        Args:
            output_types: The output types to compare to the current output types.
        """
        current = self.output_types
        if current is None or output_types is None:
            return False

        if len(current) != len(output_types) or total_size(current) != total_size(output_types):
            return False

        if self.final.out_features != total_size(output_types):
            return False

        return all(
            type(old) is type(new) and _output_type_signature(old) == _output_type_signature(new)
            for old, new in zip(current, output_types)
        )

    def replace_output_types(self, output_types, final_bias: bool = None) -> None:
        """Replaces the final layer so that the model has outputs for new output types.

        Args:
            output_types: The new output types.
            final_bias (bool | None): Whether the new final layer has a bias. If None, then it matches the current layer.
        """
        device = next(self.parameters()).device
        self.output_types = output_types
        penultimate_dims = self.penultimate.out_features
        if final_bias is None:
            final_bias = self.final.bias is not None

        self.final = nn.Linear(
            in_features=penultimate_dims, out_features=total_size(self.output_types), bias=final_bias, device=device
        )


def _output_type_signature(output_type):
    """Returns a value which is equal for output types that are equivalent.

    For types with a classification tree (a `root`), this is the path of names to each node in the tree
    in pre-order. Otherwise it is the output type itself.
    """
    root = getattr(output_type, "root", None)
    if root is None:
        return output_type
    return tuple(tuple(str(n.name) for n in (node, *node.ancestors)) for node in root.pre_order_iter())


class SequentialDebug(nn.Sequential):
    """A sequential module which prints the shapes and multiply-accumulate operations of each layer."""

    def forward(self, input):
        """Runs each module, printing its shape and the cumulative MACs. Requires `thop`."""
        macs_cummulative = 0
        from thop import profile

        console.print(f"Input shape {input.shape}")
        for module in self:
            console.print(f"Module: {module} ({type(module)})")
            macs, _ = profile(module, inputs=(input,))
            macs_cummulative += int(macs)
            console.print(f"MACs: {int(macs)} (cummulative {macs_cummulative})")

            input = module(input)
            console.print(f"Output shape: {input.shape}")

        return input


class ConvProcessor(nn.Sequential):
    """A stack of convolution, ReLU, dropout and max pooling layers which increase the number of channels."""

    def __init__(
        self,
        in_channels=8,
        cnn_layers=6,
        cnn_dims_start=64,
        kernel_size_maxpool=2,
        kernel_size=3,
        factor=2,
        dropout=0.5,
        padding="same",
        padding_mode="zeros",
    ):
        """Creates the layers.

        Args:
            in_channels (int): The number of input channels.
            cnn_layers (int): The number of convolutional layers.
            cnn_dims_start (int): The number of filters in the first convolution.
            kernel_size_maxpool (int): The size of the max pooling after each convolution.
            kernel_size (int): The kernel size of the convolutions.
            factor (float): The factor by which the number of filters grows with each layer.
            dropout (float): The dropout probability.
            padding (str): The padding of the convolutions.
            padding_mode (str): The padding mode of the convolutions.
        """
        out_channels = cnn_dims_start
        conv_layers = []
        for layer_index in range(cnn_layers):
            conv_layers += [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size_maxpool),
                # nn.Conv1d(
                #     in_channels=out_channels,
                #     out_channels=out_channels,
                #     kernel_size=kernel_size,
                #     stride=kernel_size_maxpool,
                # ),
            ]
            in_channels = out_channels
            out_channels = int(out_channels * factor)

        super().__init__(*conv_layers)


def calc_cnn_dims_start(
    macc,
    seq_len: int,
    embedding_dim: int,
    cnn_layers: int,
    kernel_size: int,
    factor: float,
    penultimate_dims: int,
    num_classes: int,
):
    r"""
    Solving equation M = s k e c + \sum_{l=1}^{L-1} \frac{s}{2^{l} } k c^2 f^{2l-1} + c f^{L-1} p + p o
    for c.

    Args:
        macc_per_base (int): the number of multiply-accumulate operations per base pair in the sequence.
        embedding_dim (int): The size of the embedding.
        cnn_layers (int): The number of CNN layers.
        kernel_size (int): The size of the kernel in the CNN
        factor (float): The multiplying factor for the CNN output layers.
    """
    b = kernel_size * embedding_dim * seq_len + factor ** (cnn_layers - 1) * penultimate_dims
    c = penultimate_dims * num_classes - macc

    if cnn_layers == 1:
        cnn_dims_start = -c / b
    else:
        a = 0.0
        for layer_index in range(1, cnn_layers):
            a += seq_len * kernel_size * (0.5**layer_index) * (factor ** (2 * layer_index - 1))

        cnn_dims_start = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    return int(cnn_dims_start + 0.5)
