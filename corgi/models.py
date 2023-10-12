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
        x = x + self.pe[:x.size(0)]
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
    def __init__(
        self,
        embedding_dim=8,
        cnn_layers=6,
        output_types=None,
        cnn_dims_start=64,
        kernel_size_maxpool=2,
        return_penultimate:bool=False,
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
        length_scaling:float = 3_000.0,
        transformer_heads: int = 8,
        transformer_layers: int = 6,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.cnn_layers = cnn_layers
        self.output_types = output_types
        self.kernel_size_maxpool = kernel_size_maxpool
        self.return_penultimate = return_penultimate

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
            encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=self.transformer_heads, batch_first=True)
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

        self.average_pool = nn.AdaptiveAvgPool1d(1)

        current_dims += int(include_length)
        self.penultimate = nn.Linear(in_features=current_dims, out_features=penultimate_dims, bias=True)
        self.final = nn.Linear(in_features=penultimate_dims, out_features=total_size(self.output_types), bias=final_bias)

    def forward(self, x):
        # Convert to int because it may be simply a byte
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
        elif hasattr(x, 'average_pool'):
            x = self.average_pool(x)
            x = torch.flatten(x, 1)
        else:
            x = torch.mean(x, axis=-1)

        if getattr(self, 'include_length', False):
            length_tensor = torch.full( (x.shape[0], 1), length/self.length_scaling, device=x.device )
            x = torch.cat([x, length_tensor], dim=1)

        penultimate_result = self.penultimate(x)
        predictions = self.final(F.relu(penultimate_result))

        split_predictions = split_tensor(predictions, self.output_types, feature_axis=1)

        if self.return_penultimate:
            return split_predictions, penultimate_result
    
        return split_predictions

    def replace_output_types(self, output_types, final_bias:bool=None) -> None:
        self.output_types = output_types
        penultimate_dims = self.penultimate.out_features
        if final_bias is None:
            final_bias = self.final.bias
        
        self.final = nn.Linear(in_features=penultimate_dims, out_features=total_size(self.output_types), bias=final_bias)
        


class SequentialDebug(nn.Sequential):
    def forward(self, input):
        macs_cummulative = 0
        from thop import profile

        console.print(f"Input shape {input.shape}")
        for module in self:
            console.print(f"Module: {module} ({type(module)})")
            macs, _ = profile(module, inputs=(input, ))
            macs_cummulative += int(macs)
            console.print(f"MACs: {int(macs)} (cummulative {macs_cummulative})")

            input = module(input)
            console.print(f"Output shape: {input.shape}")

        return input



class ConvProcessor(nn.Sequential):
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
    seq_len:int,
    embedding_dim:int,
    cnn_layers:int,
    kernel_size:int,
    factor:float,
    penultimate_dims: int,
    num_classes: int,
):
    """
    Solving equation M = s k e c + \sum_{l=1}^{L-1} \frac{s}{2^{l} } k c^2 f^{2l-1} + c f^{L-1} p + p o
    for c.

    Args:
        macc_per_base (int): the number of multiply-accumulate operations per base pair in the sequence.
        embedding_dim (int): The size of the embedding.
        cnn_layers (int): The number of CNN layers.
        kernel_size (int): The size of the kernel in the CNN
        factor (float): The multiplying factor for the CNN output layers.
    """
    b = kernel_size * embedding_dim * seq_len + factor ** (cnn_layers-1) * penultimate_dims
    c = penultimate_dims * num_classes - macc

    if cnn_layers == 1:
        cnn_dims_start = -c/b
    else:        
        a = 0.0
        for layer_index in range(1, cnn_layers):
            a += seq_len * kernel_size * (0.5**layer_index) * (factor**(2 * layer_index - 1))
        
        cnn_dims_start = (-b + np.sqrt(b**2 - 4*a*c))/(2 * a)

    return int(cnn_dims_start + 0.5)