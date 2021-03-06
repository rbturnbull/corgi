from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
    ):
        super().__init__()

        num_embeddings = 5  # i.e. the size of the vocab which is N, A, C, G, T

        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.dropout = dropout

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

        # self.filters = filters
        # self.residual_blocks = residual_blocks
        # self.intermediate_filters = 128
        # if residual_blocks:
        #     self.cnn_layers = nn.Sequential(
        #         ResidualBlock1D(embedding_dim, embedding_dim),
        #         ResidualBlock1D(embedding_dim, self.intermediate_filters, 2),
        #         ResidualBlock1D(self.intermediate_filters, self.intermediate_filters),
        #         ResidualBlock1D(self.intermediate_filters, filters, 2),
        #         ResidualBlock1D(filters, filters),
        #     )
        # else:
        #     self.kernel_size_cnn = kernel_size_cnn
        #     self.cnn_layers = nn.Sequential(
        #         nn.Conv1d( in_channels=embedding_dim, out_channels=filters, kernel_size=kernel_size_cnn),
        #         nn.MaxPool1d(kernel_size=kernel_size_maxpool),
        #     )
        # current_dims = filters

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
        # breakpoint()

        # Convert to int because it may be simply a byte
        x = x.int()
        x = self.embed(x)

        ########################
        ## Convolutional Layer
        ########################
        # Transpose seq_len with embedding dims to suit convention of pytorch CNNs (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)

        conv_results = [conv(x) for conv in self.convolutions]
        x = torch.cat(conv_results, dim=-2)

        x = self.pool(x)

        # x = self.cnn_layers(x)
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
