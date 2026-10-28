import unittest
import torch

from corgi import models


class TestModels(unittest.TestCase):
    def setUp(self):
        self.model = models.ConvRecurrantClassifier(5)

    def test_model_str(self):
        model_str = str(self.model)
        self.assertIn("Embedding", model_str)
        self.assertIn("LSTM", model_str)
        self.assertIn("Dropout", model_str)
        self.assertIn("Conv1d", model_str)
        self.assertIn("MaxPool1d", model_str)
        self.assertIn("Linear", model_str)

    def test_model_output(self):
        x = torch.ones((64, 100), dtype=torch.uint8) # batch, seq_len
        # x = tensor.TensorDNA(x) 
        y = self.model(x)
        self.assertEqual(y.shape, (64, 5))

    def test_lstm_none(self):
        model = models.ConvRecurrantClassifier(5, lstm_dims=0)        
        x = torch.ones((64, 100), dtype=torch.uint8) # batch, seq_len
        y = model(x)
        self.assertEqual(y.shape, (64, 5))


from pathlib import Path
import pytest
from polytorch import HierarchicalData, total_size
from corgi.seqtree import SeqTree


def hierarchical_output_types():
    seqtree = SeqTree.load(Path(__file__).parent / "testdata/seqtree.pkl")
    return [HierarchicalData(root=seqtree.classification_tree)]


def test_positional_encoding():
    encoding = models.PositionalEncoding(d_model=8, dropout=0.0, max_len=50)
    x = torch.zeros(10, 3, 8)
    y = encoding(x)
    assert y.shape == (10, 3, 8)
    assert torch.equal(y[:, 0], encoding.pe[:10, 0])


def test_residual_block_same_size():
    block = models.ResidualBlock1D(4, 4)
    x = torch.randn(2, 4, 16)
    assert block(x).shape == (2, 4, 16)


def test_residual_block_stride_downsamples():
    block = models.ResidualBlock1D(4, 8, stride=2)
    x = torch.randn(2, 4, 16)
    assert block(x).shape == (2, 8, 8)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(multi_kernel_sizes=False),
        dict(multi_kernel_sizes=False, residual_blocks=True),
        dict(multi_kernel_sizes=False, lstm_dims=0),
        dict(multi_kernel_sizes=True, cnn_layers=2, filters=8, lstm_dims=8),
        dict(multi_kernel_sizes=False, final_layer_dims=16, final_bias=False),
    ],
)
def test_conv_recurrent_classifier_variants(kwargs):
    kwargs = {"filters": 8, "lstm_dims": 8, **kwargs}
    model = models.ConvRecurrantClassifier(3, **kwargs)
    x = torch.randint(0, 5, (4, 64), dtype=torch.uint8)
    assert model(x).shape == (4, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(lstm_dims=8, transformer_layers=0),
        dict(include_length=True, transformer_layers=0),
        dict(transformer_layers=1, transformer_heads=2),
    ],
)
def test_conv_classifier_variants(kwargs):
    output_types = hierarchical_output_types()
    model = models.ConvClassifier(
        output_types=output_types,
        cnn_layers=2,
        cnn_dims_start=8,
        penultimate_dims=16,
        **{"transformer_layers": 0, **kwargs},
    )
    x = torch.randint(0, 5, (3, 32), dtype=torch.uint8)
    result = model(x)
    assert isinstance(result, (list, tuple))
    assert sum(r.shape[1] for r in result) == total_size(output_types)
    assert all(r.shape[0] == 3 for r in result)


def test_conv_classifier_return_embeddings():
    model = models.ConvClassifier(
        output_types=hierarchical_output_types(),
        cnn_layers=2,
        cnn_dims_start=8,
        penultimate_dims=16,
        transformer_layers=0,
        return_embeddings=True,
    )
    result, embeddings = model(torch.randint(0, 5, (3, 32), dtype=torch.uint8))
    assert embeddings.shape == (3, 16)


def test_conv_classifier_replace_output_types():
    model = models.ConvClassifier(
        output_types=hierarchical_output_types(),
        cnn_layers=1,
        cnn_dims_start=8,
        penultimate_dims=16,
        transformer_layers=0,
    )
    new_output_types = hierarchical_output_types()
    model.replace_output_types(new_output_types, final_bias=True)
    assert model.final.out_features == total_size(new_output_types)
    assert model.output_types is new_output_types

    model.replace_output_types(new_output_types)
    assert model.final.out_features == total_size(new_output_types)


def test_conv_processor():
    processor = models.ConvProcessor(in_channels=4, cnn_layers=2, cnn_dims_start=8)
    x = torch.randn(2, 4, 32)
    assert processor(x).shape == (2, 16, 8)


def test_sequential_debug():
    pytest.importorskip("thop")
    module = models.SequentialDebug(torch.nn.Linear(4, 3), torch.nn.ReLU())
    assert module(torch.randn(2, 4)).shape == (2, 3)


def test_calc_cnn_dims_start():
    kwargs = dict(seq_len=1000, embedding_dim=8, kernel_size=3, factor=2, penultimate_dims=1028, num_classes=10)
    single = models.calc_cnn_dims_start(5_000_000, cnn_layers=1, **kwargs)
    multi = models.calc_cnn_dims_start(5_000_000, cnn_layers=6, **kwargs)
    assert isinstance(single, int) and single > 0
    assert isinstance(multi, int) and multi > 0
    # A larger compute budget allows more filters
    assert models.calc_cnn_dims_start(10_000_000, cnn_layers=6, **kwargs) > multi


def small_tree(*names):
    from hierarchicalsoftmax import SoftmaxNode

    root = SoftmaxNode("root")
    for name in names:
        SoftmaxNode(name, parent=root)
    root.set_indexes_if_unset()
    return root


def small_model(root):
    return models.ConvClassifier(
        output_types=[HierarchicalData(root=root)],
        cnn_layers=1,
        cnn_dims_start=8,
        penultimate_dims=16,
        transformer_layers=0,
    )


def test_output_types_match_same_tree():
    model = small_model(small_tree("a", "b"))
    assert model.output_types_match([HierarchicalData(root=small_tree("a", "b"))])


def test_output_types_match_different_names_same_size():
    model = small_model(small_tree("a", "b"))
    assert not model.output_types_match([HierarchicalData(root=small_tree("a", "c"))])


def test_output_types_match_different_order():
    model = small_model(small_tree("a", "b"))
    assert not model.output_types_match([HierarchicalData(root=small_tree("b", "a"))])


def test_output_types_match_different_size():
    model = small_model(small_tree("a", "b"))
    assert not model.output_types_match([HierarchicalData(root=small_tree("a", "b", "c"))])


def test_output_types_match_different_number_of_outputs():
    model = small_model(small_tree("a", "b"))
    root = small_tree("a", "b")
    assert not model.output_types_match([HierarchicalData(root=root), HierarchicalData(root=root)])


def test_output_types_match_missing():
    model = small_model(small_tree("a", "b"))
    assert not model.output_types_match(None)
    model.output_types = None
    assert not model.output_types_match([HierarchicalData(root=small_tree("a", "b"))])


def test_output_types_match_final_layer_out_of_sync():
    model = small_model(small_tree("a", "b"))
    model.final = torch.nn.Linear(16, model.final.out_features + 1)
    assert not model.output_types_match([HierarchicalData(root=small_tree("a", "b"))])
