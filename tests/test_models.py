import unittest
import torch
from polytorch import CategoricalData

from corgi import models


class TestConvAttentionPooling(unittest.TestCase):
    def test_matches_linear_scoring(self):
        pool = models.AttentionPooling(attention_hidden_size=4)
        x = torch.randn(2, 8, 13)
        actual = pool(x)
        first, activation, last = pool.attention_layer
        positions = x.transpose(1, 2)
        hidden = activation(torch.nn.functional.linear(
            positions, first.weight.squeeze(-1), first.bias,
        ))
        scores = torch.nn.functional.linear(hidden, last.weight.squeeze(-1), last.bias)
        expected = (scores.softmax(dim=1) * positions).sum(dim=1)
        torch.testing.assert_close(actual, expected)

    def test_nan_padding_and_gradients(self):
        pool = models.AttentionPooling(attention_hidden_size=4)
        x = torch.randn(2, 8, 13)
        expected = pool(x)
        padded = torch.cat((x, torch.full((2, 8, 3), torch.nan)), dim=-1)
        padded.requires_grad_()
        actual = pool(padded)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(padded.grad).all())
        for parameter in pool.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        with self.assertRaisesRegex(ValueError, "at least one valid position"):
            pool(torch.full((2, 8, 13), torch.nan))

    def test_variable_lengths_and_gradients(self):
        for transformer_layers in (0, 1):
            for include_length in (False, True):
                with self.subTest(transformer_layers=transformer_layers,
                                  include_length=include_length):
                    model = models.ConvClassifier(
                        cnn_layers=1, cnn_dims_start=8,
                        transformer_layers=transformer_layers, transformer_heads=2,
                        output_types=[CategoricalData(3)],
                        attention_pooling=True, attention_pooling_dims=4,
                        penultimate_dims=6, dropout=0, include_length=include_length,
                    )
                    # Neither pooled length equals the channel count; reuse
                    # the initialized pooling weights across sequence lengths.
                    for length in (26, 34):
                        model.zero_grad(set_to_none=True)
                        predictions, = model(torch.ones(2, length, dtype=torch.long))
                        self.assertEqual(predictions.shape, (2, 3))
                        self.assertTrue(torch.isfinite(predictions).all())
                        predictions.square().sum().backward()
                        for parameter in model.pool.parameters():
                            self.assertIsNotNone(parameter.grad)
                            self.assertTrue(torch.isfinite(parameter.grad).all())


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
