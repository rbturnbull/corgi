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
