from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import h5py

from corgi import transforms, tensor, refseq


class TestTransforms(unittest.TestCase):
    def test_model_output_slice(self):
        x = tensor.TensorDNA(np.zeros((100,)))
        transform = transforms.SliceTransform(40)

        y = transform(x)
        self.assertEqual(y.shape, (40,))

    def test_model_output_pad(self):
        x = tensor.TensorDNA(np.zeros((20,)))
        transform = transforms.SliceTransform(60)

        y = transform(x)
        self.assertEqual(y.shape, (60,))

    def test_random_slice_batch(self):
        import random
        random.seed(0)

        def rand_generator():
            return 10
        transform = transforms.RandomSliceBatch(rand_generator)
        batch = [
            (tensor.TensorDNA([1,2,3,4]),0),
            (tensor.TensorDNA([1,2,3,4,1,2,3,4,1,2,3,4,1,1,1]),1),
        ]
        result = transform(batch)

        self.assertEqual( "ACGTNNNNNN [10]", str(result[0][0]) )
        self.assertEqual( "TACGTACGTA [10]", str(result[1][0]) )
        self.assertEqual( len(result), len(batch) )
