import unittest
import pandas as pd
from fastai.data.block import DataBlock
from fastai.data.core import DataLoaders

from corgi import dataloaders, tensor


def test_dataframe():
    data = [
        [tensor.dna_seq_to_numpy("ACGTACGTACGTACGTACGTACGTACGTACGT"), "bacteria", 0],
        [tensor.dna_seq_to_numpy("CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT"), "mitochondrion", 1],
        [tensor.dna_seq_to_numpy("CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT"), "bacteria", 1],
        [tensor.dna_seq_to_numpy("GGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAA"), "archaea", 0],
        [tensor.dna_seq_to_numpy("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"), "bacteria", 0],
        [tensor.dna_seq_to_numpy("CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT"), "mitochondrion", 0],
        [tensor.dna_seq_to_numpy("GGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAA"), "archaea", 0],
    ]
    return pd.DataFrame(data, columns=['sequence', 'category', 'validation'])

def test_dls():
    df = test_dataframe()
    return dataloaders.create_dataloaders(df, seq_length=4, batch_size=2)


class TestData(unittest.TestCase):
    def test_datablock(self):
        datablock = dataloaders.create_datablock()
        assert type(datablock) == DataBlock

    def test_dataloaders(self):
        dls = test_dls()
        assert type(dls) == DataLoaders
        self.assertEqual( len(dls.train), 2 )
        self.assertEqual( len(dls.valid), 1 )  
        self.assertListEqual( list(dls.vocab), ['archaea', 'bacteria', 'mitochondrion'] )      

    def test_dataloaders_show_batch(self):
        dls = test_dls()
        dls.show_batch()
        # just testing if it runs. TODO capture output
