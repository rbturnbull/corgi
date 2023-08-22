from pathlib import Path
import unittest
import pandas as pd
import numpy as np
from fastai.data.block import DataBlock
from fastai.data.core import DataLoaders

from corgi import dataloaders, tensor

test_data = Path(__file__).parent / "testdata"

def test_dataframe():
    data = [
        [tensor.dna_seq_to_numpy("ACGTACGTACGTACGTACGTACGTACGTACGT"), "bacteria", 0],
        [
            tensor.dna_seq_to_numpy("CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT"),
            "mitochondrion",
            1,
        ],
        [
            tensor.dna_seq_to_numpy("CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT"),
            "bacteria",
            1,
        ],
        [
            tensor.dna_seq_to_numpy("GGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAA"),
            "archaea",
            0,
        ],
        [
            tensor.dna_seq_to_numpy("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"),
            "bacteria",
            0,
        ],
        [
            tensor.dna_seq_to_numpy("CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT"),
            "mitochondrion",
            0,
        ],
        [tensor.dna_seq_to_numpy("GGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAA"), "archaea", 0],
    ]
    return pd.DataFrame(data, columns=["sequence", "category", "validation"])


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
        self.assertEqual(len(dls.train), 3)
        self.assertEqual(len(dls.valid), 1)
        self.assertListEqual(list(dls.vocab), ["archaea", "bacteria", "mitochondrion"])

    def test_dataloaders_show_batch(self):
        dls = test_dls()
        dls.show_batch()
        # just testing if it runs. TODO capture output


class TestStratifiedDL(unittest.TestCase):
    def test_stratified_dl(self):
        batch_size = 3
        groups = [
            [1,2,3],
            [4,5,6,7,8,9,10],
            list(range(11,20)),
        ]
        dl = dataloaders.StratifiedDL(
            np.arange(20), 
            bs=batch_size,
            groups=groups,
            shuffle=True,
        )
        batches = list(dl)        
        self.assertEqual(len(batches), 3)
        for batch in batches:
            self.assertEqual(batch.shape[0], batch_size)
            for group in groups:
                self.assertEqual( len(set(batch.numpy()) & set(group)), 1 )

        
def test_create_seqbank_dataloaders():
    dls = dataloaders.create_seqbank_dataloaders(
        csv=test_data/"accessions.csv",
        seqbank=test_data/"seqbank.h5",
    )
    assert isinstance(dls, DataLoaders)
    assert isinstance(dls.valid, dataloaders.TfmdDL)
    assert isinstance(dls.train, dataloaders.TfmdDL)
    assert dls.n_inp == 1
    assert dls.classification_tree.render_equal(
        """
        root
        ├── Virus
        ├── Eukaryota
        ├── Plant
        └── Bacteria
        """        
    )
