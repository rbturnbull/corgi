import tempfile
from pathlib import Path
from corgi.seqbank import SeqBank
import numpy as np


def test_seqbank_add():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        seqbank = SeqBank(path=tmpdirname/"seqbank.h5")
        seqbank.add("ATCG", "test")
        assert "test" in seqbank
        assert np.all(seqbank["test"] == np.array([1, 4, 2, 3], dtype="u1"))