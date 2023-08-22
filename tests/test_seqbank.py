import tempfile
from pathlib import Path
from corgi.seqbank import SeqBank
import numpy as np

test_data = Path(__file__).parent / "testdata"

def test_seqbank_add():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        seqbank = SeqBank(path=tmpdirname/"seqbank.h5")
        seqbank.add("ATCG", "test")
        assert "test" in seqbank
        assert np.all(seqbank["test"] == np.array([1, 4, 2, 3], dtype="u1"))


def test_seqbank_add_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        seqbank = SeqBank(path=tmpdirname/"seqbank.h5")
        # seqbank = SeqBank(path=test_data/"seqbank.h5")

        seqbank.add_file(test_data/"NC_024664.1.trunc.fasta")
        assert "NC_024664.1" in seqbank
        assert len(seqbank["NC_024664.1"]) == 840

        seqbank.add_file(test_data/"NZ_JAJNFP010000161.1.fasta")
        assert len(seqbank["NZ_JAJNFP010000161.1"]) == 849

        seqbank.add_file(test_data/"NC_036112.1.fasta")
        assert len(seqbank["NC_036112.1"]) == 980
        
        seqbank.add_file(test_data/"NC_044840.1.fasta")
        assert len(seqbank["NC_044840.1"]) == 700

        seqbank.add_file(test_data/"NC_010663.1.gb", format="genbank")
        assert len(seqbank["NC_010663.1"]) == 1066

        seqbank.add_file(test_data/"NC_036113.1.fasta")
        assert len(seqbank["NC_036113.1"]) == 1050

        

        
    
    