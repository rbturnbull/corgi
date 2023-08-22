from typing import Union
import numpy as np
from pathlib import Path
import h5py
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from zlib import adler32
from attrs import define

from .tensor import TensorDNA, dna_seq_to_numpy

class SeqBankError(Exception):
    pass


@define
class SeqBank():
    path:Path
    read_h5:object = None
    write_h5:object = None
    
    def __getstate__(self):
        # Only returns required elements
        # Needed because h5 files cannot be pickled
        return dict(path=self.path)

    def key(self, accession:str) -> str:
        # Using adler32 for a fast deterministic hash
        accession_hash = str(adler32(accession.encode('ascii')))
        return f"/{accession_hash[-6:-3]}/{accession_hash[-3:]}/{accession}"

    def get_read_h5(self):
        if not self.read_h5:
            self.read_h5 = h5py.File(self.path, "r")
        return self.read_h5

    def __getitem__(self, accession:str) -> TensorDNA:
        try:
            key = self.key(accession)
            file = self.get_read_h5()
            return file[key][:]
        except Exception as err:
            raise SeqBankError(f"Failed to read {accession} in SeqBank {self.path}:\n{err}")

    def __contains__(self, accession:str) -> bool:
        return self.key(accession) in self.get_read_h5()

    def add(self, seq:Union[str, Seq, SeqRecord, np.ndarray], accession:str):
        key = self.key(accession)

        if not self.write_h5:
            self.write_h5 = h5py.File(self.path, "a")

        if key in self.write_h5:
            return self.write_h5[key]
        
        if isinstance(seq, SeqRecord):
            seq = seq.seq
        if isinstance(seq, Seq):
            seq = str(seq)
        if isinstance(seq, str):
            seq = dna_seq_to_numpy(seq)
        
        return self.write_h5.create_dataset(
            self.key(accession),
            data=seq,
            dtype="u1",
            compression="gzip",
            compression_opts=9,
        )

    def add_file(self, path:Path, format="fasta"):
        with open(path) as f:
            for record in SeqIO.parse(f, format):
                self.add(record, record.id)
