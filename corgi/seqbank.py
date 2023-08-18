from pathlib import Path
import h5py
from zlib import adler32
from attrs import define

from .tensor import TensorDNA

class SeqBankError(Exception):
    pass


@define
class SeqBank():
    path:Path
    read_h5:object = None
    
    def __getstate__(self):
        # Only returns required elements
        # Needed because h5 files cannot be pickled
        return dict(path=self.path)

    def key(self, accession:str) -> str:
        # Using adler32 for a fast deterministic hash
        accession_hash = str(adler32(accession.encode('ascii')))
        return f"/{accession_hash[-6:-3]}/{accession_hash[-3:]}/{accession}"

    def __getitem__(self, accession:str) -> TensorDNA:
        if not hasattr(self, 'read_h5'):
            self.read_h5 = h5py.File(self.path, "r")

        try:
            return self.read_h5[self.key(accession)]
        except Exception as err:
            raise SeqBankError(f"Failed to read {accession} in SeqBank {self.path}:\n{err}")
