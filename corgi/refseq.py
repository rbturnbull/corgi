from zlib import adler32
import time
from dataclasses import dataclass
from os import access
import urllib.request
import requests
import humanize
import sys
import gzip
from Bio import SeqIO
import re
import progressbar
import h5py
import pandas as pd
import asyncio
import httpx
from appdirs import user_data_dir
from bs4 import BeautifulSoup

from pathlib import Path
from . import tensor


def open_gzip(path:Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


REFSEQ_CATEGORIES = [
    "archaea",  # prokaryotic
    "bacteria",  # prokaryotic
    "fungi",  # eukaryotic
    "invertebrate",  # eukaryotic
    "mitochondrion",  # organellar
    "plant",  # eukaryotic
    "plasmid",
    "plastid",  # organellar
    "protozoa",  # eukaryotic
    "vertebrate_mammalian",  # eukaryotic
    "vertebrate_other",  # eukaryotic
    "viral",  # viral
]

PROKARYOTIC = (
    "archaea",
    "bacteria",
)
EUKARYOTIC = (
    "fungi",
    "invertebrate",
    "plant",
    "protozoa",
    "vertebrate_mammalian",
    "vertebrate_other",
)
ORGANELLAR = (
    "mitochondrion",
    "plastid",
)


def root_dir():
    """
    Returns the path to the root directory of this project.

    This is useful for finding the data directory.
    """
    return Path(__file__).parent.resolve()


def global_data_dir():
    """Returns the path to the directory to hold all the data from the VBA website."""
    return root_dir() / "data"


def filesize_readable(path: (str, Path)) -> str:
    path = Path(path)
    if not path.exists():
        return f"File {path} does not exist."
    return humanize.naturalsize(path.stat().st_size)


@dataclass
class RefSeqCategory:
    name: str
    max_files: int = None
    base_dir: str = global_data_dir()

    def __getstate__(self):
        # Only returns required elements
        # Needed because h5 files cannot be pickled
        return dict(name=self.name, max_files=self.max_files, base_dir=self.base_dir)

    def base_url(self):
        return f"https://ftp.ncbi.nlm.nih.gov/refseq/release/{self.name}"

    def filename(self, index) -> str:
        """The filename in the RefSeq database for this index."""
        if self.max_files:
            assert index < self.max_files

        return f"{self.name}.{index+1}.1.genomic.fna.gz"

    def fasta_url(self, index: int) -> str:
        """The url for the fasta file for this index online."""
        return f"{self.base_url()}/{self.filename(index)}"

    def index_html(self):
        index_path = Path(user_data_dir()) / f"{self.name}.index.html"
        if not index_path.exists():
            url = self.base_url()
            print("Downloading:", url)
            urllib.request.urlretrieve(url, index_path)
        with open(index_path, 'r') as f:
            contents = f.read()
            soup = BeautifulSoup(contents, 'html.parser')
        return soup

    async def download(self, index: int):
        local_path = self.fasta_path(index, download=False)

        if not local_path.exists():
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            async with httpx.AsyncClient(limits=limits) as client:
                url = self.fasta_url(index)
                print(f"downloading {url}")
                response = await client.get(url)
                open(local_path, 'wb').write(response.content)
                print(f"done {local_path}")
        return local_path

    async def download_all_async(self):
        paths = []
        max_files = self.max_files_available()
        if self.max_files:
            max_files = min(max_files, self.max_files)
        print(f"max_files = {max_files}")
        for index in range(max_files):
            paths.append(self.download(index))
        await asyncio.gather(*paths)

    def download_all(self):
        max_files = self.max_files_available()
        if self.max_files:
            max_files = min(max_files, self.max_files)
        print(f"max_files = {max_files}")
        for index in range(max_files):
            print(f"{self.name} - {index}")
            try:
                self.fasta_path(index, download=True)
            except Exception:
                print(f"failed to download. {self.name} file {index}")

    def fasta_path(self, index: int, download=True) -> Path:
        """
        Returns the local path for the file at this index.

        If the file does not exist already in the base_dir then it is downloaded.
        """
        local_path = Path(self.base_dir) / self.name / self.filename(index)
        local_path.parent.mkdir(exist_ok=True, parents=True)
        if download and not local_path.exists():
            url = self.fasta_url(index)
            print("Downloading:", url)
            urllib.request.urlretrieve(url, local_path)
        return local_path

    def h5_path(self):
        base_dir = Path(self.base_dir)
        base_dir.mkdir(exist_ok=True, parents=True)
        return base_dir / f"{self.name}.h5"

    def fasta_seq_count(self, index: int) -> int:
        fasta_path = self.fasta_path(index)
        seq_count = 0
        with gzip.open(fasta_path, "rt") as fasta:
            for line in fasta:
                if line.startswith(">"):
                    seq_count += 1
        return seq_count

    def get_seq(self, accession):
        if not hasattr(self, 'read_h5'):
            self.read_h5 = h5py.File(self.h5_path(), "r")
        try:
            return self.read_h5[self.dataset_key(accession)]
        except Exception:
            raise Exception(f"Failed to read {accession} in {self.name}")
            # print(f"Failed to read {accession} in {self.name}")
            # return []

    def dataset_key(self, accession):
        # Using adler32 for a fast deterministic hash
        accession_hash = str(adler32(accession.encode('ascii')))
        return f"/{accession_hash[-6:-3]}/{accession_hash[-3:]}/{accession}"

    def max_files_available(self):
        max_files = 0
        soup = self.index_html()
        for link in soup.findAll("a"):
            m = re.match(self.name + r"\.(\d+)\.1\.genomic\.fna\.gz", link.get("href"))
            if m:
                max_files = max(int(m.group(1)), max_files)
        return max_files

    def get_accessions(self):
        accessions = set()

        with h5py.File(self.h5_path(), "a") as h5:
            for key0 in h5.keys():
                for key1 in h5[f"/{key0}"].keys():
                    dir_accessions = h5[f"/{key0}/{key1}"].keys()
                    # for accession in dir_accessions:
                    #     print(accession)
                    accessions.update(dir_accessions)
        return accessions

    def write_h5(self, show_bar=True, file_indexes=None):
        result = []
        if not sys.stdout.isatty():
            show_bar = False

        if file_indexes is None or len(file_indexes) == 0:
            max_files = self.max_files_available()
            if self.max_files:
                max_files = min(max_files, self.max_files)

            file_indexes = range(max_files)

        accessions = self.get_accessions()
        print(f"{len(accessions)} sequences already in HDF5 file for {self.name}.")
        with h5py.File(self.h5_path(), "a") as h5:
            for file_index in file_indexes:
                print(f"Preprocessing file {file_index} from {self.name}", flush=True)
                # Try to get next file
                try:
                    fasta_path = self.fasta_path(file_index)
                    seq_count = self.fasta_seq_count(file_index)
                except Exception:
                    # If it fails, then assume it doesn't exist and exit
                    print(f"Fasta file at index {file_index} for {self.name} not found.")
                    continue

                result.extend(self.import_fasta(fasta_path, h5, show_bar=True))
                print()

        df = pd.DataFrame(result)
        return df

    def h5_filesize(self) -> str:
        return filesize_readable(self.h5_path())

    def fasta_filesize(self, index) -> str:
        return filesize_readable(self.fasta_path(index))

    def total_fasta_filesize(self) -> str:
        size = sum(self.fasta_path(i).stat().st_size for i in range(self.max_files))
        return humanize.naturalsize(size)

    def total_fasta_filesize_server_bytes(self) -> int:
        size = 0
        for index in range(self.max_files):
            url = self.fasta_url(index)
            response = requests.head(url, allow_redirects=True)
            size += int(response.headers.get("content-length", 0))
        return size

    def total_fasta_filesize_server(self) -> str:
        return humanize.naturalsize(self.total_fasta_filesize_server_bytes())

    def add_ncbi_datasets(self, accessions, base_dir):
        results = []
        all_accessions = self.get_accessions()

        with h5py.File(self.h5_path(), "a") as h5:
            for accession in accessions:
                path = base_dir / "ncbi_dataset/data" / accession
                for fasta_path in path.glob("*.fna"):
                    print(fasta_path)
                    results.extend(
                        self.import_fasta(fasta_path, h5, show_bar=True, accessions=all_accessions, use_gzip=False)
                    )

        df = pd.DataFrame(results)
        return df

    def add_individual_accessions(self, accessions, email=None):
        results = []
        all_accessions = self.get_accessions()

        with h5py.File(self.h5_path(), "a") as h5:
            for accession in accessions:
                fasta_path = self.individual_accession_path(accession, email=email)
                if fasta_path:
                    results.extend(self.import_fasta(fasta_path, h5, show_bar=False, accessions=all_accessions))

        df = pd.DataFrame(results)
        return df

    def individual_accession_path(self, accession: str, download: bool = True, email=None) -> Path:
        local_path = Path(self.base_dir) / self.name / "individual" / f"{accession}.fa.gz"
        local_path.parent.mkdir(exist_ok=True, parents=True)
        db = "nucleotide"
        if download and not local_path.exists():
            from Bio import Entrez

            if email:
                Entrez.email = email
            else:
                raise Exception("no email provided")

            print(f"Trying to download '{accession}'")
            try:
                print("trying nucleotide database")
                net_handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
            except Exception as err:
                print(f'failed {err}')
                print("trying genome database")
                time.sleep(3)
                try:
                    net_handle = Entrez.efetch(db="genome", id=accession, rettype="fasta", retmode="text")
                except Exception as err:
                    print(f'failed {err}')
                    print("trying nuccore database")
                    try:
                        net_handle = Entrez.efetch(db="nuccore", id=accession, rettype="fasta", retmode="text")
                    except:
                        print(f'failed {err}')
                        return None

            with gzip.open(local_path, "wt") as f:
                f.write(net_handle.read())
            net_handle.close()
        return local_path

    def import_fasta(self, fasta_path: Path, h5, show_bar: bool = True, accessions=None, use_gzip=True):
        accessions = accessions or self.get_accessions()
        seq_count = 0
        with open_fasta(fasta_path, use_gzip) as fasta:
            for line in fasta:
                if line.startswith(">"):
                    seq_count += 1

        is_mitochondria = self.name.lower().startswith("mitochondr")
        is_plastid = self.name.lower().startswith("plastid")

        with open_fasta(fasta_path, use_gzip) as fasta:
            result = []
            if show_bar:
                bar = progressbar.ProgressBar(max_value=seq_count - 1)
            seqs = SeqIO.parse(fasta, "fasta")
            for i, seq in enumerate(seqs):
                dataset_key = self.dataset_key(seq.name)
                description = seq.description.lower()

                if not is_mitochondria and "mitochondr" in description:
                    continue

                if not is_plastid and (
                    "plastid" in description
                    or "chloroplast" in description
                    or "apicoplast" in description
                    or "kinetoplast" in description
                ):
                    continue

                if not is_plastid and not is_mitochondria and "organelle" in description:
                    continue

                # Check if we already have this dataset. If not then add.
                if not seq.name in accessions:
                    t = tensor.dna_seq_to_numpy(seq.seq)
                    dset = h5.create_dataset(
                        dataset_key,
                        data=t,
                        dtype="u1",
                        compression="gzip",
                        compression_opts=9,
                    )
                    accessions.add(seq.name)

                result.append(dict(category=self.name, accession=seq.name))
                if i % 20 == 0:
                    if show_bar:
                        bar.update(i)
            if show_bar:
                bar.update(i)

        return result
