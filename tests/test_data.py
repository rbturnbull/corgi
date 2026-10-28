import gzip
import pickle
import pytest
from pathlib import Path
import torch
from torch import Tensor
from seqbank import SeqBank
from corgi.seqtree import SeqTree
from corgi.data import (
    Collate,
    CollateRandomLength,
    CollateFixedLength,
    CorgiDataModule,
    CorgiTrainingDataset,
    SeqIODataloader,
    generate_overlapping_intervals,
    seqbank_worker_init,
    slice_tensor,
)


TEST_DATA = Path(__file__).parent / "testdata"


def test_collate_random_lengths():
    collation = CollateRandomLength(minimum_length=150, maximum_length=3_000)
    samples = 2_000
    lengths = torch.tensor([collation.get_length() for _ in range(samples)])
    assert lengths.min() >= 150
    assert lengths.max() <= 3_000
    assert 1300 < lengths.float().mean() < 1400
    
    plot = False
    if plot:
        import plotly.express as px
        fig = px.histogram(lengths, nbins=100)
        fig.show()


def test_collate_random():
    collation = CollateRandomLength(minimum_length=150, maximum_length=3_000)
    batch = [
        (Tensor([1,2,3,4]),0),
        (Tensor([1,2,3,4,1,2,3,4,1,2,3,4,1,1,1]),1),
    ]
    x,y = collation(batch)
    assert x.shape == (2, 694)
    assert y.shape == (2,)
    assert x[0].tolist() == [1,2,3,4] + [0]*690
    assert x[1].tolist() == [1,2,3,4,1,2,3,4,1,2,3,4,1,1,1] + [0]*679


def test_collate_fixed():
    collation = CollateFixedLength(length=20)

    batch = [
        (Tensor([1,2,3,4]),0),
        (Tensor([1,2,3,4,1,2,3,4,1,2,3,4,1,1,1]),1),
    ]

    x,y = collation(batch)

    assert x.shape == (2, 20)
    assert y.shape == (2,)
    assert x[0].tolist() == [1,2,3,4] + [0]*16
    assert x[1].tolist() == [1,2,3,4,1,2,3,4,1,2,3,4,1,1,1] + [0]*5


ENGINES = ["rocksdict", "lmdb", "speedict"]


@pytest.fixture(params=ENGINES)
def seqbank(request):
    """
    Opens the test SeqBank for each storage engine.

    Skips if the engine's Python package is not installed (e.g. speedict on Python >= 3.13).
    The SeqBank is closed afterwards because LMDB cannot open the same environment twice in one process.
    """
    engine = request.param
    pytest.importorskip(engine)
    seqbank = SeqBank(TEST_DATA/f"seqbank.{engine}.sb", engine=engine)
    yield seqbank
    seqbank.close()


def test_train_dataloader(seqbank):
    seqtree = SeqTree.load(TEST_DATA/"seqtree.pkl")
    data = CorgiDataModule(
        seqtree=seqtree, 
        seqbank=seqbank, 
        validation_partition=1, 
        batch_size=4, 
        maximum_length=667,
        num_workers=0,
    )
    data.setup()
    dataloader = data.train_dataloader()
    x,y = next(iter(dataloader))
    assert x.shape == (2, 663)
    assert y.shape == (2,)


def test_val_dataloader(seqbank):
    seqtree = SeqTree.load(TEST_DATA/"seqtree.pkl")
    data = CorgiDataModule(
        seqtree=seqtree, 
        seqbank=seqbank, 
        validation_partition=1, 
        batch_size=4, 
        maximum_length=667,
        validation_length=999,
        num_workers=1,
    )
    data.setup()
    dataloader = data.val_dataloader()
    x,y = next(iter(dataloader))
    assert x.shape == (2, 999)
    assert y.shape == (2,)


def test_generate_overlapping_intervals():
    assert generate_overlapping_intervals(20, 5, 2) == [(0, 5), (3, 8), (6, 11), (9, 14), (12, 17), (15, 20)]
    assert generate_overlapping_intervals(3, 5, 2) == [(0, 3)]
    assert generate_overlapping_intervals(5, 5, 2) == [(0, 5)]


def test_generate_overlapping_intervals_variable_size():
    intervals = generate_overlapping_intervals(23, 10, 2, variable_size=True)
    assert intervals[0][0] == 0
    assert intervals[-1][1] == 23
    sizes = {end - start for start, end in intervals}
    assert len(sizes) == 1
    assert sizes.pop() <= 10


@pytest.mark.parametrize("total", [11, 50, 101, 1000])
def test_generate_overlapping_intervals_cover_range(total):
    intervals = generate_overlapping_intervals(total, 10, 3)
    assert intervals[0][0] == 0
    assert intervals[-1][1] == total
    for previous, current in zip(intervals, intervals[1:]):
        assert previous[1] - current[0] >= 3


def test_generate_overlapping_intervals_bad_overlap():
    with pytest.raises(AssertionError):
        generate_overlapping_intervals(100, 5, 5)


def test_slice_tensor():
    tensor = torch.arange(10)
    assert slice_tensor(tensor, 4, start_index=2).tolist() == [2, 3, 4, 5]
    assert slice_tensor(tensor, 4, start_index=8).tolist() == [8, 9, 0, 0]
    assert slice_tensor(tensor, 4, start_index=8, pad=False).tolist() == [8, 9]
    assert slice_tensor(tensor, 20).tolist() == list(range(10)) + [0] * 10
    assert slice_tensor(tensor, 4).shape == (4,)


def test_collate_base_not_implemented():
    with pytest.raises(NotImplementedError):
        Collate().get_length()


def make_dataset(seqbank, deterministic=False, maximum=0):
    seqtree = SeqTree.load(TEST_DATA / "seqtree.pkl")
    return CorgiTrainingDataset(
        accessions=sorted(seqtree.keys()),
        seqbank=seqbank,
        seqtree=seqtree,
        deterministic=deterministic,
        maximum=maximum,
    )


def test_training_dataset_getitem(seqbank):
    dataset = make_dataset(seqbank)
    assert len(dataset) == 5
    x, y = dataset[0]
    assert x.shape[0] == len(seqbank[dataset.accessions[0]])
    assert y == dataset.seqtree[dataset.accessions[0]].node_id


def test_training_dataset_truncates(seqbank):
    dataset = make_dataset(seqbank, deterministic=True, maximum=50)
    first, _ = dataset[0]
    again, _ = dataset[0]
    assert first.shape == (50,)
    assert torch.equal(first, again)

    random_dataset = make_dataset(seqbank, deterministic=False, maximum=50)
    assert random_dataset[0][0].shape == (50,)


def test_training_dataset_pickle_reopens_seqbank(seqbank):
    dataset = make_dataset(seqbank, maximum=50)
    clone = pickle.loads(pickle.dumps(dataset))
    assert clone.accessions == dataset.accessions
    assert clone.seqbank is not dataset.seqbank
    assert clone.seqbank.path == dataset.seqbank.path
    assert clone[0][0].shape == (50,)
    clone.seqbank.close()  # LMDB cannot open the same environment twice in one process


def test_training_dataset_reopen_seqbank(seqbank):
    dataset = make_dataset(seqbank)
    original = dataset.seqbank
    dataset[0]
    dataset.reopen_seqbank()
    assert dataset.seqbank is not original
    assert dataset[0][0].shape[0] > 0


def datamodule(seqbank, **kwargs):
    kwargs.setdefault("validation_partition", 1)
    kwargs.setdefault("batch_size", 4)
    kwargs.setdefault("num_workers", 0)
    return CorgiDataModule(seqtree=SeqTree.load(TEST_DATA / "seqtree.pkl"), seqbank=seqbank, **kwargs)


def test_datamodule_default_num_workers(seqbank):
    data = datamodule(seqbank, num_workers=None)
    data.setup()
    assert 1 <= data.num_workers <= 8


def test_datamodule_train_all(seqbank):
    data = datamodule(seqbank)
    data.setup()
    train_count = len(data.train_dataset)

    data = datamodule(seqbank, train_all=True)
    data.setup()
    assert len(data.train_dataset) > train_count
    assert len(data.train_dataset) == len(data.train_dataset.seqtree) - len(
        list(data.train_dataset.seqtree.accessions_in_partition(0))
    )


def test_datamodule_max_items(seqbank):
    data = datamodule(seqbank, max_items=1)
    data.setup()
    assert len(data.train_dataset) == 1
    assert len(data.val_dataset) == 1


def test_datamodule_no_validation_accessions_falls_back_to_training(seqbank):
    data = datamodule(seqbank, validation_partition=99)
    data.setup()
    assert data.val_dataset.accessions == data.train_dataset.accessions


def test_datamodule_same_validation_and_test_partition(seqbank):
    data = datamodule(seqbank, validation_partition=1, test_partition=1)
    with pytest.raises(AssertionError):
        data.setup()


def test_seqbank_worker_init(seqbank):
    from unittest.mock import patch

    dataset = make_dataset(seqbank)
    with patch("torch.utils.data.get_worker_info") as get_worker_info:
        get_worker_info.return_value.dataset = dataset
        original = dataset.seqbank
        seqbank_worker_init(0)
    assert dataset.seqbank is not original


FASTA = ">seq1 first\n" + "ACGT" * 100 + "\n>seq2 second\n" + "TTGCA" * 60 + "\n>short\nACGT\n"


@pytest.fixture
def fasta_dir(tmp_path):
    (tmp_path / "a.fasta").write_text(FASTA)
    with gzip.open(tmp_path / "b.fna.gz", "wt") as f:
        f.write(FASTA)
    (tmp_path / "ignored.txt").write_text("not a sequence file")
    return tmp_path


def test_seqio_dataloader_single_file(fasta_dir):
    loader = SeqIODataloader(fasta_dir / "a.fasta", batch_size=2, min_length=50, max_length=200, overlap=20)
    assert loader.files == [fasta_dir / "a.fasta"]
    assert len(loader) == 5  # seq1 (400) -> 3 chunks, seq2 (300) -> 2 chunks; 'short' is skipped

    batches = list(loader)
    assert sum(batch[0].shape[0] for batch in batches) == len(loader)
    assert all(batch[0].shape[1] <= 200 for batch in batches)
    assert len(loader.chunk_details) == len(loader)
    assert loader.chunk_details[0][1] == "seq1"
    assert loader.chunk_details[-1][1] == "seq2"


def test_seqio_dataloader_directory_and_gzip(fasta_dir):
    loader = SeqIODataloader(fasta_dir, batch_size=4, min_length=50, max_length=1000, overlap=20)
    assert {f.name for f in loader.files} == {"a.fasta", "b.fna.gz"}
    assert len(loader) == 4
    assert [(f.name, r.id) for f, r in loader.iter_records()].count(("b.fna.gz", "seq1")) == 1


def test_seqio_dataloader_max_seqs(fasta_dir):
    loader = SeqIODataloader([fasta_dir / "a.fasta"], min_length=50, max_length=1000, overlap=20, max_seqs=1)
    assert len(loader) == 1
    assert sum(batch[0].shape[0] for batch in loader) == 1


def test_seqio_dataloader_ignores_other_files(fasta_dir):
    loader = SeqIODataloader([fasta_dir / "ignored.txt"])
    assert loader.files == []
    assert len(loader) == 0
    assert list(loader) == []


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.fa", "fasta"),
        ("a.fna.gz", "fasta"),
        ("a.gb", "genbank"),
        ("a.gbk.gz", "genbank"),
        ("a.tsv", "tsv"),
        ("a.fq", "fastq"),
        ("A.FASTA", "fasta"),
    ],
)
def test_seqio_dataloader_file_format(fasta_dir, name, expected):
    loader = SeqIODataloader([])
    assert loader.get_file_format(Path(name)) == expected


def test_seqio_dataloader_file_format_explicit_and_unknown():
    assert SeqIODataloader([], format="genbank").get_file_format("whatever.xyz") == "genbank"
    with pytest.raises(ValueError):
        SeqIODataloader([]).get_file_format("file.unknown")


def test_seqio_dataloader_genbank(tmp_path):
    from shutil import copy

    copy(TEST_DATA / "NC_010663.1.gb", tmp_path / "x.gb")
    loader = SeqIODataloader(tmp_path / "x.gb", min_length=10)
    assert loader.files == []  # .gb is not one of the scanned extensions
    loader.files = [tmp_path / "x.gb"]
    assert len(list(loader.iter_records())) == 1


@pytest.mark.parametrize("variable_size", [False, True])
def test_generate_overlapping_intervals_many_sizes(variable_size):
    for total in range(1, 150):
        for interval_size in [5, 10, 17]:
            for min_overlap in [0, 1, 3]:
                if min_overlap >= interval_size:
                    continue
                intervals = generate_overlapping_intervals(
                    total, interval_size, min_overlap, check=True, variable_size=variable_size
                )
                assert intervals[0][0] == 0
                assert intervals[-1][1] == total
