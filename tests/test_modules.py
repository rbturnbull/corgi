from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from polytorch import HierarchicalData

from corgi.models import ConvClassifier
from corgi.modules import CorgiLightningModule, read_embeddings, read_memmap
from corgi.seqtree import SeqTree

TEST_DATA = Path(__file__).parent / "testdata"
PENULTIMATE_DIMS = 16


@pytest.fixture
def module():
    seqtree = SeqTree.load(TEST_DATA / "seqtree.pkl")
    model = ConvClassifier(
        output_types=[HierarchicalData(root=seqtree.classification_tree)],
        cnn_layers=1,
        cnn_dims_start=8,
        penultimate_dims=PENULTIMATE_DIMS,
        transformer_layers=0,
    )
    module = CorgiLightningModule(
        model=model, loss_function=lambda *args: torch.tensor(0.0), max_learning_rate=1e-3, metrics=[]
    )
    module.eval()
    return module


def batch(size):
    return (torch.randint(0, 5, (size, 32), dtype=torch.uint8),)


def test_initial_state(module):
    assert module.embeddings_path is None
    assert module.embeddings is None
    assert module.dataloader is None
    assert module.current_position == 0


def test_predict_step_without_embeddings(module):
    result = module.predict_step(batch(3))
    assert module.model.return_embeddings is False
    assert result[0].shape[0] == 3
    assert module.embeddings is None


def test_predict_step_requires_tuple(module):
    with pytest.raises(AssertionError):
        module.predict_step(batch(1)[0])


def test_predict_step_with_embeddings(module, tmp_path):
    path = tmp_path / "out" / "embeddings.npy"
    dataloader = SimpleNamespace(
        count=5,
        chunk_details=[
            ("a\tfile.fasta", 1, "", 0),
            ("a\tfile.fasta", 1, "", 1),
            ("b.fasta", "id\t2", "", 0),
            ("b.fasta", "id\t3", "", 0),
            ("b.fasta", "id\t3", "", 1),
        ],
    )
    module.set_embedding_path(path, dataloader)
    assert module.embeddings_path == path

    with torch.no_grad():
        first = module.predict_step(batch(3))
        second = module.predict_step(batch(2))
    assert module.model.return_embeddings is True
    assert first[0].shape[0] == 3 and second[0].shape[0] == 2
    assert module.current_position == 5

    module.on_predict_epoch_end()
    assert module.embeddings is None
    assert module.current_position == 0

    # The memmap is half precision with one row per chunk
    embeddings = read_memmap(path, count=5)
    assert embeddings.shape == (5, PENULTIMATE_DIMS)
    assert embeddings.dtype == np.float16

    index_lines = path.with_suffix(".index").read_text().splitlines()
    assert index_lines == ["a_file.fasta\t1", "a_file.fasta\t1", "b.fasta\tid_2", "b.fasta\tid_3", "b.fasta\tid_3"]

    # Read back with chunks from the same record averaged
    merged, index = read_embeddings(path)
    assert merged.shape == (3, PENULTIMATE_DIMS)
    assert index == [["a_file.fasta", "1"], ["b.fasta", "id_2"], ["b.fasta", "id_3"]]
    np.testing.assert_allclose(merged[0], embeddings[:2].astype(np.float64).mean(axis=0), rtol=1e-3, atol=1e-3)

    unmerged, index = read_embeddings(path, merge=False)
    assert unmerged.shape == (5, PENULTIMATE_DIMS)
    assert len(index) == 5


def test_on_predict_epoch_end_without_embeddings(module):
    module.on_predict_epoch_end()
    assert module.embeddings is None


def test_read_embeddings_missing_index(tmp_path):
    with pytest.raises(AssertionError):
        read_embeddings(tmp_path / "missing.npy")


def test_read_embeddings_explicit_index(tmp_path):
    path = tmp_path / "embeddings.npy"
    np.arange(6, dtype=np.float16).reshape(2, 3).tofile(path)
    index = tmp_path / "custom.index"
    index.write_text("x\ty\nx\tz\n")
    embeddings, names = read_embeddings(path, embeddings_index=index, merge=False)
    assert embeddings.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert names == [["x", "y"], ["x", "z"]]
