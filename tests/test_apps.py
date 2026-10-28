from torchapp.testing import TorchAppTestCase, CliRunner
import re
import pytest
import torch
import typer
from corgi.apps import Corgi, limit_torch_threads
from pathlib import Path
from corgi.seqtree import SeqTree
from polytorch import HierarchicalData
from unittest.mock import patch


def test_limit_torch_threads():
    original_threads = torch.get_num_threads()
    try:
        limit_torch_threads(1)
        assert torch.get_num_threads() == 1

        with pytest.raises(typer.BadParameter):
            limit_torch_threads(0)
    finally:
        torch.set_num_threads(original_threads)


def mock_init(self):
    self.seqtree = SeqTree.load(Path(__file__).parent/"testdata/seqtree.pkl")
    self.classification_tree = self.seqtree.classification_tree
    self.output_types = [
        HierarchicalData(root=self.classification_tree),
    ]
    super(Corgi, self).__init__()


@patch("corgi.apps.Corgi.__init__", new=mock_init)
class TestCorgi(TorchAppTestCase):
    app_class = Corgi

    def test_version_main(self):
        app = self.get_app()
        runner = CliRunner()
        result = runner.invoke(app.main_app, ["--version"])
        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}: {result.stdout}"
        pep440_regex = r"^\d+(\.\d+)*([a-zA-Z]+\d+)?([+-][\w\.]+)?$"
        assert re.match(pep440_regex, result.stdout)


import pandas as pd
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock
from corgi.apps import output_results_bar_chart, set_alphas_with_phi

TEST_DATA = Path(__file__).parent / "testdata"


@pytest.fixture
def app():
    with patch("corgi.apps.Corgi.__init__", new=mock_init):
        yield Corgi()


@pytest.fixture
def fasta_file(tmp_path):
    path = tmp_path / "input" / "example.fasta"
    path.parent.mkdir()
    path.write_text(">NC_1#Bacteria first\n" + "ACGT" * 100 + "\n>seq2 second\n" + "TTGCA" * 60 + "\n")
    return path


def test_output_results_bar_chart(capsys):
    predictions = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d"])
    output_results_bar_chart(predictions, top_k=2, bar_size=10)
    output = capsys.readouterr().out
    assert "Prediction" in output
    assert "Other Categories" in output
    assert "Total" in output
    assert "11" in output


def test_output_results_bar_chart_no_other(capsys):
    output_results_bar_chart(pd.Series(["a", "a", "b"]), top_k=10)
    output = capsys.readouterr().out
    assert "Other Categories" not in output


def test_set_alphas_with_phi():
    seqtree = SeqTree.load(TEST_DATA / "seqtree.pkl")
    tree = seqtree.classification_tree
    set_alphas_with_phi(tree, phi=0.5)
    assert tree.alpha == 1.0
    for child in tree.children:
        assert child.alpha == 0.5
    set_alphas_with_phi(tree)
    assert all(node.alpha == 1.0 for node in tree.pre_order_iter())


def test_limit_torch_threads_none():
    original_threads = torch.get_num_threads()
    limit_torch_threads(None)
    assert torch.get_num_threads() == original_threads


def test_app_simple_methods(app):
    assert app.package_name() == "bio-corgi"
    assert app.monitor() == "superkingdom_accuracy"
    assert app.checkpoint().startswith("https://")
    assert app.checkpoint(large=True) != app.checkpoint()
    assert app.checkpoint(checkpoint=Path("local.ckpt")) == Path("local.ckpt")
    assert app.module_class().__name__ == "CorgiLightningModule"
    assert app.loss_function() is not None
    assert [name for name, _ in app.metrics()] == ["rank_accuracy"]


def test_app_node_to_str(app):
    node = app.classification_tree.children[0]
    assert app.node_to_str(node) == str(node)


def test_app_model(app):
    model = app.model(cnn_layers=2, cnn_dims_start=8, penultimate_dims=16)
    x = torch.randint(0, 5, (2, 64), dtype=torch.uint8)
    assert len(model(x)) > 0


def test_app_model_from_macc(app):
    model = app.model(cnn_layers=2, penultimate_dims=16, macc=1_000_000)
    assert model.cnn_layers == 2


def test_app_model_pretrained_missing(app, tmp_path):
    with pytest.raises(FileNotFoundError):
        app.model(pretrained=tmp_path / "missing.ckpt")


def test_app_model_pretrained(app, tmp_path):
    original = app.model(cnn_layers=1, cnn_dims_start=8, penultimate_dims=16)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("placeholder")
    module = SimpleNamespace(model=original)
    module_class = MagicMock()
    module_class.load_from_checkpoint.return_value = module
    with patch.object(app, "module_class", return_value=module_class):
        model = app.model(pretrained=checkpoint, final_bias=False)
    assert model is original
    module_class.load_from_checkpoint.assert_called_once_with(str(checkpoint))
    assert model.final.bias is None


def test_app_model_pretrained_default(app, tmp_path):
    checkpoint = tmp_path / "default.ckpt"
    checkpoint.write_text("placeholder")
    original = app.model(cnn_layers=1, cnn_dims_start=8, penultimate_dims=16)
    module_class = MagicMock()
    module_class.load_from_checkpoint.return_value = SimpleNamespace(model=original)
    with patch.object(app, "module_class", return_value=module_class), patch.object(
        app, "process_location", return_value=checkpoint
    ) as process_location:
        app.model(pretrained=Path("default"))
    process_location.assert_called_once()
    module_class.load_from_checkpoint.assert_called_once_with(str(checkpoint))


def test_app_data(app):
    data = app.data(
        seqtree=TEST_DATA / "seqtree.pkl",
        seqbank=TEST_DATA / "seqbank.rocksdict.sb",
        batch_size=2,
        validation_partition=1,
    )
    assert data.batch_size == 2
    assert app.classification_tree is not None
    data.seqbank.close()


def test_app_data_requires_inputs(app):
    with pytest.raises(AssertionError):
        app.data(seqtree=None, seqbank=TEST_DATA / "seqbank.rocksdict.sb")
    with pytest.raises(AssertionError):
        app.data(seqtree=TEST_DATA / "seqtree.pkl", seqbank=None)


def test_find_input_paths(app, tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ["a.fasta", "b.fq", "c.gb", "d.txt", "sub/e.fna.gz", "sub/f.tsv.gz"]:
        (tmp_path / name).write_text("")
    found = {p.name for p in app.find_input_paths(tmp_path)}
    assert found == {"a.fasta", "b.fq", "c.gb", "e.fna.gz", "f.tsv.gz"}

    assert [p.name for p in app.find_input_paths(str(tmp_path / "a.fasta"))] == ["a.fasta"]
    assert app.find_input_paths([tmp_path / "d.txt"]) == []
    assert app.find_input_paths([tmp_path / "d.txt"], base_extensions={".txt"}) == [tmp_path / "d.txt"]

    fasta = {p.name for p in app.find_fasta_paths([tmp_path])}
    assert fasta == {"a.fasta", "e.fna.gz"}


def test_prediction_dataloader(app, fasta_file, tmp_path):
    module = MagicMock()
    dataloader = app.prediction_dataloader(
        module,
        input=[fasta_file.parent],
        seqtree=TEST_DATA / "seqtree.pkl",
        min_length=10,
        max_length=300,
        batch_size=2,
        num_threads=1,
    )
    assert len(dataloader) == 5  # The chunks overlap by at least 256 bases
    assert app.classification_tree is not None
    module.set_embedding_path.assert_not_called()

    embeddings = tmp_path / "embeddings.npy"
    app.prediction_dataloader(module, file=[fasta_file], seqtree=TEST_DATA / "seqtree.pkl", embeddings=embeddings)
    module.set_embedding_path.assert_called_once()
    assert module.set_embedding_path.call_args.kwargs["embeddings_path"] == embeddings


def test_prediction_dataloader_no_files(app):
    with pytest.raises(typer.BadParameter):
        app.prediction_dataloader(MagicMock(), input=[])


def test_prediction_dataloader_classification_tree_from_module(app, fasta_file):
    seqtree = SeqTree.load(TEST_DATA / "seqtree.pkl")
    module = MagicMock()
    module.hparams = {"classification_tree": seqtree.classification_tree}
    app.classification_tree = None
    app.prediction_dataloader(module, input=[fasta_file], seqtree=TEST_DATA / "does-not-exist.st")
    assert app.classification_tree is seqtree.classification_tree


def run_predictions(app, fasta_file):
    """ Runs the dataloader through an untrained model to give the inputs for output_results. """
    model = app.model(cnn_layers=1, cnn_dims_start=8, penultimate_dims=16, include_length=True)
    model.eval()
    app.prediction_dataloader(
        MagicMock(), input=[fasta_file], seqtree=TEST_DATA / "seqtree.pkl", min_length=10, max_length=300, batch_size=2
    )
    outputs = []
    with torch.no_grad():
        for (batch,) in app.dataloader:
            outputs.append(torch.cat(model(batch), dim=1))
    return [torch.cat(outputs)]


def test_output_results(app, fasta_file, tmp_path, capsys):
    results = run_predictions(app, fasta_file)
    output_csv = tmp_path / "results.csv"
    output_tips_csv = tmp_path / "tips.csv"
    app.output_results(results, output_csv=output_csv, output_tips_csv=output_tips_csv)

    assert "Total" in capsys.readouterr().out
    df = pd.read_csv(output_csv)
    assert list(df["original_id"]) == ["NC_1#Bacteria", "seq2"]
    assert list(df["accession"]) == ["NC_1", "seq2"]
    assert {"file", "prediction", "probability", "description"} <= set(df.columns)
    assert df["probability"].between(0, 1).all()


def test_output_results_images(app, fasta_file, tmp_path):
    results = run_predictions(app, fasta_file)
    image_dir = tmp_path / "images"
    with patch("hierarchicalsoftmax.inference.render_probabilities") as render:
        app.output_results(results, image_dir=image_dir, image_format="svg")
    render.assert_called_once()
    paths = render.call_args.kwargs["filepaths"]
    assert [p.name for p in paths] == ["NC_1.svg", "seq2.svg"]
    assert all(p.parent == image_dir / "example.fasta" for p in paths)


def test_output_results_no_outputs(app, fasta_file, capsys):
    app.output_results(run_predictions(app, fasta_file))
    assert "No output files requested." in capsys.readouterr().out


def test_extra_hyperparameters(app):
    assert app.extra_hyperparameters()["classification_tree"] is app.classification_tree


def test_refseq_to_seqtree_tool(app, tmp_path):
    with patch("corgi.refseq.refseq_to_seqtree") as refseq_to_seqtree:
        result = CliRunner().invoke(
            app.tools_app,
            [
                "refseq-to-seqtree",
                "--output", str(tmp_path / "out.st"),
                "--seqbank", str(tmp_path / "in.sb"),
                "--seqbank-output", str(tmp_path / "out.sb"),
                "--engine", "lmdb",
                "--max-seqs", "7",
                "--ranks", "phylum",
            ],
        )
    assert result.exit_code == 0, result.output
    kwargs = refseq_to_seqtree.call_args.kwargs
    assert kwargs["engine"] == "lmdb"
    assert kwargs["seqbank_output"] == tmp_path / "out.sb"
    assert kwargs["max_seqs"] == 7
    assert kwargs["ranks"] == "phylum"
    assert kwargs["output"] == tmp_path / "out.st"
