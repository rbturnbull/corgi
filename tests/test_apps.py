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
