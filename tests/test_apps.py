from torchapp.testing import TorchAppTestCase, CliRunner
import re
from corgi.apps import Corgi
from pathlib import Path
from corgi.seqtree import SeqTree
from polytorch import HierarchicalData
from unittest.mock import patch

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
