from torchapp.testing import TorchAppTestCase
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

