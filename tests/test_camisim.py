import gzip
from unittest.mock import patch

import pandas as pd
from typer.testing import CliRunner

from corgi.camisim import app


class FakeTaxonomy:
    superkingdoms = {"1": "Bacteria", "2": "Archaea", "3": "Eukaryota", "4": "Viruses"}

    def parent_rank(self, taxon_id, rank):
        assert rank == "superkingdom"
        return self.superkingdoms[str(taxon_id)]


def test_camisim(tmp_path):
    mapping = tmp_path / "gsa_mapping.tsv.gz"
    rows = [
        ("NC_1.1", 1, "g1"),
        ("NC_2.1", 2, "g2"),
        ("NC_3.1", 3, "g3"),
        ("NC_4.1", 4, "g4"),
        ("NC_5.1", 1, "g5"),
        ("NC_6.1", 1, "g6"),
        ("NC_7.1", 1, "g7"),
    ]
    with gzip.open(mapping, "wt") as f:
        f.write("contig_id\ttaxon_id\tgenome\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")

    genomes_dir = tmp_path / "genomes"
    genomes_dir.mkdir()
    descriptions = {
        "NC_1.1": "Escherichia coli chromosome",
        "NC_2.1": "Methanococcus chromosome",
        "NC_3.1": "Zea mays chromosome",
        "NC_4.1": "Virus chromosome",
        "NC_5.1": "Homo sapiens mitochondrion",
        "NC_6.1": "Zea mays plastid",
        "NC_7.1": "Escherichia coli plasmid pX",
        "NC_unmapped.1": "Not in the mapping file",
    }
    (genomes_dir / "all.fasta").write_text(
        "".join(f">{accession} {description}\nACGTACGTAC\n" for accession, description in descriptions.items())
    )
    output = tmp_path / "labels.csv"

    with patch("corgi.camisim.NcbiTx", return_value=FakeTaxonomy()):
        result = CliRunner().invoke(app, [str(mapping), str(genomes_dir), str(output)])

    assert result.exit_code == 0, result.output
    df = pd.read_csv(output).set_index("contig_id")
    assert df["corgi_category"].to_dict() == {
        "NC_1.1": "Nuclear/Bacteria",
        "NC_2.1": "Nuclear/Archaea",
        "NC_3.1": "Nuclear/Eukaryota",
        "NC_4.1": "Nuclear/Viruses",
        "NC_5.1": "Mitochondrion",
        "NC_6.1": "Plastid",
        "NC_7.1": "Plasmid",
    }
    assert df["tiara_category"].to_dict() == {
        "NC_1.1": "bac",
        "NC_2.1": "arc",
        "NC_3.1": "euk",
        "NC_4.1": "unk1",
        "NC_5.1": "mit",
        "NC_6.1": "pla",
        "NC_7.1": "unk2",
    }
