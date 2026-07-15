import gzip
from pathlib import Path

from experiments.gcp_mdlm.corpus.cif_source import accession_from_path, decompressed_cif

FIXTURES = Path(__file__).parent / "fixtures"


def test_accession_from_path():
    assert accession_from_path("AF-P14921-F1-model_v4.cif.gz") == "P14921"
    assert accession_from_path(Path("/x/AF-Q9K3Z0-F1-model_v4.cif.gz")) == "Q9K3Z0"


def test_decompressed_cif_roundtrip():
    gz = next(FIXTURES.glob("*.cif.gz"))
    with decompressed_cif(gz) as cif_path:
        assert cif_path.exists() and cif_path.suffix == ".cif"
        head = cif_path.read_text()[:200]
        assert head.startswith("data_")  # valid mmCIF
        raw = gzip.open(gz, "rt").read()
        assert cif_path.read_text() == raw
    assert not cif_path.exists()  # cleaned up
