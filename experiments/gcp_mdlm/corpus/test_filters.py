from pathlib import Path

from experiments.gcp_mdlm.corpus.cif_source import decompressed_cif
from experiments.gcp_mdlm.corpus.filters import CorpusFilters, classify, mean_plddt_from_cif

FIXTURES = Path(__file__).parent / "fixtures"


def test_mean_plddt_is_plausible():
    gz = next(FIXTURES.glob("*.cif.gz"))
    with decompressed_cif(gz) as cif:
        plddt = mean_plddt_from_cif(cif)
    assert 0.0 <= plddt <= 100.0  # pLDDT scale


def test_classify_plddt_threshold():
    f = CorpusFilters(min_mean_plddt=70.0)
    assert classify(mean_plddt=85.0, filters=f) == "accepted"
    assert classify(mean_plddt=42.0, filters=f) == "rejected_plddt"
