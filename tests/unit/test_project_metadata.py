from pathlib import Path


def test_exact_structure_graph_dependency_is_declared():
    pyproject = Path(__file__).parents[2] / "pyproject.toml"

    assert '"torch-cluster>=1.6.3",' in pyproject.read_text()
