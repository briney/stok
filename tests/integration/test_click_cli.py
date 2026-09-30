from click.testing import CliRunner

from stok.cli.cli import cli


def test_click_cli_smoke_test_runs_with_override():
    runner = CliRunner()
    result = runner.invoke(cli, ["smoke-test", "model.encoder.n_layers=6"])  # type: ignore[arg-type]
    assert result.exit_code == 0
    assert "OK" in result.output
