"""Integration tests for `stok unfold` (stubbed in Phase 2)."""

from __future__ import annotations

from click.testing import CliRunner

from stok.cli.cli import cli


class TestUnfoldStub:
    def test_exits_nonzero_with_helpful_message(self, tmp_path):
        """unfold is not implemented; CLI must exit non-zero with a clear
        error mentioning ``untokenize`` as the workaround.
        """
        struct_file = tmp_path / "input.pdb"
        struct_file.write_text("HEADER    placeholder\n")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "unfold",
                "--input-structure-file", str(struct_file),
                "--output", str(tmp_path / "out.fasta"),
            ],
        )
        assert result.exit_code != 0
        assert "untokenize" in result.output.lower()
