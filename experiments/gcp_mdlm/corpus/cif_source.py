"""Read gzipped AlphaFold CIFs: decompress to a temp path, parse the accession."""

from __future__ import annotations

import gzip
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_ACCESSION = re.compile(r"AF-([0-9A-Za-z]+)-F\d+-model")


def accession_from_path(path: str | Path) -> str:
    """Extract the UniProt accession from an AlphaFold filename."""
    name = Path(path).name
    match = _ACCESSION.search(name)
    if match is None:
        raise ValueError(f"cannot parse accession from {name!r}")
    return match.group(1)


@contextmanager
def decompressed_cif(path: str | Path) -> Iterator[Path]:
    """Decompress a ``.cif.gz`` to a temp ``.cif`` file; yield its path, then delete it."""
    path = Path(path)
    tmp = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
    try:
        with gzip.open(path, "rb") as src:
            shutil.copyfileobj(src, tmp)
        tmp.close()
        yield Path(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
