"""mmCIF writer for protein backbone structures.

Mirrors :func:`stok.utils.pdb.build_pdb` but writes the mmCIF format via
:class:`Bio.PDB.MMCIFIO`. This exists so ``stok design`` / ``fold`` /
``untokenize`` can emit modern mmCIF files when callers prefer them to
legacy PDB.
"""

from __future__ import annotations

import numpy as np
import torch
from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure
from Bio.SeqUtils import seq3


def build_mmcif(
    coordinates: np.ndarray | torch.Tensor,
    output_file: str,
    sequence: str | None = None,
) -> None:
    """Write backbone coordinates to an mmCIF file.

    Args:
        coordinates: Shape ``(L, 3, 3)`` for N, CA, C atoms per residue.
        output_file: Path to save the mmCIF file.
        sequence: Optional one-letter amino acid sequence (e.g., ``"ACDE"``).
            When omitted every residue is labelled ``UNK``.
    """
    if sequence is None:
        sequence = "X" * coordinates.shape[0]
    if len(sequence) != coordinates.shape[0]:
        raise ValueError(
            "Sequence length must match number of residues in coordinates."
        )
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.numpy()

    structure = Structure.Structure("protein")
    model = Model.Model(0)
    chain = Chain.Chain("A")

    atom_serial = 1
    for i, aa in enumerate(sequence):
        res_name = seq3(aa).upper()
        res = Residue.Residue(id=(" ", i + 1, " "), resname=res_name, segid=" ")
        for j, atom_name in enumerate(["N", "CA", "C"]):
            coord = coordinates[i, j]
            if not isinstance(coord, np.ndarray):
                coord = np.array(coord, dtype=float)
            atom = Atom.Atom(
                name=atom_name,
                coord=coord,
                bfactor=0.0,
                occupancy=1.0,
                altloc=" ",
                fullname=f" {atom_name.ljust(3)}",
                serial_number=atom_serial,
                element=atom_name[0],
            )
            res.add(atom)
            atom_serial += 1
        chain.add(res)

    model.add(chain)
    structure.add(model)

    io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_file)
