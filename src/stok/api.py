"""Public Python API for stok generation.

All business logic (model loading, sampling, structure decoding, file I/O)
lives here. The CLI subcommands are thin wrappers that parse arguments, build
a typed :class:`MDLMModelConfig`, and call into this module.

Users can also import and call these functions directly from Python::

    from stok.api import load_model, load_decoder, fold, MDLMModelConfig

    cfg = MDLMModelConfig(tracks="joint", d_model=64, n_heads=4, n_layers=2)
    loaded = load_model("model.pt", config=cfg)
    decoder = load_decoder("base")
    result = fold(
        loaded.model,
        sequences=["ACDEFGHIK"],
        decoder=decoder,
        codebook=loaded.codebook,
        tokenizer=loaded.tokenizer,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import torch

from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule
from stok.utils.codebook import load_codebook
from stok.utils.decoding import decode_coords, indices_to_codes
from stok.utils.sampling import sample
from stok.utils.tokenizer import Tokenizer

if TYPE_CHECKING:
    from stok.models.decoder import GeometricDecoder
    from stok.models.structure_encoder import StructureEncoder

logger = logging.getLogger(__name__)

__all__ = [
    "MDLMModelConfig",
    "NoiseScheduleConfig",
    "LoadedModel",
    "GenerationResult",
    "EncodeResult",
    "load_model",
    "load_decoder",
    "load_encoder",
    "design",
    "fold",
    "unfold",
    "tokenize",
    "untokenize",
    "encode",
]


# ---------------------------------------------------------------------------
# Configs and result types
# ---------------------------------------------------------------------------


@dataclass
class NoiseScheduleConfig:
    """Typed config for a single-track diffusion noise schedule."""

    type: str = "cosine"
    sigmoid_k: float = 6.0
    log_linear_k: float = 3.0
    eps: float = 1e-5

    def build(self) -> NoiseSchedule:
        """Instantiate the corresponding :class:`NoiseSchedule`."""
        return NoiseSchedule(
            schedule_type=self.type,
            sigmoid_k=self.sigmoid_k,
            log_linear_k=self.log_linear_k,
            eps=self.eps,
        )


@dataclass
class MDLMModelConfig:
    """Typed config for reconstructing an MDLM model from a checkpoint.

    Keeps Hydra/OmegaConf out of the public API. The CLI extracts these
    fields from the composed Hydra config via :meth:`from_omegaconf`; library
    users build this directly.
    """

    tracks: Literal["seq_only", "joint"] = "joint"
    seq_vocab_size: int = 32
    seq_pad_id: int = 1
    d_model: int = 1536
    n_heads: int = 24
    n_layers: int = 36
    ffn_mult: float = 2.667
    dropout: float = 0.0
    attn_dropout: float = 0.0
    norm_type: str = "layernorm"
    noise_schedule_seq: NoiseScheduleConfig = field(default_factory=NoiseScheduleConfig)
    noise_schedule_struct: NoiseScheduleConfig | None = None
    codebook_preset: Literal["base", "lite"] | None = "base"
    codebook_path: Path | None = None
    lambda_seq: float = 1.0
    lambda_struct: float = 1.0
    classifier_kwargs: dict | None = None
    tie_seq_embeddings: bool = True
    time_conditioning: str = "adaln"
    time_embed_dim: int | None = None
    time_combine: str = "sum"

    @classmethod
    def from_omegaconf(cls, cfg) -> MDLMModelConfig:
        """Convert a composed Hydra ``DictConfig`` into a typed config.

        Reads the same fields that the previous monolithic generation path
        pulled from the Hydra config tree. OmegaConf types are converted to
        plain Python primitives so the resulting dataclass is free of any
        Hydra coupling.
        """
        mdlm_cfg = cfg.train.mdlm
        tracks = str(mdlm_cfg.get("tracks", "joint"))

        ns_seq_cfg = mdlm_cfg.get("noise_schedule_seq", {})
        noise_schedule_seq = NoiseScheduleConfig(
            type=str(ns_seq_cfg.get("type", "cosine")),
            sigmoid_k=float(ns_seq_cfg.get("sigmoid_k", 6.0)),
            log_linear_k=float(ns_seq_cfg.get("log_linear_k", 3.0)),
            eps=float(ns_seq_cfg.get("eps", 1e-5)),
        )

        noise_schedule_struct: NoiseScheduleConfig | None = None
        classifier_kwargs: dict | None = None
        if tracks == "joint":
            ns_struct_cfg = mdlm_cfg.get("noise_schedule_struct", {})
            noise_schedule_struct = NoiseScheduleConfig(
                type=str(ns_struct_cfg.get("type", "cosine")),
                sigmoid_k=float(ns_struct_cfg.get("sigmoid_k", 6.0)),
                log_linear_k=float(ns_struct_cfg.get("log_linear_k", 3.0)),
                eps=float(ns_struct_cfg.get("eps", 1e-5)),
            )
            classifier_kwargs = dict(
                use_cosine=bool(cfg.model.classifier.use_cosine),
                learnable_temperature=bool(cfg.model.classifier.learnable_temperature),
                bias_from_code_norm=bool(cfg.model.classifier.bias_from_code_norm),
                projector_dim=cfg.model.classifier.projector_dim,
            )

        codebook_path_raw = cfg.model.codebook.get("path")
        codebook_path = Path(str(codebook_path_raw)) if codebook_path_raw else None

        time_embed_raw = mdlm_cfg.get("time_embed_dim")
        time_embed_dim = int(time_embed_raw) if time_embed_raw else None

        return cls(
            tracks=tracks,  # type: ignore[arg-type]
            seq_vocab_size=int(cfg.model.encoder.vocab_size),
            seq_pad_id=int(cfg.model.encoder.pad_id),
            d_model=int(cfg.model.encoder.d_model),
            n_heads=int(cfg.model.encoder.n_heads),
            n_layers=int(cfg.model.encoder.n_layers),
            ffn_mult=float(cfg.model.encoder.ffn_mult),
            dropout=float(cfg.model.encoder.dropout),
            attn_dropout=float(cfg.model.encoder.attn_dropout),
            norm_type=str(cfg.model.encoder.norm),
            noise_schedule_seq=noise_schedule_seq,
            noise_schedule_struct=noise_schedule_struct,
            codebook_preset=cfg.model.codebook.get("preset"),
            codebook_path=codebook_path,
            lambda_seq=float(mdlm_cfg.get("lambda_seq", 1.0)),
            lambda_struct=float(mdlm_cfg.get("lambda_struct", 1.0)),
            classifier_kwargs=classifier_kwargs,
            tie_seq_embeddings=bool(mdlm_cfg.get("tie_seq_embeddings", True)),
            time_conditioning=str(mdlm_cfg.get("time_conditioning", "adaln")),
            time_embed_dim=time_embed_dim,
            time_combine=str(mdlm_cfg.get("time_combine", "sum")),
        )


class LoadedModel(NamedTuple):
    """Output of :func:`load_model`: model + tokenizer + optional codebook."""

    model: MDLMModel
    tokenizer: Tokenizer
    codebook: torch.Tensor | None


class EncodeResult(NamedTuple):
    """Structured result of :func:`encode`.

    Attributes:
        sample_ids: One identifier per accepted structure sample. A file may
            yield multiple chain-qualified identifiers.
        sequences: One-letter amino-acid sequences as parsed from the
            structure, trimmed to the encoder's ``max_length``.
        struct_tokens: Per-sample lists of VQ code indices, already
            trimmed to each sample's true residue count.
    """

    sample_ids: list[str]
    sequences: list[str]
    struct_tokens: list[list[int]]


@dataclass
class GenerationResult:
    """Structured result of a generation run.

    Which fields are populated depends on the API function that produced it:

    - ``sequences``, ``seq_tokens``, ``sample_ids``, ``tracks``: always set.
    - ``struct_tokens``: set for joint-track runs.
    - ``coordinates``: set when a decoder was run and ``return_coordinates``
      is True.
    - ``structure_paths``: set when an ``output_dir`` was provided.
    """

    sequences: list[str]
    seq_tokens: torch.Tensor
    struct_tokens: torch.Tensor | None
    coordinates: torch.Tensor | None
    structure_paths: list[Path] | None
    sample_ids: list[str]
    tracks: Literal["seq_only", "joint"]


# ---------------------------------------------------------------------------
# Model + decoder loading
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: Path | str,
    *,
    config: MDLMModelConfig,
    device: torch.device | str = "cpu",
) -> LoadedModel:
    """Reconstruct an MDLM model from a checkpoint and typed config.

    Args:
        checkpoint_path: Path to a .pt checkpoint written via
            ``torch.save(model.state_dict(), ...)`` or a wrapped dict
            containing ``model_state_dict`` or ``state_dict`` keys.
        config: Typed model config describing the model shape.
        device: Device to place the model on.

    Returns:
        A :class:`LoadedModel` in ``eval()`` mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(device) if isinstance(device, str) else device
    tokenizer = Tokenizer()

    noise_schedule_seq = config.noise_schedule_seq.build()
    noise_schedule_struct = None
    codebook: torch.Tensor | None = None
    if config.tracks == "joint":
        if config.noise_schedule_struct is None:
            raise ValueError("noise_schedule_struct is required for joint tracks")
        noise_schedule_struct = config.noise_schedule_struct.build()
        codebook = load_codebook(
            preset=config.codebook_preset,
            path=str(config.codebook_path) if config.codebook_path else None,
        )

    model = MDLMModel(
        tracks=config.tracks,
        seq_vocab_size=config.seq_vocab_size,
        seq_pad_id=config.seq_pad_id,
        seq_mask_id=tokenizer.mask_token_id,
        codebook=codebook,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ffn_mult=config.ffn_mult,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        norm_type=config.norm_type,
        noise_schedule_seq=noise_schedule_seq,
        noise_schedule_struct=noise_schedule_struct,
        lambda_seq=config.lambda_seq,
        lambda_struct=config.lambda_struct,
        classifier_kwargs=config.classifier_kwargs,
        tie_seq_embeddings=config.tie_seq_embeddings,
        time_conditioning=config.time_conditioning,
        time_embed_dim=config.time_embed_dim,
        time_combine=config.time_combine,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return LoadedModel(model=model, tokenizer=tokenizer, codebook=codebook)


def load_decoder(
    preset: Literal["base", "lite"] = "base",
    *,
    path: Path | str | None = None,
    device: torch.device | str = "cpu",
) -> GeometricDecoder:
    """Load a pretrained :class:`GeometricDecoder`.

    Thin wrapper around :func:`stok.models.decoder.load_pretrained_decoder`
    with the frozen (eval + no-grad) defaults that make sense for inference.

    Args:
        preset: Architecture preset name (``"base"`` or ``"lite"``).
        path: Optional explicit weights path. Overrides download caching.
        device: Device to move the decoder onto.

    Returns:
        A frozen :class:`GeometricDecoder`.
    """
    from stok.models.decoder import load_pretrained_decoder

    return load_pretrained_decoder(
        preset=preset,
        path=str(path) if path is not None else None,
        device=device,
        freeze=True,
    )


def load_encoder(
    preset: Literal["base", "lite"] = "base",
    *,
    path: Path | str | None = None,
    device: torch.device | str = "cpu",
) -> StructureEncoder:
    """Load a pretrained :class:`StructureEncoder` for coordinates → tokens.

    Thin wrapper around
    :func:`stok.models.structure_encoder.load_pretrained_encoder` with the
    frozen (eval + no-grad) defaults that make sense for inference.

    Args:
        preset: Architecture preset name (``"base"`` or ``"lite"``).
        path: Optional explicit weights path. Overrides download caching.
        device: Device to move the encoder onto.

    Returns:
        A frozen :class:`StructureEncoder`.
    """
    from stok.models.structure_encoder import load_pretrained_encoder

    return load_pretrained_encoder(
        preset=preset,
        path=str(path) if path is not None else None,
        device=device,
        freeze=True,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_fasta_or_text(path: Path | str) -> list[str]:
    """Read sequences from a FASTA or plain-text file (one per line).

    Blank lines are ignored. FASTA header lines (``>``) end the current
    sequence.
    """
    sequences: list[str] = []
    current: list[str] = []
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def _tokens_to_sequences(tokens: torch.Tensor, tokenizer: Tokenizer) -> list[str]:
    """Convert a ``[B, L]`` token-id tensor to a list of amino acid strings."""
    sequences: list[str] = []
    for i in range(tokens.shape[0]):
        ids = tokens[i].tolist()
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        sequences.append(decoded.replace(" ", ""))
    return sequences


def _make_sample_ids(n: int) -> list[str]:
    """Generate ``sample_XXXX`` identifiers, widening past 4 digits as needed."""
    width = max(4, len(str(max(n, 1))))
    return [f"sample_{i:0{width}d}" for i in range(1, n + 1)]


def _tokenize_to_tensor(
    sequences: list[str],
    tokenizer: Tokenizer,
    length: int,
    pad_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode sequences to a right-padded ``[N, length]`` tensor of ids."""
    padded: list[list[int]] = []
    for seq in sequences:
        ids = tokenizer.encode(seq, add_special_tokens=False)
        ids = ids[:length]
        ids = ids + [pad_id] * (length - len(ids))
        padded.append(ids)
    return torch.tensor(padded, dtype=torch.long, device=device)


def _decode_struct_tokens(
    decoder: GeometricDecoder,
    codebook: torch.Tensor,
    struct_tokens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Decode structure tokens to ``[B, L, 3, 3]`` backbone coordinates.

    Looks up codes via :func:`indices_to_codes` against the codebook
    returned by :func:`load_model`, then calls :func:`decode_coords` so
    that the required ``mask`` argument is passed to
    :meth:`GeometricDecoder.forward`.
    """
    codebook = codebook.to(device)
    B, L = struct_tokens.shape
    C = codebook.shape[0]
    indices = struct_tokens.to(device).clamp(0, C - 1)
    codes = indices_to_codes(codebook, indices)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    return decode_coords(decoder, codes, mask)


def _write_structure(
    coords: torch.Tensor,
    path: Path,
    *,
    sequence: str | None = None,
) -> None:
    """Write a single structure file. Format is inferred from ``path.suffix``.

    Supported suffixes:

    - ``.pdb`` / ``.ent`` → :func:`stok.utils.pdb.build_pdb`
    - ``.cif`` / ``.mmcif`` → :func:`stok.utils.mmcif.build_mmcif`
    """
    suffix = path.suffix.lower()
    if suffix in (".pdb", ".ent"):
        from stok.utils.pdb import build_pdb

        build_pdb(coordinates=coords, output_file=str(path), sequence=sequence)
    elif suffix in (".cif", ".mmcif"):
        from stok.utils.mmcif import build_mmcif

        build_mmcif(coordinates=coords, output_file=str(path), sequence=sequence)
    else:
        raise ValueError(
            f"Unknown structure format: {suffix!r}. "
            "Expected one of: .pdb, .ent, .cif, .mmcif"
        )


def _write_sample_structures(
    coordinates: torch.Tensor,
    sequences: list[str] | None,
    sample_ids: list[str],
    output_dir: Path,
    format: Literal["pdb", "cif"] = "pdb",
) -> list[Path]:
    """Write per-sample structure files to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    N = coordinates.shape[0]
    for i in range(N):
        path = output_dir / f"{sample_ids[i]}.{format}"
        seq: str | None = None
        if sequences is not None and i < len(sequences):
            seq = sequences[i]
            n_res = int(coordinates[i].shape[0])
            if len(seq) != n_res:
                seq = (seq + "X" * n_res)[:n_res]
        _write_structure(coordinates[i], path, sequence=seq)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Public generation functions
# ---------------------------------------------------------------------------


def design(
    model: MDLMModel,
    *,
    length: int,
    num_samples: int = 10,
    num_steps: int = 100,
    temperature: float = 1.0,
    condition_seq: torch.Tensor | None = None,
    condition_seq_mask: torch.Tensor | None = None,
    condition_struct: torch.Tensor | None = None,
    condition_struct_mask: torch.Tensor | None = None,
    decoder: GeometricDecoder | None = None,
    codebook: torch.Tensor | None = None,
    tokenizer: Tokenizer | None = None,
    output_dir: Path | str | None = None,
    format: Literal["pdb", "cif"] = "pdb",
    return_coordinates: bool = True,
    device: torch.device | str = "cpu",
) -> GenerationResult:
    """Unconditional or (partially) conditioned generation from an MDLM model.

    This is the general-purpose entry point. It subsumes the previous
    ``codesign``/``forward``/``inverse``/``scaffold`` modes by accepting
    explicit condition tensors and boolean "regenerate" masks.

    Args:
        model: A trained MDLM model (``seq_only`` or ``joint``).
        length: Target sequence length.
        num_samples: Number of samples to generate. Ignored when conditioning
            tensors are provided (their batch dim wins).
        num_steps: Iterative unmasking steps.
        temperature: Sampling temperature.
        condition_seq: Optional pre-specified sequence token tensor ``[N, L]``.
        condition_seq_mask: ``[N, L]`` bool mask. ``True`` = regenerate this
            position; ``False`` = hold ``condition_seq`` fixed. Never overload
            ``pad_id`` as a sentinel; pass an explicit mask instead.
        condition_struct: Optional pre-specified structure token tensor
            ``[N, L]``. Joint mode only.
        condition_struct_mask: ``[N, L]`` bool mask for the structure track.
        decoder: Optional :class:`GeometricDecoder`. When provided on a joint
            model, structure tokens are decoded to 3D coordinates.
        codebook: Required when ``decoder`` is provided. Shape ``[C, d_code]``.
        tokenizer: Optional tokenizer for decoding. Defaults to a fresh
            :class:`Tokenizer` (the built-in vocab).
        output_dir: If provided, per-sample PDB files are written here.
        format: Structure file format (``"pdb"`` or ``"cif"``).
        return_coordinates: Whether to populate ``result.coordinates`` when a
            decoder is provided.
        device: Device to run generation on.

    Returns:
        A :class:`GenerationResult` with fields populated based on inputs.

    Raises:
        ValueError: If a decoder is provided for a seq_only model, or if a
            decoder is provided without a codebook.
    """
    device = torch.device(device) if isinstance(device, str) else device
    is_joint = model.tracks == "joint"

    if decoder is not None and not is_joint:
        raise ValueError(
            "Seq-only model cannot produce structure; remove --decoder-preset "
            "or load a joint checkpoint."
        )
    if decoder is not None and codebook is None:
        raise ValueError("codebook is required when decoder is provided.")

    if condition_seq is not None:
        num_samples = int(condition_seq.shape[0])
    elif condition_struct is not None:
        num_samples = int(condition_struct.shape[0])

    # Forward-mode default: if seq is conditioned and struct is neither
    # provided nor masked, mask all struct positions so they get generated.
    if (
        is_joint
        and condition_seq is not None
        and condition_struct is None
        and condition_struct_mask is None
    ):
        condition_struct_mask = torch.ones(
            num_samples, length, dtype=torch.bool, device=device
        )

    result = sample(
        model,
        length=length,
        num_samples=num_samples,
        num_steps=num_steps,
        condition_seq=condition_seq,
        condition_struct=condition_struct,
        seq_mask_positions=condition_seq_mask,
        struct_mask_positions=condition_struct_mask,
        temperature=temperature,
        device=device,
    )

    seq_tokens = result["seq_tokens"].cpu()
    struct_tokens: torch.Tensor | None = None
    if is_joint and "struct_tokens" in result:
        struct_tokens = result["struct_tokens"].cpu()

    if tokenizer is None:
        tokenizer = Tokenizer()
    sequences = _tokens_to_sequences(seq_tokens, tokenizer)

    coordinates: torch.Tensor | None = None
    if decoder is not None and struct_tokens is not None and return_coordinates:
        coordinates = _decode_struct_tokens(
            decoder, codebook, struct_tokens, device  # type: ignore[arg-type]
        ).cpu()

    sample_ids = _make_sample_ids(num_samples)

    structure_paths: list[Path] | None = None
    if output_dir is not None and coordinates is not None:
        structure_paths = _write_sample_structures(
            coordinates=coordinates,
            sequences=sequences,
            sample_ids=sample_ids,
            output_dir=Path(output_dir),
            format=format,
        )

    return GenerationResult(
        sequences=sequences,
        seq_tokens=seq_tokens,
        struct_tokens=struct_tokens,
        coordinates=coordinates,
        structure_paths=structure_paths,
        sample_ids=sample_ids,
        tracks=model.tracks,
    )


def fold(
    model: MDLMModel,
    *,
    sequences: list[str],
    decoder: GeometricDecoder,
    codebook: torch.Tensor,
    tokenizer: Tokenizer,
    length: int | None = None,
    num_steps: int = 100,
    temperature: float = 1.0,
    output_dir: Path | str | None = None,
    format: Literal["pdb", "cif"] = "pdb",
    return_coordinates: bool = True,
    device: torch.device | str = "cpu",
) -> GenerationResult:
    """Sequence → 3D structure. Always requires a joint model + decoder.

    Internally equivalent to ``untokenize(tokenize(sequences))`` but inlined
    to avoid a second tensor round-trip.

    Raises:
        ValueError: If ``model.tracks != "joint"``.
    """
    if model.tracks != "joint":
        raise ValueError(
            "fold requires a joint-track model. Use `design` for seq_only "
            "generation or load a joint checkpoint."
        )
    device = torch.device(device) if isinstance(device, str) else device

    if length is None:
        length = max(len(s) for s in sequences)

    condition_seq = _tokenize_to_tensor(
        sequences, tokenizer, length, model.seq_pad_id, device
    )
    num_samples = condition_seq.shape[0]
    struct_mask = torch.ones(num_samples, length, dtype=torch.bool, device=device)

    return design(
        model,
        length=length,
        num_samples=num_samples,
        num_steps=num_steps,
        temperature=temperature,
        condition_seq=condition_seq,
        condition_struct_mask=struct_mask,
        decoder=decoder,
        codebook=codebook,
        tokenizer=tokenizer,
        output_dir=output_dir,
        format=format,
        return_coordinates=return_coordinates,
        device=device,
    )


def tokenize(
    model: MDLMModel,
    *,
    sequences: list[str],
    tokenizer: Tokenizer,
    length: int | None = None,
    num_steps: int = 100,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
) -> GenerationResult:
    """Sequence → structure tokens. Joint model only. No coords, no files.

    Raises:
        ValueError: If ``model.tracks != "joint"``.
    """
    if model.tracks != "joint":
        raise ValueError(
            "tokenize requires a joint-track model. Use `design` for seq_only "
            "generation or load a joint checkpoint."
        )
    device = torch.device(device) if isinstance(device, str) else device

    if length is None:
        length = max(len(s) for s in sequences)

    condition_seq = _tokenize_to_tensor(
        sequences, tokenizer, length, model.seq_pad_id, device
    )
    num_samples = condition_seq.shape[0]
    struct_mask = torch.ones(num_samples, length, dtype=torch.bool, device=device)

    return design(
        model,
        length=length,
        num_samples=num_samples,
        num_steps=num_steps,
        temperature=temperature,
        condition_seq=condition_seq,
        condition_struct_mask=struct_mask,
        decoder=None,
        codebook=None,
        tokenizer=tokenizer,
        device=device,
    )


def untokenize(
    decoder: GeometricDecoder,
    codebook: torch.Tensor,
    *,
    struct_tokens: torch.Tensor,
    sequences: list[str] | None = None,
    output_dir: Path | str | None = None,
    format: Literal["pdb", "cif"] = "pdb",
    return_coordinates: bool = True,
    device: torch.device | str = "cpu",
) -> GenerationResult:
    """Structure tokens → 3D coordinates.

    Args:
        decoder: Trained :class:`GeometricDecoder`.
        codebook: Codebook tensor ``[C, d_code]`` used during model training.
        struct_tokens: Integer token tensor ``[N, L]`` to decode.
        sequences: Optional matching amino-acid sequences for PDB residue
            names. When ``None``, all residues are written as ``UNK``.
        output_dir: If provided, per-sample structure files are written here.
        format: Structure file format (``"pdb"`` or ``"cif"``).
        return_coordinates: Whether to populate ``result.coordinates``.
        device: Device to run decoding on.
    """
    device = torch.device(device) if isinstance(device, str) else device
    struct_tokens_cpu = struct_tokens.cpu()
    B, L = struct_tokens_cpu.shape

    coordinates: torch.Tensor | None = None
    if return_coordinates:
        coordinates = _decode_struct_tokens(
            decoder, codebook, struct_tokens.to(device), device
        ).cpu()

    placeholder_sequences: list[str]
    if sequences is None:
        placeholder_sequences = ["X" * L] * B
    else:
        placeholder_sequences = list(sequences)

    # No seq tokens were used; expose an empty tensor so callers have a
    # consistent shape. Avoids surprising None handling downstream.
    seq_tokens = torch.zeros(B, L, dtype=torch.long)
    sample_ids = _make_sample_ids(B)

    structure_paths: list[Path] | None = None
    if output_dir is not None and coordinates is not None:
        structure_paths = _write_sample_structures(
            coordinates=coordinates,
            sequences=placeholder_sequences,
            sample_ids=sample_ids,
            output_dir=Path(output_dir),
            format=format,
        )

    return GenerationResult(
        sequences=placeholder_sequences,
        seq_tokens=seq_tokens,
        struct_tokens=struct_tokens_cpu,
        coordinates=coordinates,
        structure_paths=structure_paths,
        sample_ids=sample_ids,
        tracks="joint",
    )


def unfold(*args, **kwargs) -> GenerationResult:
    """Structure → sequence. **Not yet implemented.**

    Blocked on integrating a pretrained coordinates→tokens encoder. Use
    :func:`untokenize` if you already have structure tokens.
    """
    raise NotImplementedError(
        "unfold requires a pretrained coordinates->tokens encoder, which is "
        "not yet integrated. Use `untokenize` if you already have structure "
        "tokens, or wait for encoder support."
    )


def encode(
    structures: list[Path | str] | Path | str,
    *,
    encoder: StructureEncoder,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
    output_path: Path | str | None = None,
) -> EncodeResult:
    """PDB/mmCIF file(s) → VQ structure token indices.

    Runs a pretrained :class:`~stok.models.structure_encoder.StructureEncoder`
    over one or more structures and returns per-sample token indices trimmed
    to each accepted sample's true sequence length. Upstream-rejected samples,
    including proteins longer than the encoder's ``max_length``, are omitted.

    Args:
        structures: A single file path, a directory (recursively scanned for
            ``.pdb`` / ``.ent`` / ``.cif`` / ``.mmcif``), or a list of paths.
        encoder: A pretrained :class:`StructureEncoder` (typically built via
            :func:`load_encoder`).
        batch_size: Mini-batch size for encoding. Larger batches give better
            throughput at the cost of memory.
        device: Device to run the encoder on. The encoder is moved here
            before encoding.
        output_path: Optional Parquet file to write results to. The schema
            is ``(sample_id, sequence, struct_tokens, length)`` — compatible
            with :func:`untokenize` downstream.

    Returns:
        An :class:`EncodeResult` with one row per accepted structure sample.
    """
    from stok.utils.structure_loader import NoAcceptedStructuresError, load_structures

    device = torch.device(device) if isinstance(device, str) else device
    encoder = encoder.to(device).eval()

    if isinstance(structures, (str, Path)):
        structures_list: list[Path | str] = [structures]
    else:
        structures_list = list(structures)

    sample_ids: list[str] = []
    sequences: list[str] = []
    struct_tokens_out: list[list[int]] = []

    max_length = encoder.max_length

    for start in range(0, len(structures_list), batch_size):
        batch_paths = structures_list[start : start + batch_size]
        try:
            loaded = load_structures(
                batch_paths,
                max_length=max_length,
                device=device,
            )
        except NoAcceptedStructuresError:
            continue

        with torch.inference_mode():
            out = encoder(loaded.graph, loaded.mask, loaded.nan_mask)

        indices_cpu = out["indices"].detach().cpu()
        valid_cpu = out["valid"].detach().cpu()

        for i, (pid, seq) in enumerate(zip(loaded.pids, loaded.sequences)):
            length = int(valid_cpu[i].sum().item())
            tokens = indices_cpu[i, :length].tolist()
            sample_ids.append(pid)
            sequences.append(seq)
            struct_tokens_out.append(tokens)

    result = EncodeResult(
        sample_ids=sample_ids,
        sequences=sequences,
        struct_tokens=struct_tokens_out,
    )

    if output_path is not None:
        _write_encode_manifest(result, Path(output_path))

    return result


def _write_encode_manifest(result: EncodeResult, output_path: Path) -> None:
    """Serialize an :class:`EncodeResult` to the shared manifest schema.

    Columns: ``sample_id``, ``sequence``, ``struct_tokens``, ``length``.
    The Parquet schema overlaps with :func:`_write_manifest` enough that
    :func:`untokenize` can ingest the output directly.
    """
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "sample_id": sid,
            "sequence": seq,
            "struct_tokens": tokens,
            "length": len(tokens),
        }
        for sid, seq, tokens in zip(
            result.sample_ids, result.sequences, result.struct_tokens
        )
    ]
    pd.DataFrame(records).to_parquet(output_path, index=False)
