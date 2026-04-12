"""MDLM generation: load checkpoint, run iterative unmasking, save results."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    cfg: DictConfig,
    device: torch.device,
):
    """Build an MDLMModel from config and load weights from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint (.pt).
        cfg: Full Hydra config (must contain model and train.mdlm sections).
        device: Device to place the model on.

    Returns:
        An MDLMModel with loaded weights in eval mode.
    """
    from stok.models.mdlm import MDLMModel
    from stok.models.noise_schedule import NoiseSchedule
    from stok.utils.codebook import load_codebook
    from stok.utils.tokenizer import Tokenizer

    mdlm_cfg = cfg.train.mdlm
    tracks = str(mdlm_cfg.get("tracks", "joint"))

    # Build noise schedules
    ns_seq_cfg = mdlm_cfg.get("noise_schedule_seq", {})
    noise_schedule_seq = NoiseSchedule(
        schedule_type=str(ns_seq_cfg.get("type", "cosine")),
        sigmoid_k=float(ns_seq_cfg.get("sigmoid_k", 6.0)),
        log_linear_k=float(ns_seq_cfg.get("log_linear_k", 3.0)),
        eps=float(ns_seq_cfg.get("eps", 1e-5)),
    )

    noise_schedule_struct = None
    codebook = None
    if tracks == "joint":
        ns_struct_cfg = mdlm_cfg.get("noise_schedule_struct", {})
        noise_schedule_struct = NoiseSchedule(
            schedule_type=str(ns_struct_cfg.get("type", "cosine")),
            sigmoid_k=float(ns_struct_cfg.get("sigmoid_k", 6.0)),
            log_linear_k=float(ns_struct_cfg.get("log_linear_k", 3.0)),
            eps=float(ns_struct_cfg.get("eps", 1e-5)),
        )
        codebook = load_codebook(
            preset=cfg.model.codebook.get("preset"),
            path=cfg.model.codebook.get("path"),
        )

    tokenizer = Tokenizer()

    model = MDLMModel(
        tracks=tracks,
        seq_vocab_size=cfg.model.encoder.vocab_size,
        seq_pad_id=cfg.model.encoder.pad_id,
        seq_mask_id=tokenizer.mask_token_id,
        codebook=codebook,
        d_model=cfg.model.encoder.d_model,
        n_heads=cfg.model.encoder.n_heads,
        n_layers=cfg.model.encoder.n_layers,
        ffn_mult=cfg.model.encoder.ffn_mult,
        dropout=cfg.model.encoder.dropout,
        attn_dropout=cfg.model.encoder.attn_dropout,
        norm_type=cfg.model.encoder.norm,
        noise_schedule_seq=noise_schedule_seq,
        noise_schedule_struct=noise_schedule_struct,
        lambda_seq=float(mdlm_cfg.get("lambda_seq", 1.0)),
        lambda_struct=float(mdlm_cfg.get("lambda_struct", 1.0)),
        classifier_kwargs=(
            dict(
                use_cosine=cfg.model.classifier.use_cosine,
                learnable_temperature=cfg.model.classifier.learnable_temperature,
                bias_from_code_norm=cfg.model.classifier.bias_from_code_norm,
                projector_dim=cfg.model.classifier.projector_dim,
            )
            if tracks == "joint"
            else None
        ),
        tie_seq_embeddings=bool(mdlm_cfg.get("tie_seq_embeddings", True)),
        time_conditioning=str(mdlm_cfg.get("time_conditioning", "adaln")),
        time_embed_dim=(
            int(mdlm_cfg.time_embed_dim) if mdlm_cfg.get("time_embed_dim") else None
        ),
        time_combine=str(mdlm_cfg.get("time_combine", "sum")),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer


def _read_conditioning_file(path: str) -> list[str]:
    """Read sequences from a FASTA or plain-text file.

    Supports:
    - FASTA format (lines starting with '>')
    - Plain text (one sequence per line)

    Args:
        path: Path to input file.

    Returns:
        List of sequence strings.
    """
    sequences: list[str] = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.strip()
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


def _decode_tokens_to_sequences(
    tokens: torch.Tensor,
    tokenizer,
) -> list[str]:
    """Convert token IDs back to amino acid sequences.

    Args:
        tokens: Token tensor of shape [B, L].
        tokenizer: Tokenizer instance for decoding.

    Returns:
        List of decoded sequence strings.
    """
    sequences = []
    for i in range(tokens.shape[0]):
        ids = tokens[i].tolist()
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        # Remove spaces introduced by tokenizer decoding
        decoded = decoded.replace(" ", "")
        sequences.append(decoded)
    return sequences


def run_generation(cfg: DictConfig):
    """Run MDLM generation from config.

    Loads a checkpoint, runs iterative unmasking sampling, optionally
    decodes structure tokens to coordinates, and saves results to Parquet.

    Args:
        cfg: Full Hydra config with generation-specific overrides.
    """
    import pandas as pd

    from stok.utils.sampling import sample

    gen_cfg = cfg.get("generate", {})
    checkpoint_path = Path(str(gen_cfg.get("checkpoint", "")))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    mode = str(gen_cfg.get("mode", "codesign"))
    length = int(gen_cfg.get("length", 100))
    num_samples = int(gen_cfg.get("num_samples", 10))
    num_steps = int(gen_cfg.get("num_steps", 100))
    temperature = float(gen_cfg.get("temperature", 1.0))
    output_path = Path(str(gen_cfg.get("output", "generated.parquet")))
    condition_seq_file = gen_cfg.get("condition_seq_file")
    condition_struct_file = gen_cfg.get("condition_struct_file")
    decoder_preset = str(gen_cfg.get("decoder_preset", "base"))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {checkpoint_path}")
    model, tokenizer = _load_model_from_checkpoint(checkpoint_path, cfg, device)
    is_joint = model.tracks == "joint"
    print(f"Model loaded (tracks={model.tracks}, device={device})")

    # Prepare conditioning
    condition_seq = None
    condition_struct = None
    seq_mask_positions = None
    struct_mask_positions = None

    if condition_seq_file is not None:
        sequences = _read_conditioning_file(str(condition_seq_file))
        encoded = [tokenizer.encode(s, add_special_tokens=False) for s in sequences]
        # Pad/truncate to length
        padded = []
        for enc in encoded:
            ids = enc[:length]
            ids = ids + [model.seq_pad_id] * (length - len(ids))
            padded.append(ids)
        condition_seq = torch.tensor(padded, dtype=torch.long, device=device)
        num_samples = condition_seq.shape[0]

    if condition_struct_file is not None and is_joint:
        # Struct file: one line per sample, space-separated integer indices
        struct_seqs = []
        with open(str(condition_struct_file)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                indices = [int(x) for x in line.split()]
                indices = indices[:length]
                indices = indices + [model.struct_pad_id] * (length - len(indices))
                struct_seqs.append(indices)
        condition_struct = torch.tensor(struct_seqs, dtype=torch.long, device=device)
        num_samples = condition_struct.shape[0]

    # Set up mask positions based on mode
    if mode == "codesign":
        # Both tracks fully masked — generate everything (default behavior)
        pass
    elif mode == "forward":
        # Condition on sequence, generate structure
        if condition_seq is None:
            raise ValueError("Forward mode requires --condition-seq-file")
        struct_mask_positions = torch.ones(
            num_samples, length, dtype=torch.bool, device=device
        )
    elif mode == "inverse":
        # Condition on structure, generate sequence
        if condition_struct is None:
            raise ValueError("Inverse mode requires --condition-struct-file")
        seq_mask_positions = torch.ones(
            num_samples, length, dtype=torch.bool, device=device
        )
    elif mode == "scaffold":
        # Partial conditioning — positions not in condition files get generated
        if condition_seq is not None:
            # Generate positions that are padding (not provided)
            seq_mask_positions = condition_seq == model.seq_pad_id
        if condition_struct is not None:
            struct_mask_positions = condition_struct == model.struct_pad_id
    else:
        raise ValueError(
            f"Unknown generation mode: {mode!r}. "
            "Expected 'codesign', 'forward', 'inverse', or 'scaffold'."
        )

    print(
        f"Generating {num_samples} samples (mode={mode}, length={length}, "
        f"steps={num_steps}, temperature={temperature})"
    )

    result = sample(
        model,
        length=length,
        num_samples=num_samples,
        num_steps=num_steps,
        condition_seq=condition_seq,
        condition_struct=condition_struct,
        seq_mask_positions=seq_mask_positions,
        struct_mask_positions=struct_mask_positions,
        temperature=temperature,
        device=device,
    )

    # Decode tokens to sequences
    seq_tokens = result["seq_tokens"].cpu()
    sequences = _decode_tokens_to_sequences(seq_tokens, tokenizer)

    # Build output dataframe
    records = []
    for i in range(num_samples):
        record = {
            "pid": f"generated_{i:04d}",
            "sequence": sequences[i],
            "seq_tokens": seq_tokens[i].tolist(),
        }
        if is_joint and "struct_tokens" in result:
            struct_tokens = result["struct_tokens"].cpu()
            record["struct_tokens"] = struct_tokens[i].tolist()
        records.append(record)

    df = pd.DataFrame(records)

    # Optionally decode structure tokens to coordinates
    if is_joint and "struct_tokens" in result and gen_cfg.get("decode_structure", False):
        try:
            from stok.models.decoder import load_pretrained_decoder

            decoder = load_pretrained_decoder(
                preset=decoder_preset, device=device, freeze=True
            )
            struct_tokens = result["struct_tokens"]  # [B, L]
            codebook = model.head_struct.codebook  # [C, d_code]
            # Look up codebook vectors
            # Clamp to valid codebook range
            valid_indices = struct_tokens.clamp(0, codebook.shape[0] - 1)
            code_vectors = codebook[valid_indices]  # [B, L, d_code]
            coords = decoder(code_vectors)  # [B, L, 3, 3]
            coords_list = coords.cpu().tolist()
            df["coordinates"] = coords_list
            print(f"Decoded structure tokens to coordinates using '{decoder_preset}' decoder")
        except Exception as e:
            logger.warning("Structure decoding failed: %s", e)
            print(f"Warning: structure decoding failed ({e}), skipping coordinates")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")
