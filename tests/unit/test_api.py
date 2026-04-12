"""Direct tests for the public :mod:`stok.api` surface.

These tests bypass Click / Hydra entirely and exercise each API function
with tiny in-memory MDLM checkpoints. They also lock in the two previously
silent bugs in the structure-decoding path (missing codebook attribute and
missing ``mask`` argument to the geometric decoder).
"""

from __future__ import annotations

import pytest
import torch

from stok.api import (
    GenerationResult,
    LoadedModel,
    MDLMModelConfig,
    NoiseScheduleConfig,
    design,
    fold,
    load_decoder,
    load_model,
    tokenize,
    unfold,
    untokenize,
)
from stok.models.mdlm import MDLMModel
from stok.models.noise_schedule import NoiseSchedule


CODEBOOK_SIZE = 16
CODEBOOK_DIM = 8
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
FFN_MULT = 2.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_seq_only_model() -> MDLMModel:
    ns = NoiseSchedule(schedule_type="cosine")
    return MDLMModel(
        tracks="seq_only",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ffn_mult=FFN_MULT,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns,
        time_conditioning="adaln",
    )


def _build_joint_model(codebook: torch.Tensor) -> MDLMModel:
    ns_seq = NoiseSchedule(schedule_type="cosine")
    ns_struct = NoiseSchedule(schedule_type="cosine")
    return MDLMModel(
        tracks="joint",
        seq_vocab_size=32,
        seq_pad_id=1,
        seq_mask_id=31,
        codebook=codebook,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ffn_mult=FFN_MULT,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=ns_seq,
        noise_schedule_struct=ns_struct,
        time_conditioning="adaln",
    )


def _seq_only_config() -> MDLMModelConfig:
    return MDLMModelConfig(
        tracks="seq_only",
        seq_vocab_size=32,
        seq_pad_id=1,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ffn_mult=FFN_MULT,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=NoiseScheduleConfig(type="cosine"),
        noise_schedule_struct=None,
        codebook_preset=None,
    )


def _joint_config(codebook_path) -> MDLMModelConfig:
    return MDLMModelConfig(
        tracks="joint",
        seq_vocab_size=32,
        seq_pad_id=1,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ffn_mult=FFN_MULT,
        dropout=0.0,
        attn_dropout=0.0,
        noise_schedule_seq=NoiseScheduleConfig(type="cosine"),
        noise_schedule_struct=NoiseScheduleConfig(type="cosine"),
        codebook_preset=None,
        codebook_path=codebook_path,
        classifier_kwargs=dict(
            use_cosine=False,
            learnable_temperature=True,
            bias_from_code_norm=True,
            projector_dim=None,
        ),
    )


@pytest.fixture
def seq_only_loaded(tmp_path) -> LoadedModel:
    model = _build_seq_only_model()
    ckpt_path = tmp_path / "seq_only.pt"
    torch.save(model.state_dict(), ckpt_path)
    return load_model(ckpt_path, config=_seq_only_config(), device="cpu")


@pytest.fixture
def joint_codebook(tmp_path) -> tuple[torch.Tensor, "object"]:
    codebook = torch.randn(CODEBOOK_SIZE, CODEBOOK_DIM) * 0.1
    codebook_path = tmp_path / "codebook.pt"
    torch.save(codebook, codebook_path)
    return codebook, codebook_path


@pytest.fixture
def joint_loaded(tmp_path, joint_codebook) -> LoadedModel:
    codebook, codebook_path = joint_codebook
    model = _build_joint_model(codebook)
    ckpt_path = tmp_path / "joint.pt"
    torch.save(model.state_dict(), ckpt_path)
    return load_model(ckpt_path, config=_joint_config(codebook_path), device="cpu")


@pytest.fixture
def tiny_decoder():
    """A tiny randomly-initialized GeometricDecoder matching CODEBOOK_DIM."""
    pytest.importorskip("x_transformers")
    from stok.models.decoder import GeometricDecoder

    dec = GeometricDecoder(
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_mult=2.0,
        max_length=64,
        d_code=CODEBOOK_DIM,
        num_memory_tokens=0,
        attn_kv_heads=2,
    )
    dec.eval()
    for p in dec.parameters():
        p.requires_grad = False
    return dec


# ---------------------------------------------------------------------------
# load_model / load_decoder
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_seq_only_roundtrip(self, seq_only_loaded):
        assert isinstance(seq_only_loaded, LoadedModel)
        assert seq_only_loaded.model.tracks == "seq_only"
        assert seq_only_loaded.codebook is None
        assert seq_only_loaded.tokenizer is not None

    def test_joint_roundtrip(self, joint_loaded, joint_codebook):
        codebook, _ = joint_codebook
        assert isinstance(joint_loaded, LoadedModel)
        assert joint_loaded.model.tracks == "joint"
        assert joint_loaded.codebook is not None
        assert joint_loaded.codebook.shape == codebook.shape

    def test_state_dict_keys_and_shapes_match(self, tmp_path, joint_codebook):
        """Ensure from-config reconstruction yields the identical state dict."""
        codebook, codebook_path = joint_codebook
        original = _build_joint_model(codebook)
        ckpt_path = tmp_path / "joint.pt"
        torch.save(original.state_dict(), ckpt_path)

        loaded = load_model(
            ckpt_path, config=_joint_config(codebook_path), device="cpu"
        )
        orig_keys = set(original.state_dict().keys())
        new_keys = set(loaded.model.state_dict().keys())
        assert orig_keys == new_keys
        for k in orig_keys:
            assert original.state_dict()[k].shape == loaded.model.state_dict()[k].shape

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(
                tmp_path / "nope.pt", config=_seq_only_config(), device="cpu"
            )


class TestLoadDecoder:
    def test_is_callable(self):
        assert callable(load_decoder)


# ---------------------------------------------------------------------------
# design — behavior matrix
# ---------------------------------------------------------------------------


class TestDesignBehaviorMatrix:
    def test_seq_only_no_decoder_generates_sequences(self, seq_only_loaded):
        result = design(
            seq_only_loaded.model,
            length=8,
            num_samples=2,
            num_steps=3,
            tokenizer=seq_only_loaded.tokenizer,
        )
        assert isinstance(result, GenerationResult)
        assert result.tracks == "seq_only"
        assert result.seq_tokens.shape == (2, 8)
        assert result.struct_tokens is None
        assert result.coordinates is None
        assert len(result.sequences) == 2
        assert all(isinstance(s, str) and len(s) > 0 for s in result.sequences)

    def test_seq_only_with_decoder_raises(self, seq_only_loaded, tiny_decoder):
        with pytest.raises(ValueError, match="Seq-only model cannot produce structure"):
            design(
                seq_only_loaded.model,
                length=8,
                num_samples=1,
                num_steps=3,
                decoder=tiny_decoder,
                codebook=torch.randn(CODEBOOK_SIZE, CODEBOOK_DIM),
            )

    def test_joint_no_decoder_generates_both_tracks(self, joint_loaded):
        result = design(
            joint_loaded.model,
            length=8,
            num_samples=2,
            num_steps=3,
            tokenizer=joint_loaded.tokenizer,
        )
        assert result.tracks == "joint"
        assert result.seq_tokens.shape == (2, 8)
        assert result.struct_tokens is not None
        assert result.struct_tokens.shape == (2, 8)
        assert result.coordinates is None
        assert result.structure_paths is None

    def test_joint_with_decoder_produces_nonzero_coords(
        self, joint_loaded, joint_codebook, tiny_decoder
    ):
        """Locks in the fix for the silent decode-path bugs."""
        _, _ = joint_codebook
        result = design(
            joint_loaded.model,
            length=8,
            num_samples=2,
            num_steps=3,
            decoder=tiny_decoder,
            codebook=joint_loaded.codebook,
            tokenizer=joint_loaded.tokenizer,
        )
        assert result.coordinates is not None
        assert result.coordinates.shape == (2, 8, 3, 3)
        assert torch.isfinite(result.coordinates).all()
        assert result.coordinates.abs().max().item() > 0.0


class TestDesignReproducibility:
    def test_fixed_seed_gives_identical_output(self, seq_only_loaded):
        torch.manual_seed(0)
        r1 = design(
            seq_only_loaded.model,
            length=8,
            num_samples=2,
            num_steps=3,
            tokenizer=seq_only_loaded.tokenizer,
        )
        torch.manual_seed(0)
        r2 = design(
            seq_only_loaded.model,
            length=8,
            num_samples=2,
            num_steps=3,
            tokenizer=seq_only_loaded.tokenizer,
        )
        assert torch.equal(r1.seq_tokens, r2.seq_tokens)


class TestDesignScaffolding:
    def test_explicit_condition_mask_preserves_unmasked_positions(
        self, seq_only_loaded
    ):
        """Full-length condition_seq + mask=True on first half: first half
        regenerated, second half preserved verbatim.
        """
        B, L = 2, 10
        half = L // 2
        torch.manual_seed(42)
        condition_seq = torch.randint(4, 24, (B, L), dtype=torch.long)
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, :half] = True

        result = design(
            seq_only_loaded.model,
            length=L,
            num_samples=B,
            num_steps=5,
            condition_seq=condition_seq,
            condition_seq_mask=mask,
            tokenizer=seq_only_loaded.tokenizer,
        )
        # The preserved half must be byte-identical.
        assert torch.equal(
            result.seq_tokens[:, half:], condition_seq[:, half:]
        )
        # No mask tokens should remain anywhere in the preserved half.
        assert (result.seq_tokens != seq_only_loaded.model.seq_mask_id).all()


class TestDesignJointForwardMode:
    def test_condition_seq_freezes_seq_track(self, joint_loaded):
        """Joint model with condition_seq but no struct should hold seq fixed."""
        B, L = 2, 8
        torch.manual_seed(0)
        condition_seq = torch.randint(4, 24, (B, L), dtype=torch.long)
        result = design(
            joint_loaded.model,
            length=L,
            num_samples=B,
            num_steps=3,
            condition_seq=condition_seq,
            tokenizer=joint_loaded.tokenizer,
        )
        # Seq should be held fixed (no mask positions provided).
        assert torch.equal(result.seq_tokens, condition_seq)
        # Struct should have been generated (no mask tokens remaining).
        assert result.struct_tokens is not None
        assert (result.struct_tokens != joint_loaded.model.struct_mask_id).all()


# ---------------------------------------------------------------------------
# fold / tokenize / untokenize / unfold
# ---------------------------------------------------------------------------


class TestFold:
    def test_seq_only_raises(self, seq_only_loaded, tiny_decoder):
        with pytest.raises(ValueError, match="joint"):
            fold(
                seq_only_loaded.model,
                sequences=["ACDE"],
                decoder=tiny_decoder,
                codebook=torch.randn(CODEBOOK_SIZE, CODEBOOK_DIM),
                tokenizer=seq_only_loaded.tokenizer,
            )

    def test_joint_produces_coordinates(self, joint_loaded, tiny_decoder):
        result = fold(
            joint_loaded.model,
            sequences=["ACDEFGHI", "KLMNPQRS"],
            decoder=tiny_decoder,
            codebook=joint_loaded.codebook,
            tokenizer=joint_loaded.tokenizer,
            num_steps=3,
        )
        assert result.coordinates is not None
        assert result.coordinates.shape == (2, 8, 3, 3)
        assert torch.isfinite(result.coordinates).all()
        assert result.coordinates.abs().max().item() > 0.0


class TestTokenize:
    def test_joint_produces_struct_tokens(self, joint_loaded):
        result = tokenize(
            joint_loaded.model,
            sequences=["ACDE"],
            tokenizer=joint_loaded.tokenizer,
            num_steps=3,
        )
        assert result.struct_tokens is not None
        assert result.struct_tokens.shape[0] == 1
        assert result.coordinates is None

    def test_seq_only_raises(self, seq_only_loaded):
        with pytest.raises(ValueError, match="joint"):
            tokenize(
                seq_only_loaded.model,
                sequences=["ACDE"],
                tokenizer=seq_only_loaded.tokenizer,
            )


class TestUntokenize:
    def test_basic_decode(self, joint_loaded, tiny_decoder):
        B, L = 2, 6
        struct_tokens = torch.randint(0, CODEBOOK_SIZE, (B, L))
        result = untokenize(
            tiny_decoder,
            joint_loaded.codebook,
            struct_tokens=struct_tokens,
            sequences=["ACDEFG", "HIKLMN"],
        )
        assert result.coordinates is not None
        assert result.coordinates.shape == (B, L, 3, 3)
        assert torch.isfinite(result.coordinates).all()
        assert result.coordinates.abs().max().item() > 0.0


class TestRoundTrip:
    def test_tokenize_then_untokenize_produces_nonzero_coords(
        self, joint_loaded, tiny_decoder
    ):
        tok = tokenize(
            joint_loaded.model,
            sequences=["ACDE"],
            tokenizer=joint_loaded.tokenizer,
            num_steps=3,
        )
        assert tok.struct_tokens is not None

        coords_result = untokenize(
            tiny_decoder,
            joint_loaded.codebook,
            struct_tokens=tok.struct_tokens,
            sequences=["ACDE"],
        )
        assert coords_result.coordinates is not None
        assert coords_result.coordinates.abs().max().item() > 0.0


class TestUnfold:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="untokenize"):
            unfold()


# ---------------------------------------------------------------------------
# output_dir file writing (Phase 1: PDB only)
# ---------------------------------------------------------------------------


class TestOutputDirWriting:
    def test_design_with_output_dir_writes_pdb_files(
        self, tmp_path, joint_loaded, tiny_decoder
    ):
        out_dir = tmp_path / "design_out"
        result = design(
            joint_loaded.model,
            length=6,
            num_samples=2,
            num_steps=3,
            decoder=tiny_decoder,
            codebook=joint_loaded.codebook,
            tokenizer=joint_loaded.tokenizer,
            output_dir=out_dir,
        )
        assert result.structure_paths is not None
        assert len(result.structure_paths) == 2
        for p in result.structure_paths:
            assert p.exists()
            assert p.suffix == ".pdb"
            assert p.read_text().startswith(("HEADER", "ATOM", "MODEL", "TITLE"))

    def test_untokenize_with_output_dir_writes_pdb_files(
        self, tmp_path, joint_loaded, tiny_decoder
    ):
        out_dir = tmp_path / "untok_out"
        struct_tokens = torch.randint(0, CODEBOOK_SIZE, (2, 6))
        result = untokenize(
            tiny_decoder,
            joint_loaded.codebook,
            struct_tokens=struct_tokens,
            sequences=["ACDEFG", "HIKLMN"],
            output_dir=out_dir,
        )
        assert result.structure_paths is not None
        assert len(result.structure_paths) == 2
        for p in result.structure_paths:
            assert p.exists()
