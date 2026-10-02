"""Grouped reverse sampling follows the absorbing finite-state process."""

import itertools
import random

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from stok.data.mdlm import CANONICAL_AA, prepare_mdlm_batch
from stok.models.mdlm import STokMDLM
from stok.utils.mdlm import build_mask_groups
from stok.utils.tokenizer import DEFAULT_VOCAB, Tokenizer


class FixedDenoiser(nn.Module):
    """Known member probabilities make the sampler's transition measurable."""

    def __init__(self, tokenizer, fail=False):
        super().__init__()
        self.pad_id = tokenizer.pad_token_id
        self.embed = nn.Embedding(len(tokenizer), 2)
        self.codebook_size = 2
        self.mdlm_regime_weights = {"joint_tied": 1}
        self.calls = []
        self.fail = fail

    def forward(self, sequence_tokens, structure_tokens):
        self.calls.append((sequence_tokens.clone(), structure_tokens.clone()))
        random.random()
        np.random.rand()
        torch.rand(1)
        if self.fail:
            raise RuntimeError("denoiser failed")
        seq = torch.full((*sequence_tokens.shape, self.embed.num_embeddings), 100.0)
        seq[..., self.aa[0]], seq[..., self.aa[1]] = np.log(0.8), np.log(0.2)
        seq[..., self.aa[2:]] = -1000
        struct = torch.tensor([0.3, 0.7]).log().expand(*sequence_tokens.shape, 2)
        return {"sequence_logits": seq, "structure_logits": struct}


def fixture(sequence="A", labels=None, count=1):
    tokenizer = Tokenizer()
    batch = prepare_mdlm_batch(
        [
            {
                "dataset": "fixture",
                "sequence_id": str(i),
                "sequence": sequence,
                "structure_tokens": labels or [1] * len(sequence),
            }
            for i in range(count)
        ],
        tokenizer,
        max_len=len(sequence) + 3,
        codebook_size=2,
        crop="center",
        seeds=[0] * count,
    )
    aa = torch.tensor(tokenizer.convert_tokens_to_ids(list(CANONICAL_AA)))
    model = FixedDenoiser(tokenizer)
    model.aa = aa
    return model, batch, aa


def groups(batch, generate, placement="token", tied=True):
    return torch.stack(
        [
            build_mask_groups(
                mask,
                residue,
                placement=placement,
                tied=tied,
                span_mean=3,
                generator=torch.Generator().manual_seed(4),
            )
            for mask, residue in zip(generate, batch["residue_mask"])
        ]
    )


def run(model, batch, aa, generate, *, seeds=None, ids=None, **kwargs):
    from stok.utils.sampling import sample_mdlm

    return sample_mdlm(
        model,
        batch,
        generate_mask=generate,
        group_ids=groups(batch, generate) if ids is None else ids,
        schedule=OmegaConf.create({"name": "linear", "power": 2}),
        steps=kwargs.pop("steps", 4),
        seeds=list(range(len(generate))) if seeds is None else seeds,
        canonical_aa_ids=aa,
        **kwargs,
    )


def test_reverse_kernel_matches_finite_state_reference():
    # Four clean states + one absorbing group; tuple probability factors.
    count = 5000
    model, batch, aa = fixture(count=count)
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2).clone()
    ids = groups(batch, generate)
    original_ids = ids.clone()
    result = run(
        model,
        batch,
        aa,
        generate,
        ids=ids,
        steps=3,
        time_grid=torch.tensor([1.0, 0.75, 0.25, 0.0]),
    )
    assert len(model.calls) == 3
    member = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float64)
    states = list(itertools.product(range(2), repeat=2))
    clean = torch.tensor([member[0, a] * member[1, b] for a, b in states])
    # t=.75 -> s=.25: 2/3 reveal probability, as in Task3 Bayes reference.
    kernel = torch.eye(5, dtype=torch.float64)
    kernel[4, :4], kernel[4, 4] = (2 / 3) * clean, 1 / 3
    torch.testing.assert_close(kernel.sum(1), torch.ones(5, dtype=torch.float64))
    first, second = model.calls[1], model.calls[2]
    hidden = first[0][:, 1].eq(batch["sequence_mask_id"])
    revealed = second[0][:, 1].ne(batch["sequence_mask_id"])
    assert abs(float(revealed[hidden].float().mean()) - 2 / 3) < 0.03
    assert torch.equal(revealed, second[1][:, 1].ne(3))
    for i, (a, b) in enumerate(states):
        frequency = (
            (second[0][:, 1] == aa[a]) & (second[1][:, 1] == b) & hidden
        ).sum() / hidden.sum()
        assert abs(float(frequency) - float(kernel[4, i])) < 0.025
    for previous, current in zip(model.calls, model.calls[1:]):
        for track, sentinel in enumerate([batch["sequence_mask_id"], 3]):
            visible = generate[..., track] & previous[track].ne(sentinel)
            assert torch.equal(current[track][visible], previous[track][visible])
    for track, name in enumerate(["sequence_tokens", "structure_tokens"]):
        visible = generate[..., track] & model.calls[-1][track].ne(
            [batch["sequence_mask_id"], 3][track]
        )
        assert torch.equal(result[name][visible], model.calls[-1][track][visible])
    assert torch.equal(ids, original_ids)


@pytest.mark.parametrize("tracks", [(0,), (1,), (0, 1)])
@pytest.mark.parametrize(
    "placement,tied",
    [("token", False), ("token", True), ("span", False), ("span", True)],
)
def test_conditions_completion_missing_observations_and_no_clean_leak(
    tracks, placement, tied
):
    model, batch, aa = fixture("ACDE", [0, None, 1, 0])
    generate = torch.zeros((*batch["residue_mask"].shape, 2), dtype=torch.bool)
    for track in tracks:
        generate[..., track] = batch["residue_mask"]
    result = run(
        model, batch, aa, generate, ids=groups(batch, generate, placement, tied)
    )
    for track, name in enumerate(["sequence_tokens", "structure_tokens"]):
        assert torch.equal(
            result[name][~generate[..., track]], batch[name][~generate[..., track]]
        )
        for inputs in model.calls:
            assert torch.equal(
                inputs[track][~generate[..., track]], batch[name][~generate[..., track]]
            )
        assert (
            model.calls[0][track][generate[..., track]]
            == [batch["sequence_mask_id"], 3][track]
        ).all()
    assert not batch["structure_valid"][0, 2]
    if 1 in tracks:
        assert (
            (result["structure_tokens"][generate[..., 1]] >= 0)
            & (result["structure_tokens"][generate[..., 1]] < 2)
        ).all()
    if 0 in tracks:
        assert torch.isin(result["sequence_tokens"][generate[..., 0]], aa).all()


def test_per_sample_draws_ignore_batch_order_and_preserve_inputs():
    model, batch, aa = fixture("ACD", count=3)
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2).clone()
    first = run(model, batch, aa, generate, seeds=[17, 28, 39])
    order = torch.tensor([2, 0, 1])
    reordered = {
        key: value[order]
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 3
        else value
        for key, value in batch.items()
    }
    second = run(model, reordered, aa, generate[order], seeds=[39, 17, 28])
    for name in ["sequence_tokens", "structure_tokens"]:
        assert torch.equal(first[name][order], second[name])
        singleton = {
            key: value[:1]
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 3
            else value
            for key, value in batch.items()
        }
        solo = run(model, singleton, aa, generate[:1], seeds=[17])
        assert torch.equal(first[name][:1], solo[name])
    assert batch["sequence_tokens"][0, 1] == aa[0]


@pytest.mark.parametrize("fail", [False, True])
def test_preserves_rng_and_each_module_mode_on_success_and_failure(fail):
    model, batch, aa = fixture()
    model.fail = fail
    model.train()
    model.embed.eval()
    python, numpy, tensor = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2).clone()
    if fail:
        with pytest.raises(RuntimeError, match="denoiser failed"):
            run(model, batch, aa, generate)
    else:
        run(model, batch, aa, generate)
    assert model.training and not model.embed.training
    assert python == random.getstate()
    restored = np.random.get_state()
    assert (
        numpy[0] == restored[0]
        and np.array_equal(numpy[1], restored[1])
        and numpy[2:] == restored[2:]
    )
    assert torch.equal(tensor, torch.get_rng_state())


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_rejects_invalid_steps(steps):
    model, batch, aa = fixture()
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2)
    with pytest.raises(ValueError, match="steps"):
        run(model, batch, aa, generate, steps=steps)


@pytest.mark.parametrize(
    "grid",
    [
        [1, 0.5, 0.5, 0, 0],
        [1, 0.2, 0.3, 0.1, 0],
        [0.9, 0.7, 0.4, 0.1, 0],
        [1, 0.7, 0.4, 0.1, 0.01],
        [1, float("nan"), 0.4, 0.1, 0],
        [1, 0],
    ],
)
def test_rejects_invalid_time_grid(grid):
    model, batch, aa = fixture()
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2)
    with pytest.raises(ValueError, match="grid"):
        run(model, batch, aa, generate, time_grid=torch.tensor(grid))


@pytest.mark.parametrize(
    "bad",
    [
        "partial_group",
        "clamped_group",
        "boundary",
        "canonical_mask",
        "metadata",
        "seeds",
        "nonfinite",
    ],
)
def test_rejects_invalid_sampling_contract(bad):
    model, batch, aa = fixture("AC")
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2).clone()
    ids = groups(batch, generate)
    kwargs = {}
    if bad == "partial_group":
        ids[0, 1, 0] = -1
    elif bad == "clamped_group":
        generate[0, 1, 0] = False
    elif bad == "boundary":
        generate[0, 0] = True
    elif bad == "canonical_mask":
        aa[0] = batch["sequence_mask_id"]
    elif bad == "metadata":
        batch["codebook_size"] = 3
    elif bad == "seeds":
        kwargs["seeds"] = []
    elif bad == "nonfinite":
        model.aa = aa
        original = model.forward

        def forward(*args, **kwargs):
            output = original(*args, **kwargs)
            output["structure_logits"] = torch.full_like(
                output["structure_logits"], float("nan")
            )
            return output

        model.forward = forward
    with pytest.raises((ValueError, FloatingPointError)):
        run(model, batch, aa, generate, ids=ids, **kwargs)


@pytest.mark.parametrize(
    "weights",
    [
        None,
        {"structure_only": 1},
        {"sequence_only": 1},
        {"joint_tied": 0, "structure_only": 1},
    ],
)
def test_pure_conditional_checkpoint_requires_explicit_joint_qualification(weights):
    model, batch, aa = fixture()
    if weights is None:
        del model.mdlm_regime_weights
    else:
        model.mdlm_regime_weights = weights
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2).clone()
    with pytest.raises(ValueError, match="qualified"):
        run(model, batch, aa, generate)
    generate[..., 0] = False
    run(model, batch, aa, generate)


def test_real_mdlm_forward_generates_and_keeps_frozen_codebook():
    _, batch, aa = fixture("ACD")
    model = STokMDLM(
        vocab_size=len(Tokenizer()),
        pad_id=Tokenizer().pad_token_id,
        codebook=torch.randn(2, 4),
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn_mult=2,
        dropout=0.1,
        attn_dropout=0.1,
    )
    generate = torch.zeros((*batch["residue_mask"].shape, 2), dtype=torch.bool)
    generate[..., 1] = batch["residue_mask"]
    before = model.structure_codebook.clone()
    result = run(model, batch, aa, generate)
    assert (result["structure_tokens"][generate[..., 1]] < 2).all()
    assert torch.equal(before, model.structure_codebook)


@pytest.mark.parametrize(
    "placement,tied",
    [("token", False), ("token", True), ("span", False), ("span", True)],
)
def test_every_supplied_group_reveals_together_throughout_trajectory(placement, tied):
    model, batch, aa = fixture("ACDEFGHIKLMNPQ", count=12)
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2).clone()
    ids = groups(batch, generate, placement, tied)
    original_ids = ids.clone()
    result = run(model, batch, aa, generate, ids=ids, steps=8)
    for sequence, structure in model.calls:
        masked = torch.stack(
            (sequence.eq(batch["sequence_mask_id"]), structure.eq(3)), -1
        )
        for i in range(len(generate)):
            for group in ids[i][generate[i]].unique():
                members = masked[i][ids[i] == group]
                assert members.all() or not members.any()
    assert torch.equal(ids, original_ids)
    assert (
        not result["sequence_tokens"][generate[..., 0]]
        .eq(batch["sequence_mask_id"])
        .any()
    )
    assert not result["structure_tokens"][generate[..., 1]].eq(3).any()


@pytest.mark.parametrize(
    "name,power", [("linear", 2), ("cosine", 2), ("power", 0.5), ("power", 3)]
)
def test_reveal_frequency_uses_selected_schedule(name, power):
    from stok.utils.sampling import sample_mdlm

    model, batch, aa = fixture(count=2500)
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2).clone()
    sample_mdlm(
        model,
        batch,
        generate_mask=generate,
        group_ids=groups(batch, generate),
        schedule=OmegaConf.create({"name": name, "power": power}),
        steps=2,
        seeds=list(range(2500)),
        canonical_aa_ids=aa,
    )
    # Analytic masking probabilities at t=.5, independent of schedule helper.
    p_half = 0.5**power if name == "power" else 0.5
    frequency = model.calls[1][0][:, 1].ne(batch["sequence_mask_id"]).float().mean()
    assert abs(float(frequency) - (1 - p_half)) < 0.035


def test_tokenizer_mask_and_canonical_ids_are_authoritative(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(["<mask>"] + list(reversed(DEFAULT_VOCAB[:-1]))))
    tokenizer = Tokenizer(vocab_file=str(path))
    batch = prepare_mdlm_batch(
        [
            {
                "dataset": "fixture",
                "sequence_id": "one",
                "sequence": "AC",
                "structure_tokens": [0, None],
            }
        ],
        tokenizer,
        max_len=5,
        codebook_size=2,
        crop="center",
        seeds=[0],
    )
    aa = torch.tensor(tokenizer.convert_tokens_to_ids(list(CANONICAL_AA)))
    model = FixedDenoiser(tokenizer)
    model.aa = aa
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2).clone()
    result = run(model, batch, aa, generate)
    assert batch["sequence_mask_id"] == 0
    assert (model.calls[0][0][generate[..., 0]] == 0).all()
    assert torch.isin(result["sequence_tokens"][generate[..., 0]], aa).all()


@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize("boundary", ["bos_token_id", "eos_token_id"])
def test_rejects_observable_boundary_ids_in_clean_output_support(
    tmp_path, reordered, boundary
):
    if reordered:
        path = tmp_path / "vocab.txt"
        path.write_text("\n".join(["<mask>"] + list(reversed(DEFAULT_VOCAB[:-1]))))
        tokenizer = Tokenizer(vocab_file=str(path))
    else:
        tokenizer = Tokenizer()
    batch = prepare_mdlm_batch(
        [
            {
                "dataset": "fixture",
                "sequence_id": "one",
                "sequence": "AC",
                "structure_tokens": [0, 1],
            }
        ],
        tokenizer,
        max_len=5,
        codebook_size=2,
        crop="center",
        seeds=[0],
    )
    aa = torch.tensor(tokenizer.convert_tokens_to_ids(list(CANONICAL_AA)))
    aa[0] = getattr(tokenizer, boundary)
    model = FixedDenoiser(tokenizer)
    model.aa = aa
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2).clone()
    with pytest.raises(ValueError, match="canonical"):
        run(model, batch, aa, generate)
    assert not model.calls


@pytest.mark.parametrize("weights", [None, [], "joint_tied"])
def test_malformed_regime_provenance_reports_qualification_error(weights):
    model, batch, aa = fixture()
    model.mdlm_regime_weights = weights
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2).clone()
    with pytest.raises(ValueError, match="qualified.*metadata"):
        run(model, batch, aa, generate)
    assert not model.calls


def test_small_clean_support_remains_usable_for_finite_state_sampling():
    model, batch, aa = fixture()
    aa = aa[:2]
    generate = batch["residue_mask"][..., None].expand(-1, -1, 2).clone()
    result = run(model, batch, aa, generate)
    assert torch.isin(result["sequence_tokens"][generate[..., 0]], aa).all()
