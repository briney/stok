"""Absorbing composite groups use factorized clean member predictions."""

import itertools
import math

from omegaconf import OmegaConf
import pytest
import torch

from stok.data.mdlm import prepare_mdlm_batch
from stok.utils.tokenizer import DEFAULT_VOCAB, Tokenizer

REGIMES = ["joint_independent", "structure_only", "sequence_only", "joint_tied"]


def config(**changes):
    result = OmegaConf.create(
        {
            "regime_weights": {
                name: int(name == "joint_independent") for name in REGIMES
            },
            "placement": "token",
            "span_mean": 8.0,
            "noise": {"name": "linear", "power": 2.0, "min_mask_probability": 1e-4},
        }
    )
    return OmegaConf.merge(result, changes)


def batch(tokenizer=None):
    return prepare_mdlm_batch(
        [
            {
                "dataset": "fixture",
                "canonical_id": "a" * 64,
                "sequence_id": "one",
                "sequence": "AXCDEFGH",
                "structure_tokens": [0, 1, None, 3, 4, 5, 6, 7],
            }
        ],
        tokenizer or Tokenizer(),
        max_len=12,
        codebook_size=16,
        crop="center",
        seeds=[0],
    )


@pytest.mark.parametrize("name", ["linear", "cosine", "power"])
def test_schedule_endpoints_derivative_inverse(name):
    from stok.utils.mdlm import mask_schedule, time_from_mask_probability

    endpoints, _ = mask_schedule(torch.tensor([0.0, 1.0]), name=name)
    torch.testing.assert_close(endpoints, torch.tensor([0.0, 1.0]), rtol=0, atol=0)
    t = torch.linspace(0.1, 0.9, 19, dtype=torch.float64)
    p, dp = mask_schedule(t, name=name)
    plus, _ = mask_schedule(t + 1e-6, name=name)
    minus, _ = mask_schedule(t - 1e-6, name=name)
    torch.testing.assert_close(dp, (plus - minus) / 2e-6)
    torch.testing.assert_close(time_from_mask_probability(p, name=name), t)
    assert (p.diff() > 0).all()
    if name == "power":
        torch.testing.assert_close(time_from_mask_probability(p, name=name), p.sqrt())
    # The same p_min gives the same truncated integral, independent of schedule.
    p_min = torch.tensor(1e-4, dtype=torch.float64)
    t_min = time_from_mask_probability(p_min, name=name)
    grid = torch.linspace(t_min, 1.0, 100001, dtype=torch.float64)
    probability, derivative = mask_schedule(grid, name=name)
    integral = torch.trapezoid(derivative / probability, grid)
    assert abs(float(integral) + math.log(1e-4)) < 0.002


@pytest.mark.parametrize(
    "name,power",
    [
        ("bad", 2),
        ("power", 0),
        ("power", -1),
        ("linear", math.inf),
        ("cosine", math.nan),
    ],
)
def test_schedule_rejects_invalid_parameters(name, power):
    from stok.utils.mdlm import mask_schedule, time_from_mask_probability

    for fn in (mask_schedule, time_from_mask_probability):
        with pytest.raises(ValueError):
            fn(torch.tensor(0.5), name=name, power=power)


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf])
def test_schedule_rejects_invalid_inputs(value):
    from stok.utils.mdlm import mask_schedule, time_from_mask_probability

    for fn in (mask_schedule, time_from_mask_probability):
        with pytest.raises(ValueError):
            fn(torch.tensor(value), name="linear")


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("placement", ["token", "span"])
def test_regime_placement_preserves_conditions_and_targets(regime, placement):
    from stok.utils.mdlm import corrupt_mdlm_batch

    clean = batch()
    out = corrupt_mdlm_batch(
        clean,
        config(),
        seeds=[21],
        mask_probability=0.5,
        regime=regime,
        placement=placement,
    )
    assert out["eligible"].shape == (1, 12, 2)
    for track, key in enumerate(["sequence_tokens", "structure_tokens"]):
        active = not (
            track == 0
            and regime == "structure_only"
            or track == 1
            and regime == "sequence_only"
        )
        expected = (
            clean[["sequence_valid", "structure_valid"][track]] & clean["residue_mask"]
        )
        if not active:
            expected = torch.zeros_like(expected)
        assert torch.equal(out["eligible"][..., track], expected)
        masked = out["masked"][..., track]
        assert not (masked & ~expected).any()
        assert torch.equal(out[key][~masked], clean[key][~masked])
        mask_id = (
            clean["sequence_mask_id"] if track == 0 else clean["codebook_size"] + 1
        )
        assert (out[key][masked] == mask_id).all()
        assert (out["group_ids"][..., track][~expected] == -1).all()
        assert (out["group_ids"][..., track][expected] >= 0).all()
        # Clean inputs must never be mutated.
        assert torch.equal(clean[key], batch()[key])
    if regime == "joint_tied":
        paired = out["eligible"].all(-1)
        assert torch.equal(
            out["group_ids"][..., 0][paired], out["group_ids"][..., 1][paired]
        )
        assert torch.equal(out["masked"][..., 0][paired], out["masked"][..., 1][paired])
    for group in out["group_ids"].unique():
        if group >= 0:
            statuses = out["masked"][out["group_ids"] == group]
            assert (statuses == statuses[0]).all()
    assert out["weight"].tolist() == [1.0]
    assert out["regimes"] == [regime]
    again = corrupt_mdlm_batch(
        clean,
        config(),
        seeds=[21],
        mask_probability=0.5,
        regime=regime,
        placement=placement,
    )
    for key in ["sequence_tokens", "structure_tokens", "masked", "group_ids"]:
        assert torch.equal(out[key], again[key])


def test_independent_tracks_and_occurrences_can_differ_without_global_rng_changes():
    from stok.utils.mdlm import corrupt_mdlm_batch, stable_seed

    clean = batch()
    state = torch.random.get_rng_state().clone()
    results = [
        corrupt_mdlm_batch(
            clean,
            config(),
            seeds=[stable_seed(["occurrence", i])],
            mask_probability=0.5,
            regime="joint_independent",
        )
        for i in range(12)
    ]
    assert torch.equal(torch.random.get_rng_state(), state)
    paired = clean["sequence_valid"] & clean["structure_valid"]
    assert any(
        not torch.equal(r["masked"][..., 0][paired], r["masked"][..., 1][paired])
        for r in results
    )
    assert any(not torch.equal(results[0]["masked"], r["masked"]) for r in results[1:])
    assert stable_seed([{"a": 1, "b": 2}, "crop"]) == stable_seed(
        [{"b": 2, "a": 1}, "crop"]
    )
    assert (
        len(
            {
                stable_seed([8, purpose])
                for purpose in [
                    "crop",
                    "regime",
                    "time",
                    "partition",
                    "corruption",
                    "sampling",
                ]
            }
        )
        == 6
    )
    with pytest.raises(ValueError):
        stable_seed([math.nan])


def test_spans_partition_physical_slots_and_reset_at_biological_gaps():
    from stok.utils.mdlm import build_mask_groups

    residues = torch.tensor([False, True, True, True, True, False, True, True, False])
    eligible = residues[:, None].expand(-1, 2).clone()
    eligible[2:4, 1] = False
    groups = build_mask_groups(
        eligible,
        residues,
        placement="span",
        tied=True,
        span_mean=1e30,
        generator=torch.Generator().manual_seed(2),
    )
    assert groups[1, 0] == groups[4, 0] == groups[4, 1]
    assert groups[1, 0] != groups[6, 0]
    assert (groups[2:4, 1] == -1).all()
    # Ineligible physical residues still contribute span-boundary RNG draws.
    full = build_mask_groups(
        residues[:, None].expand(-1, 2),
        residues,
        placement="span",
        tied=True,
        span_mean=2,
        generator=torch.Generator().manual_seed(17),
    )
    sparse = build_mask_groups(
        eligible,
        residues,
        placement="span",
        tied=True,
        span_mean=2,
        generator=torch.Generator().manual_seed(17),
    )
    assert torch.equal(full[eligible], sparse[eligible])


@pytest.mark.parametrize(
    "placement,span_mean",
    [("bad", 8), ("span", 0.99), ("token", math.inf), ("span", math.nan)],
)
def test_groups_reject_invalid_parameters(placement, span_mean):
    from stok.utils.mdlm import build_mask_groups

    with pytest.raises(ValueError):
        build_mask_groups(
            torch.ones(5, 2, dtype=torch.bool),
            torch.ones(5, dtype=torch.bool),
            placement=placement,
            tied=False,
            span_mean=span_mean,
            generator=torch.Generator().manual_seed(0),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"regime_weights": {"joint_independent": -1}},
        {"regime_weights": {name: 0 for name in REGIMES}},
        {"regime_weights": {"bad": 1}},
        {"regime_weights": {"joint_tied": math.nan}},
        {"placement": "bad"},
        {"span_mean": 0.5},
        {"span_mean": math.inf},
        {"noise": {"name": "bad"}},
        {"noise": {"power": 0}},
        {"noise": {"min_mask_probability": 0}},
        {"noise": {"min_mask_probability": 1}},
        {"noise": {"min_mask_probability": math.nan}},
    ],
)
def test_corruption_rejects_invalid_training_configuration(changes):
    from stok.utils.mdlm import corrupt_mdlm_batch

    with pytest.raises(ValueError):
        corrupt_mdlm_batch(batch(), config(**changes), seeds=[0])


@pytest.mark.parametrize(
    "overrides",
    [
        {"regime": "joint_tied"},
        {"mask_probability": 0.5},
        {"placement": "span"},
        {"regime": "bad", "mask_probability": 0.5},
        *[
            {"regime": "joint_tied", "mask_probability": value}
            for value in [-0.1, 1.1, math.nan, math.inf]
        ],
    ],
)
def test_corruption_rejects_partial_or_invalid_diagnostics(overrides):
    from stok.utils.mdlm import corrupt_mdlm_batch

    with pytest.raises(ValueError):
        corrupt_mdlm_batch(batch(), config(), seeds=[0], **overrides)


@pytest.mark.parametrize("name", ["linear", "cosine", "power"])
def test_training_weight_and_probability_use_shared_schedule(name):
    from stok.utils.mdlm import (
        corrupt_mdlm_batch,
        time_from_mask_probability,
        mask_schedule,
    )

    cfg = config(noise={"name": name})
    clean = batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = corrupt_mdlm_batch(clean, cfg, seeds=[13])
    p = out["mask_probability"]
    assert p.dtype == out["weight"].dtype == torch.float64
    assert ((p >= 1e-4) & (p < 1)).all()
    t = time_from_mask_probability(p, name=name)
    _, derivative = mask_schedule(t, name=name)
    t_min = time_from_mask_probability(
        torch.tensor(1e-4, dtype=torch.float64), name=name
    )
    torch.testing.assert_close(out["weight"], (1 - t_min) * derivative / p)
    assert out["regimes"] == ["joint_independent"]
    cfg.regime_weights = {name: int(name == "sequence_only") for name in REGIMES}
    assert corrupt_mdlm_batch(clean, cfg, seeds=[13])["regimes"] == ["sequence_only"]


def test_reordered_tokenizer_mask_id_and_codebook_metadata(tmp_path):
    from stok.utils.mdlm import corrupt_mdlm_batch

    vocab = ["<mask>"] + list(reversed(DEFAULT_VOCAB[:-1]))
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    tok = Tokenizer(vocab_file=str(path))
    clean = batch(tok)
    assert clean["sequence_mask_id"] == tok.mask_token_id == 0
    assert clean["codebook_size"] == 16
    out = corrupt_mdlm_batch(
        clean, config(), seeds=[1], regime="joint_tied", mask_probability=1.0
    )
    assert (out["sequence_tokens"][clean["sequence_valid"]] == 0).all()
    assert (out["structure_tokens"][clean["structure_valid"]] == 17).all()
    assert out["sequence_tokens"][0, 2] == tok.convert_tokens_to_ids("X")


def test_prepare_rejects_missing_or_invalid_mask_token_metadata():
    tok = Tokenizer()
    tok.mask_token = None
    with pytest.raises(ValueError, match="mask"):
        batch(tok)


def test_group_marginals_and_zero_mask_draws():
    from stok.utils.mdlm import corrupt_mdlm_batch

    clean = batch()
    counts = torch.zeros(12, 2)
    zero = 0
    unequal = False
    for seed in range(1200):
        out = corrupt_mdlm_batch(
            clean,
            config(span_mean=3),
            seeds=[seed],
            placement="span",
            regime="joint_tied",
            mask_probability=0.2,
        )
        counts += out["masked"][0]
        zero += int(not out["masked"].any())
        ids, lengths = out["group_ids"][0][..., 0].unique(return_counts=True)
        lengths = lengths[ids >= 0]
        unequal |= len(lengths.unique()) > 1
    frequency = counts / 1200
    eligible = out["eligible"][0]
    assert unequal and zero > 0
    assert ((frequency[eligible] - 0.2).abs() < 0.05).all()
    assert abs(float(frequency[8, 0]) - 0.2) < 0.05  # short terminal spans
    for p in (0.0, 1.0):
        out = corrupt_mdlm_batch(
            clean, config(), seeds=[0], regime="joint_tied", mask_probability=p
        )
        assert torch.equal(
            out["masked"], out["eligible"] if p else torch.zeros_like(out["eligible"])
        )


def test_two_member_absorbing_forward_reverse_and_factorized_kl():
    from stok.utils.mdlm import mask_schedule

    # Four clean tuples and one absorbing state; no partly masked states.
    states = list(itertools.product(range(2), repeat=2)) + [None]
    p_s = float(
        mask_schedule(torch.tensor(0.25, dtype=torch.float64), name="linear")[0]
    )
    p_t = float(
        mask_schedule(torch.tensor(0.75, dtype=torch.float64), name="linear")[0]
    )
    forward = torch.eye(5, dtype=torch.float64)
    for i in range(4):
        forward[i, i] = 1 - p_t
        forward[i, 4] = p_t
    torch.testing.assert_close(forward.sum(1), torch.ones(5, dtype=torch.float64))
    reveal = (p_t - p_s) / p_t
    # Factorized clean predictions for the absorbing composite variable.
    member = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float64)
    prediction = torch.tensor([member[0, a] * member[1, b] for a, b in states[:4]])
    reverse = torch.eye(5, dtype=torch.float64)
    reverse[4, :4] = reveal * prediction
    reverse[4, 4] = 1 - reveal
    torch.testing.assert_close(reverse.sum(1), torch.ones(5, dtype=torch.float64))
    known = states.index((0, 1))
    q = torch.zeros(5, dtype=torch.float64)
    q[known], q[4] = reveal, 1 - reveal
    assert math.isclose(reveal, 2 / 3)
    # Bayes on the clean->absorbing forward kernel independently gives q_reverse.
    q_s = torch.zeros(5, dtype=torch.float64)
    q_s[known], q_s[4] = 1 - p_s, p_s
    s_to_t_mask = (p_t - p_s) / (1 - p_s)
    step = torch.eye(5, dtype=torch.float64)
    for i in range(4):
        step[i, i], step[i, 4] = 1 - s_to_t_mask, s_to_t_mask
    torch.testing.assert_close(step.sum(1), torch.ones(5, dtype=torch.float64))
    torch.testing.assert_close(q_s @ step, forward[known])
    likelihood = torch.tensor([s_to_t_mask] * 4 + [1.0], dtype=torch.float64)
    torch.testing.assert_close(q, q_s * likelihood / p_t)
    positive = q > 0
    kl = (q[positive] * (q[positive].log() - reverse[4, positive].log())).sum()
    member_ce = -member[0, 0].log() - member[1, 1].log()
    torch.testing.assert_close(kl, reveal * member_ce)


def test_two_member_composite_corruption_matches_absorbing_reference():
    from stok.utils.mdlm import corrupt_mdlm_batch

    clean = prepare_mdlm_batch(
        [
            {
                "dataset": "binary",
                "canonical_id": "b" * 64,
                "sequence_id": "known",
                "sequence": "A",
                "structure_tokens": [1],
            }
        ],
        Tokenizer(),
        max_len=3,
        codebook_size=2,
        crop="center",
        seeds=[0],
    )
    absorbing = 0
    for seed in range(500):
        out = corrupt_mdlm_batch(
            clean, config(), seeds=[seed], mask_probability=0.75, regime="joint_tied"
        )
        assert out["group_ids"][0, 1, 0] == out["group_ids"][0, 1, 1]
        state = out["masked"][0, 1]
        assert bool(state.all()) or not bool(state.any())
        absorbing += int(state.all())
    assert abs(absorbing / 500 - 0.75) < 0.05


def test_integer_diagnostic_probability_keeps_floating_output_contract():
    from stok.utils.mdlm import corrupt_mdlm_batch

    out = corrupt_mdlm_batch(
        batch(), config(), seeds=[0], mask_probability=1, regime="sequence_only"
    )
    assert out["mask_probability"].is_floating_point()
    assert out["weight"].is_floating_point()


def test_training_time_stays_below_one_for_narrow_valid_noise_interval():
    from stok.utils.mdlm import corrupt_mdlm_batch

    cfg = config(noise={"name": "power", "power": 1e7})
    clean = batch()
    for seed in range(100):
        p = corrupt_mdlm_batch(clean, cfg, seeds=[seed])["mask_probability"]
        assert (p < 1).all()


def test_numerically_unrepresentable_time_interval_is_rejected():
    from stok.utils.mdlm import corrupt_mdlm_batch

    with pytest.raises(ValueError, match="numerical support"):
        corrupt_mdlm_batch(
            batch(), config(noise={"name": "power", "power": 1e20}), seeds=[0]
        )
