"""Per-env (per-batch) loss normalization for the GRPO step.

The DAPO loss used a single global token denominator (1/Σ|o_i|), so an env
whose completions are long (code) dominated the shared optimizer step over an
env whose completions are short (math). These tests pin the fix: normalize
token-level *within* each batch, then recombine with explicit weights, so each
env's gradient contribution is independent of its token mass.
"""
import pytest

import torch
from dataclasses import dataclass, field
from types import SimpleNamespace

from reliquary.validator.training import (
    _batch_loss_weights,
    _plan_from_batches,
    _accumulate_grouped_grads,
    _compute_advantages,
)


# ---------------------------------------------------------------------------
# Fakes (self-contained — mirrors test_train_step_microbatch.py)
# ---------------------------------------------------------------------------

@dataclass
class _FakeRollout:
    tokens: list
    reward: float
    env_name: str = ""
    commit: dict = field(default_factory=dict)


@dataclass
class _FakeGroup:
    rollouts: list
    prompt_idx: int = 0


def _build_rollout(tokens, reward, prompt_length, env_name=""):
    n = len(tokens) - prompt_length
    return _FakeRollout(tokens=tokens, reward=reward, env_name=env_name, commit={
        "tokens": tokens,
        "rollout": {"prompt_length": prompt_length, "token_logprobs": [-1.0] * n},
    })


class _Base(torch.nn.Module):
    """Embedding-only base (no cross-token attention): a padded batch forward
    gives per-row hidden states identical to single-sequence forwards."""

    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


class _QwenLike(torch.nn.Module):
    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.model = _Base(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, *a, **k):  # pragma: no cover - must not be used
        raise AssertionError("full-logits forward should not be used")


def _frozen(model):
    import copy
    f = copy.deepcopy(model).eval()
    for p in f.parameters():
        p.requires_grad = False
    return f


def _grads_after(model, fn):
    for p in model.parameters():
        p.grad = None
    fn()
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


def _rel_l2(a, b):
    num = sum(((x - y) ** 2).sum() for x, y in zip(a, b))
    den = sum((x ** 2).sum() for x in a)
    return (num.sqrt() / den.sqrt()).item()


# ---------------------------------------------------------------------------
# Pure weight function
# ---------------------------------------------------------------------------

def test_each_env_contributes_its_weight_regardless_of_token_mass():
    """s_b * N_b == w_b for every env. With equal default weights and very
    different token masses, each env still contributes exactly 0.5."""
    scales = _batch_loss_weights([100.0, 10.0])
    assert scales[0] * 100.0 == pytest.approx(0.5)
    assert scales[1] * 10.0 == pytest.approx(0.5)


def test_single_batch_reduces_to_global_normalization():
    """One env → weight 1 → scale 1/N, i.e. the exact pre-fix DAPO denominator."""
    assert _batch_loss_weights([42.0]) == pytest.approx([1.0 / 42.0])


def test_empty_batch_is_dropped_and_weights_renormalize():
    """A batch with no surviving tokens gets scale 0; the present batch carries
    the full weight (1.0), never a fraction of a phantom split."""
    scales = _batch_loss_weights([100.0, 0.0])
    assert scales[1] == 0.0
    assert scales[0] * 100.0 == pytest.approx(1.0)


def test_explicit_weights_are_honored():
    """Non-equal w_e: a 3:1 weighting splits the total contribution 0.75/0.25."""
    scales = _batch_loss_weights([10.0, 10.0], raw_weights=[3.0, 1.0])
    assert scales[0] * 10.0 == pytest.approx(0.75)
    assert scales[1] * 10.0 == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Gradient-level invariance: a token-heavy env must not swamp the other
# ---------------------------------------------------------------------------

def _env_a():
    return _FakeGroup([
        _build_rollout([1, 2, 3, 4, 5, 6], 1.0, 2, "math"),
        _build_rollout([1, 2, 7, 8, 9], 0.0, 2, "math"),
        _build_rollout([1, 2, 3, 4, 5], 1.0, 2, "math"),
        _build_rollout([1, 2, 7], 0.0, 2, "math"),
    ])


def _env_b():
    return _FakeGroup([
        _build_rollout([4, 5, 6, 7, 8, 9], 1.0, 3, "code"),
        _build_rollout([4, 5, 6, 1], 0.0, 3, "code"),
        _build_rollout([4, 5, 6, 7, 8, 9, 10, 11], 1.0, 3, "code"),
        _build_rollout([4, 5, 6, 7], 0.0, 3, "code"),
    ])


def test_doubling_one_envs_token_mass_does_not_change_the_step():
    """Per-env normalization is the whole point: env B carrying twice the tokens
    (two identical groups) must not change env A's share — so the summed step is
    identical. Under the old global 1/Σ|o_i| this would shift with B's mass.
    """
    import copy
    torch.manual_seed(0)
    base = _QwenLike()
    device = next(base.parameters()).device

    plan1, _ = _plan_from_batches([[_env_a()], [_env_b()]])
    plan2, _ = _plan_from_batches([[_env_a()], [_env_b(), _env_b()]])

    m1 = copy.deepcopy(base)
    g1 = _grads_after(m1, lambda: _accumulate_grouped_grads(
        m1, _frozen(m1), plan1, device, budget=256, atomic=False))
    m2 = copy.deepcopy(base)
    g2 = _grads_after(m2, lambda: _accumulate_grouped_grads(
        m2, _frozen(m2), plan2, device, budget=256, atomic=False))

    assert _rel_l2(g1, g2) < 2e-2
