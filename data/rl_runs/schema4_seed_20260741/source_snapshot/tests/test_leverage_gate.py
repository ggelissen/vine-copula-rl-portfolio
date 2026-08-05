#!/usr/bin/env python3
"""Fast invariants for the schema-4 rank-partition leverage projection."""

import math

import torch


ACTION_DIM = 7
GROSS_LEVERAGE = 1.5
NET_EXPOSURE = 1.0
MAX_LONG_WEIGHT = 0.60
MAX_SHORT_WEIGHT = 0.20
DIRECTION_LOGIT_BOUND = 5.0
LEVERAGE_SOFT_TARGET = 0.80


def capped_simplex(logits: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    lower = torch.full_like(shifted[..., :1], -50.0)
    upper = torch.full_like(shifted[..., :1], 50.0)
    for _ in range(30):
        midpoint = 0.5 * (lower + upper)
        candidate = torch.minimum(torch.exp(shifted - midpoint), cap)
        too_large = candidate.sum(dim=-1, keepdim=True) > 1.0
        lower = torch.where(too_large, midpoint, lower)
        upper = torch.where(too_large, upper, midpoint)
    probabilities = torch.minimum(
        torch.exp(shifted - 0.5 * (lower + upper)), cap
    )
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def portfolio_books(raw_action: torch.Tensor):
    direction = DIRECTION_LOGIT_BOUND * torch.tanh(
        raw_action[..., :ACTION_DIM] / DIRECTION_LOGIT_BOUND
    )
    gate = torch.sigmoid(raw_action[..., -1:])
    full_short_budget = 0.5 * (GROSS_LEVERAGE - NET_EXPOSURE)
    short_support_size = int(
        math.ceil(full_short_budget / MAX_SHORT_WEIGHT - 1e-12)
    )
    short_indices = torch.topk(
        direction, k=short_support_size, dim=-1, largest=False
    ).indices
    short_mask = torch.zeros_like(direction, dtype=torch.bool)
    short_mask.scatter_(-1, short_indices, True)
    long_logits = torch.where(
        ~short_mask, direction, torch.full_like(direction, -1e9)
    )
    short_logits = torch.where(
        short_mask, -direction, torch.full_like(direction, -1e9)
    )
    short_budget = full_short_budget * gate
    long_budget = NET_EXPOSURE + short_budget
    long_probs = capped_simplex(
        long_logits,
        torch.clamp(MAX_LONG_WEIGHT / long_budget.clamp_min(1e-12), max=1.0),
    )
    short_probs = capped_simplex(
        short_logits,
        torch.clamp(MAX_SHORT_WEIGHT / short_budget.clamp_min(1e-12), max=1.0),
    )
    return long_probs, short_probs, gate, long_budget, short_budget, short_mask


def portfolio_weights(raw_action: torch.Tensor) -> torch.Tensor:
    long_probs, short_probs, _, long_budget, short_budget, _ = portfolio_books(
        raw_action
    )
    return long_budget * long_probs - short_budget * short_probs


torch.manual_seed(20260741)
raw = torch.randn(10_000, ACTION_DIM + 1, requires_grad=True)
weights = portfolio_weights(raw)
gate = torch.sigmoid(raw[..., -1])
expected_gross = abs(NET_EXPOSURE) + gate * (
    GROSS_LEVERAGE - abs(NET_EXPOSURE)
)

assert torch.isfinite(weights).all()
assert torch.max(torch.abs(weights.sum(dim=-1) - NET_EXPOSURE)) < 2e-6
assert torch.max(torch.abs(weights.abs().sum(dim=-1) - expected_gross)) < 2e-6
assert torch.max(weights) <= MAX_LONG_WEIGHT + 2e-6
assert torch.min(weights) >= -MAX_SHORT_WEIGHT - 2e-6
assert torch.all((weights < 0).sum(dim=-1) == 2)

# The same direction must have exact, materially different leverage at low and
# high gate values. The lowest-ranked assets must be the short positions.
direction = torch.tensor([[5.0, 4.0, 2.0, 1.0, 0.0, -4.0, -5.0]])
low_raw = torch.cat([direction, torch.tensor([[-12.0]])], dim=-1)
high_raw = torch.cat([direction, torch.tensor([[12.0]])], dim=-1)
low = portfolio_weights(low_raw)
high = portfolio_weights(high_raw)
assert low.abs().sum() < 1.00001
assert high.abs().sum() > 1.49999
assert torch.equal(torch.where(high.squeeze(0) < 0)[0], torch.tensor([5, 6]))

# Smooth capped-softmax starts diversified rather than needlessly binding a
# single-asset limit when all cross-sectional scores are equal.
neutral_scores = torch.zeros(1, ACTION_DIM)
balanced = portfolio_weights(torch.cat([neutral_scores, torch.tensor([[12.0]])], -1))
assert balanced.max() < 0.26
assert balanced.min() > -0.13

loss = weights.square().mean()
loss.backward()
assert torch.isfinite(raw.grad).all()
assert raw.grad[..., :ACTION_DIM].abs().max() > 0
assert raw.grad[..., -1].abs().max() > 0

# The soft saturation constraint must produce a downward gate-logit gradient
# above its target and remain inactive below it.
high_gate_logit = torch.tensor([[4.0]], requires_grad=True)
high_penalty = torch.relu(
    torch.sigmoid(high_gate_logit) - LEVERAGE_SOFT_TARGET
).square().mean()
high_penalty.backward()
assert high_gate_logit.grad.item() > 0
low_gate_logit = torch.tensor([[0.0]], requires_grad=True)
low_penalty = torch.relu(
    torch.sigmoid(low_gate_logit) - LEVERAGE_SOFT_TARGET
).square().mean()
low_penalty.backward()
assert low_gate_logit.grad.item() == 0

print("Schema-4 leverage-gate invariants passed.")
