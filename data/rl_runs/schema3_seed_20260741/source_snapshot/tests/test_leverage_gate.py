#!/usr/bin/env python3
"""Fast invariants for the TD3 actor's explicit leverage-gate projection."""

import torch


ACTION_DIM = 7
GROSS_LEVERAGE = 1.5
NET_EXPOSURE = 1.0
MAX_LONG_WEIGHT = 0.60
MAX_SHORT_WEIGHT = 0.20
DIRECTION_LOGIT_BOUND = 5.0


def capped_simplex(logits: torch.Tensor, cap: float) -> torch.Tensor:
    lower = logits.min(dim=-1, keepdim=True).values - cap
    upper = logits.max(dim=-1, keepdim=True).values
    for _ in range(40):
        midpoint = 0.5 * (lower + upper)
        candidate = torch.clamp(logits - midpoint, min=0.0, max=cap)
        too_large = candidate.sum(dim=-1, keepdim=True) > 1.0
        lower = torch.where(too_large, midpoint, lower)
        upper = torch.where(too_large, upper, midpoint)
    probabilities = torch.clamp(
        logits - 0.5 * (lower + upper), min=0.0, max=cap
    )
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def portfolio_weights(raw_action: torch.Tensor) -> torch.Tensor:
    long_logits = DIRECTION_LOGIT_BOUND * torch.tanh(
        raw_action[..., :ACTION_DIM] / DIRECTION_LOGIT_BOUND
    )
    short_logits = DIRECTION_LOGIT_BOUND * torch.tanh(
        raw_action[..., ACTION_DIM : 2 * ACTION_DIM] / DIRECTION_LOGIT_BOUND
    )
    leverage_gate = torch.sigmoid(raw_action[..., -1:])
    long_budget = 0.5 * (GROSS_LEVERAGE + NET_EXPOSURE)
    short_budget = 0.5 * (GROSS_LEVERAGE - NET_EXPOSURE)
    long_probs = capped_simplex(long_logits, MAX_LONG_WEIGHT / long_budget)
    short_probs = capped_simplex(short_logits, MAX_SHORT_WEIGHT / short_budget)
    full_risk = long_budget * long_probs - short_budget * short_probs
    neutral = torch.full_like(full_risk, NET_EXPOSURE / ACTION_DIM)
    return neutral + leverage_gate * (full_risk - neutral)


torch.manual_seed(20260741)
raw = torch.randn(10_000, 2 * ACTION_DIM + 1, requires_grad=True)
weights = portfolio_weights(raw)
assert torch.isfinite(weights).all()
assert torch.max(torch.abs(weights.sum(dim=-1) - NET_EXPOSURE)) < 1e-6
assert torch.max(weights.abs().sum(dim=-1)) <= GROSS_LEVERAGE + 1e-6
assert torch.max(weights) <= MAX_LONG_WEIGHT + 1e-6
assert torch.min(weights) >= -MAX_SHORT_WEIGHT - 1e-6

# The gate must represent materially different leverage levels for identical
# directional logits; the old action map had no explicit leverage control.
long_scores = torch.tensor([[5.0, 5.0, 2.0, -5.0, -5.0, -5.0, -5.0]])
short_scores = torch.tensor([[-5.0, -5.0, -5.0, 5.0, 5.0, 0.0, 0.0]])
direction = torch.cat([long_scores, short_scores], dim=-1)
low = portfolio_weights(torch.cat([direction, torch.tensor([[-12.0]])], dim=-1))
high = portfolio_weights(torch.cat([direction, torch.tensor([[12.0]])], dim=-1))
assert low.abs().sum() < 1.0001
assert high.abs().sum() > 1.49

loss = weights.square().mean()
loss.backward()
assert torch.isfinite(raw.grad).all()
assert raw.grad[:, -1].abs().max() > 0

print("Leverage-gate invariants passed.")
