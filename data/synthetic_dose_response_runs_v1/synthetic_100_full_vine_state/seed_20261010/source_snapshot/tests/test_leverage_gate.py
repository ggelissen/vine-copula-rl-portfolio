#!/usr/bin/env python3
"""Fast invariants for the schema-5 interior leverage projection."""

import itertools
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.action_projection import portfolio_books as shared_portfolio_books


ACTION_DIM = 7
GROSS_LEVERAGE = 1.5
NET_EXPOSURE = 1.0
MAX_LONG_WEIGHT = 0.60
MAX_SHORT_WEIGHT = 0.20
DIRECTION_LOGIT_BOUND = 5.0
PROJECTION_TEMPERATURE = 1.5
LEVERAGE_SOFT_TARGET = 0.80


def portfolio_books(raw_action: torch.Tensor):
    full_short_budget = 0.5 * (GROSS_LEVERAGE - NET_EXPOSURE)
    short_support_size = int(
        math.ceil(full_short_budget / MAX_SHORT_WEIGHT - 1e-12)
    )
    books = shared_portfolio_books(
        raw_action,
        direction_logit_bound=DIRECTION_LOGIT_BOUND,
        projection_temperature=PROJECTION_TEMPERATURE,
        net_exposure=NET_EXPOSURE,
        full_short_budget=full_short_budget,
        max_long_weight=MAX_LONG_WEIGHT,
        max_short_weight=MAX_SHORT_WEIGHT,
        short_support_size=short_support_size,
    )
    return (*books, short_support_size)


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

# The smooth interior map starts diversified rather than needlessly binding a
# single-asset limit when all cross-sectional scores are equal.
neutral_scores = torch.zeros(1, ACTION_DIM)
balanced = portfolio_weights(torch.cat([neutral_scores, torch.tensor([[12.0]])], -1))
assert balanced.max() < 0.26
assert balanced.min() > -0.13

# Bounded direction endpoints are the most adversarial inputs the actor can
# produce. Exhaust every endpoint ordering and several leverage regimes; the
# smooth map must stay strictly inside both limits, not merely within tolerance.
endpoint_directions = torch.tensor(
    list(itertools.product((-20.0, 20.0), repeat=ACTION_DIM))
)
endpoint_cases = []
for gate_logit in (-12.0, -2.2, 0.0, 1.4, 12.0):
    endpoint_cases.append(
        torch.cat(
            [
                endpoint_directions,
                torch.full((len(endpoint_directions), 1), gate_logit),
            ],
            dim=-1,
        )
    )
endpoint_weights = portfolio_weights(torch.cat(endpoint_cases, dim=0))
assert endpoint_weights.max() < MAX_LONG_WEIGHT - 1e-4
assert endpoint_weights.min() > -MAX_SHORT_WEIGHT + 1e-4
assert torch.max(torch.abs(endpoint_weights.sum(dim=-1) - NET_EXPOSURE)) < 2e-6

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

print("Schema-5 interior leverage-gate invariants passed.")
