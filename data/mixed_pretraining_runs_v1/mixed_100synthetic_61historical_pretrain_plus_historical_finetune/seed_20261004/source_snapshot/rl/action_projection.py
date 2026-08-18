"""Shared differentiable long-short action map for training and evaluation."""

import torch


def action_components(raw_action: torch.Tensor, direction_logit_bound: float):
    """Return bounded cross-sectional scores and the gross-leverage gate."""
    action_dim = raw_action.shape[-1] - 1
    raw_direction = raw_action[..., :action_dim]
    direction = direction_logit_bound * torch.tanh(
        raw_direction / direction_logit_bound
    )
    return direction, torch.sigmoid(raw_action[..., -1:])


def interior_capped_simplex(
    logits: torch.Tensor, cap: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Map scores smoothly into the interior of a capped probability simplex.

    The Lagrange multiplier is solved by eight unrolled Newton iterations so
    gradients include its dependence on every active score and on the cap.
    Masked logits must be less than -1e8.
    """
    active = logits > -1e8
    shifted = torch.where(active, logits, torch.zeros_like(logits))
    active_float = active.to(shifted.dtype)
    active_count = active_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
    lagrange = (shifted * active_float).sum(dim=-1, keepdim=True) / active_count
    for _ in range(8):
        sigmoid_values = torch.sigmoid(
            (shifted - lagrange) / temperature
        ) * active_float
        probabilities = cap * sigmoid_values
        residual = probabilities.sum(dim=-1, keepdim=True) - 1.0
        derivative = -(cap * sigmoid_values * (1.0 - sigmoid_values)).sum(
            dim=-1, keepdim=True
        ) / temperature
        newton_step = (residual / derivative.clamp(max=-1e-8)).clamp(
            -10.0 * temperature, 10.0 * temperature
        )
        lagrange = lagrange - newton_step
    probabilities = cap * torch.sigmoid(
        (shifted - lagrange) / temperature
    ) * active_float
    return probabilities / probabilities.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)


def portfolio_books(
    raw_action: torch.Tensor,
    *,
    direction_logit_bound: float,
    projection_temperature: float,
    net_exposure: float,
    full_short_budget: float,
    max_long_weight: float,
    max_short_weight: float,
    short_support_size: int,
):
    """Construct disjoint capped long/short books and the leverage budgets."""
    direction, leverage_gate = action_components(
        raw_action, direction_logit_bound
    )
    action_dim = direction.shape[-1]
    if action_dim <= short_support_size:
        raise RuntimeError(
            "Actor action dimension is incompatible with the short support."
        )
    if short_support_size > 0:
        short_indices = torch.topk(
            direction, k=short_support_size, dim=-1, largest=False
        ).indices
        short_mask = torch.zeros_like(direction, dtype=torch.bool)
        short_mask.scatter_(-1, short_indices, True)
    else:
        short_mask = torch.zeros_like(direction, dtype=torch.bool)
    long_logits = torch.where(
        ~short_mask, direction, torch.full_like(direction, -1e9)
    )
    short_logits = torch.where(
        short_mask, -direction, torch.full_like(direction, -1e9)
    )
    short_budget = full_short_budget * leverage_gate
    long_budget = net_exposure + short_budget
    long_cap = torch.clamp(
        max_long_weight / long_budget.clamp_min(1e-12), max=1.0
    )
    long_probs = interior_capped_simplex(
        long_logits, long_cap, projection_temperature
    )
    if short_support_size > 0:
        short_cap = torch.clamp(
            max_short_weight / short_budget.clamp_min(1e-12), max=1.0
        )
        short_probs = interior_capped_simplex(
            short_logits, short_cap, projection_temperature
        )
    else:
        short_probs = torch.zeros_like(long_probs)
    return long_probs, short_probs, leverage_gate, long_budget, short_budget
