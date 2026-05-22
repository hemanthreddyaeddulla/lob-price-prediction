import torch


def plain_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Vanilla masked MSE on clipped predictions. This is what worked best."""
    pred_clipped = torch.clamp(pred, -6.0, 6.0)
    se = (pred_clipped - target) ** 2
    mask_3d = mask.unsqueeze(-1).expand_as(se)
    n_scored = mask_3d.sum().clamp(min=1.0)
    return (se * mask_3d).sum() / n_scored


def soft_weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE with sqrt(|target|) weighting. Softer than |target| weighting,
    which was overfitting to extremes. In practice plain MSE still won."""
    pred_clipped = torch.clamp(pred, -6.0, 6.0)
    weights = torch.sqrt(torch.abs(target) + 0.1)
    se = weights * (pred_clipped - target) ** 2
    mask_3d = mask.unsqueeze(-1).expand_as(se)
    n_scored = mask_3d.sum().clamp(min=1.0)
    return (se * mask_3d).sum() / n_scored


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Standard Huber. Tried as an outlier-robust alternative to MSE; ~neutral."""
    pred_clipped = torch.clamp(pred, -6.0, 6.0)
    diff = pred_clipped - target
    abs_diff = torch.abs(diff)
    quadratic = 0.5 * diff ** 2
    linear = delta * (abs_diff - 0.5 * delta)
    loss = torch.where(abs_diff <= delta, quadratic, linear)
    mask_3d = mask.unsqueeze(-1).expand_as(loss)
    n_scored = mask_3d.sum().clamp(min=1.0)
    return (loss * mask_3d).sum() / n_scored


def weighted_pearson_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Differentiable version of the competition metric (negated, so minimizing it
    maximizes correlation). Computed per-target then averaged."""
    pred_clipped = torch.clamp(pred, -6.0, 6.0)
    mask_bool = mask.bool()

    total_corr = torch.tensor(0.0, device=pred.device)
    n_targets = 0

    for t_idx in range(2):
        y = target[:, :, t_idx]
        p = pred_clipped[:, :, t_idx]

        y_flat = y[mask_bool]
        p_flat = p[mask_bool]

        if len(y_flat) < 2:
            continue

        w = torch.abs(y_flat).clamp(min=1e-8)
        w_sum = w.sum()

        mean_y = (w * y_flat).sum() / w_sum
        mean_p = (w * p_flat).sum() / w_sum

        dev_y = y_flat - mean_y
        dev_p = p_flat - mean_p

        cov = (w * dev_y * dev_p).sum() / w_sum
        var_y = (w * dev_y ** 2).sum() / w_sum
        var_p = (w * dev_p ** 2).sum() / w_sum

        denom = torch.sqrt(var_y * var_p).clamp(min=1e-8)
        corr = cov / denom

        total_corr = total_corr + corr
        n_targets += 1

    if n_targets == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return -total_corr / n_targets


def per_target_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_weights: tuple = (1.0, 1.0),
    mse_mode: str = "plain",
) -> torch.Tensor:
    """MSE-style loss with a separate weight per target. Lets you experiment with
    upweighting t1 - though in this competition that didn't actually help."""
    pred_clipped = torch.clamp(pred, -6.0, 6.0)
    mask_2d = mask.unsqueeze(-1)

    total = torch.tensor(0.0, device=pred.device)
    for t_idx in range(2):
        p = pred_clipped[:, :, t_idx:t_idx+1]
        t = target[:, :, t_idx:t_idx+1]
        if mse_mode == "plain":
            se = (p - t) ** 2
        elif mse_mode == "soft":
            w = torch.sqrt(torch.abs(t) + 0.1)
            se = w * (p - t) ** 2
        elif mse_mode == "huber":
            diff = p - t
            abs_diff = torch.abs(diff)
            se = torch.where(abs_diff <= 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
        else:
            raise ValueError(f"Unknown mse_mode: {mse_mode}")
        masked = se * mask_2d
        n = mask_2d.sum().clamp(min=1.0)
        total = total + target_weights[t_idx] * masked.sum() / n

    return total / sum(target_weights)


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mse_weight: float = 1.0,
    pearson_weight: float = 0.0,
    mse_mode: str = "plain",
    target_weights: tuple = (1.0, 1.0),
) -> torch.Tensor:
    """Convex combination of MSE and weighted-Pearson. The training loop ramps
    pearson_weight from 0 to ~0.5 over the last few epochs."""
    loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

    if mse_weight > 0:
        loss = loss + mse_weight * per_target_mse(pred, target, mask, target_weights, mse_mode)

    if pearson_weight > 0:
        loss = loss + pearson_weight * weighted_pearson_loss(pred, target, mask)

    return loss
