"""BWM configuration compatibility and one-time evidence (no model state)."""

import os

import torch


BWM_MODES = ("full", "shared_directions", "shared_channels")


def resolve_bwm_enabled(use_bwm=None, use_hbwr=None):
    """Accept the old keyword without silently resolving conflicting switches."""
    if use_bwm is not None and use_hbwr is not None:
        if bool(use_bwm) != bool(use_hbwr):
            raise ValueError("Conflicting use_bwm/use_hbwr switches; use one value.")
    value = use_bwm if use_bwm is not None else use_hbwr
    return True if value is None else bool(value)


def validate_bwm_mode(mode):
    if mode not in BWM_MODES:
        raise ValueError(f"bwm_mode must be one of {BWM_MODES}, got {mode!r}")
    return mode


def is_main_process():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # RANK is global (unlike LOCAL_RANK on multi-node jobs).
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))
    return int(rank) == 0


def emit_bwm_forward_evidence(core, s_t2r, s_r2t, m_rgb, m_e, gate_rgb, gate_e):
    """Inspect the first masks actually consumed by WRITE, once on rank zero.

    Reads detached tensors only, consumes no RNG, and changes no parameters,
    buffers, gradients or input tensors. The flag is an ordinary Python bool.
    """
    if getattr(core, "_bwm_evidence_emitted", False):
        return
    if not is_main_process():
        core._bwm_evidence_emitted = True
        return

    with torch.no_grad():
        left = s_t2r.detach().float()
        right = s_r2t.detach().float()
        mask_left, mask_right = m_rgb.detach(), m_e.detach()
        finite = all(bool(torch.isfinite(t).all().item())
                     for t in (left, right, mask_left, mask_right))
        if not finite:
            raise RuntimeError("[BWM evidence] non-finite exponents or WRITE masks")

        direction_error = (left - right).abs().max().item()
        channel_error = max((left - left[:, :1, :]).abs().max().item(),
                            (right - right[:, :1, :]).abs().max().item())
        enabled = bool(core.use_bwm)
        mode = core.bwm_mode
        if not enabled:
            exponent_error = max((left - 1).abs().max().item(),
                                 (right - 1).abs().max().item())
            mask_error = max((mask_left - gate_rgb.detach()).abs().max().item(),
                             (mask_right - gate_e.detach()).abs().max().item())
            if exponent_error != 0.0 or mask_error != 0.0:
                raise RuntimeError("[BWM evidence] baseline must use s=1 and M=G")
            branch = "without_bwm"
            check = f"s_is_one=PASS mask_equals_gate=PASS max_error={mask_error:.3e}"
        elif mode == "shared_directions":
            if direction_error != 0.0:
                raise RuntimeError("[BWM evidence] direction-sharing constraint failed")
            branch = mode
            check = f"direction_sharing=PASS max_direction_diff={direction_error:.3e}"
        elif mode == "shared_channels":
            if channel_error != 0.0:
                raise RuntimeError("[BWM evidence] channel-sharing constraint failed")
            branch = mode
            check = f"channel_sharing=PASS max_channel_spread={channel_error:.3e}"
        else:
            branch = "full"
            check = ("sharing=NONE "
                     f"max_direction_diff={direction_error:.3e} "
                     f"max_channel_spread={channel_error:.3e}")

        observed_min = min(left.min().item(), right.min().item())
        observed_max = max(left.max().item(), right.max().item())
        bounds = (core.direction_router.s_min, core.direction_router.s_max)
        # Exponents have already been cast to the feature dtype. Allow its
        # rounding at the interval endpoints (especially bfloat16/fp16).
        tolerance = 2 * torch.finfo(s_t2r.dtype).eps * max(1.0, abs(bounds[1]))
        if enabled and (observed_min < bounds[0] - tolerance
                        or observed_max > bounds[1] + tolerance):
            raise RuntimeError("[BWM evidence] exponents exceeded the stage interval")

    core._bwm_evidence_emitted = True
    print(
        f"[BWM evidence] stage={core.bwm_stage} branch={branch} "
        f"WI={bool(core.use_wi)} RI={bool(core.use_ri)} BWM={enabled} "
        f"s_t2r_shape={tuple(s_t2r.shape)} s_r2t_shape={tuple(s_r2t.shape)} "
        f"mask_shape={tuple(m_rgb.shape)} range={bounds} "
        f"observed=[{observed_min:.6f},{observed_max:.6f}] {check} "
        "source=actual_WRITE_inputs",
        flush=True,
    )
