import math
from typing import Tuple
import torch


def _rfft_full_mean_weights_1d(n_last: int, device, dtype):
    """
    1D weights to reconstruct the full-spectrum mean from a half-spectrum rFFT:
      - n_last even: [1, 2, 2, ..., 2, 1] (DC = 1, Nyquist = 1)
      - n_last odd:  [1, 2, 2, ..., 2]    (DC = 1, no Nyquist bin)
    """
    Lh = n_last // 2 + 1
    w = torch.ones(Lh, device=device, dtype=dtype)
    if n_last % 2 == 0:
        if Lh > 2:
            w[1:-1] *= 2.0
    else:
        if Lh > 1:
            w[1:] *= 2.0
    return w


def apply_sea_from_ab(
    x: torch.Tensor,
    a: float,  # a_t
    b: float,  # b_t
    power_exp: float = 2.0,
    power_const: float = 1.0,
    dims=None,  # e.g., images: (-2, -3); videos THW: (-4, -3, -2)
    eps: float = 1e-16,
    norm_mode: str = "mean",  # "peak" | "mean"
    *,
    real: bool = False,
) -> torch.Tensor:
    """
    Apply an N-D separable Wiener filter using (a_t, b_t).

    - If multiple dims are given, a 1D Wiener gain is built per axis from |f|,
      and their product forms a separable N-D filter.
    - Normalization ("peak"/"mean") is applied to the full N-D filter
    """
    orig_dtype = x.dtype
    x = x.contiguous()
    x32 = x.to(torch.float32)

    # --- Auto-select dims if not provided ---
    if dims is None:
        if x32.ndim <= 2:
            dims = tuple(range(x32.ndim))
        else:
            dims = tuple(range(-2, -x32.ndim, -1))

    # --- FFT ---
    if real:
        X = torch.fft.rfftn(x32, dim=dims)
    else:
        X = torch.fft.fftn(x32, dim=dims)

    # --- Choose filter H ---
    H = None
    for i, ax in enumerate(dims):
        N = x32.shape[ax]
        if real and (i == len(dims) - 1):
            f = torch.fft.rfftfreq(N, device=x32.device, dtype=torch.float32)
        else:
            f = torch.fft.fftfreq(N, device=x32.device, dtype=torch.float32)

        rad = torch.abs(f)
        Sx0 = power_const / ((rad ** power_exp) + eps)
        H1 = (a * Sx0) / (a * a * Sx0 + (b * b) + eps)

        shape_i = [1] * x32.ndim
        shape_i[ax] = H1.shape[0]
        H1 = H1.reshape(shape_i)  # [1, ..., Ni', ..., 1]
        H = H1 if H is None else (H * H1)

    # --- Normalize  ---
    nm = norm_mode.lower()
    if nm == "peak":
        maxv = torch.amax(H)
        if torch.isfinite(maxv) and maxv > 0:
            H = H / maxv
    elif nm == "mean":
        if real:
            # Use weighted mean along the last FFT axis to match full-spectrum mean.
            N_last = int(x32.shape[dims[-1]])
            w_last = _rfft_full_mean_weights_1d(
                N_last, device=x32.device, dtype=torch.float32
            )  # [N//2+1]
            wshape = [1] * x32.ndim
            wshape[dims[-1]] = w_last.numel()
            W = w_last.view(*wshape)

            denom = torch.sum(W) * float(torch.prod(torch.tensor([x32.shape[d] for d in dims[:-1]])))
            meanv = torch.sum(H * W) / denom
        else:
            meanv = torch.mean(H)
        if torch.isfinite(meanv) and meanv > 0:
            H = H / meanv

    # --- Apply SEA filter and iFFT ---
    Y = X * H
    if real:
        s = [x32.shape[d] for d in dims]
        y = torch.fft.irfftn(Y, s=s, dim=dims)
    else:
        y = torch.fft.ifftn(Y, dim=dims).real

    return y.to(orig_dtype)


def ab_from_scheduler(
    scheduler, idx: int, mode: str = "flow"
) -> Tuple[float, float]:
    """
    Get (a_t, b_t) from a scheduler index.

    Args:
        scheduler: Diffusers scheduler instance.
        idx:       Discrete timestep index.
        mode:      "flow" (Rectified Flow-style) or
                   "vp" (VP/DDPM-style).

    Returns:
        (a, b): scalars defining the linear mixing coefficients.
    """

    def _clamp01(x):
        return max(1e-6, min(1.0 - 1e-6, float(x)))

    if isinstance(idx, torch.Tensor):
        idx = int(idx.detach().cpu().item())

    if mode == "flow":
        sigma = (
            float(scheduler.sigmas[idx])
            if hasattr(scheduler, "sigmas")
            else 1.0 - (idx + 1) / float(getattr(scheduler, "num_inference_steps", idx + 1))
        )
        sigma = _clamp01(sigma)
        a = 1.0 - sigma
        b = sigma
        return a, b

    ac = getattr(scheduler, "alphas_cumprod", None)
    if isinstance(ac, torch.Tensor) and ac.numel() > 0:
        a2 = _clamp01(ac[max(0, min(idx, ac.numel() - 1))])
        a = math.sqrt(a2)
        b = math.sqrt(max(1e-16, 1.0 - a2))
        return a, b

    # If only sigmas exist, approximate ᾱ = 1 / (1 + σ^2).
    if hasattr(scheduler, "sigmas"):
        sigma = float(scheduler.sigmas[idx])
        a2 = 1.0 / (1.0 + sigma * sigma)
        a2 = _clamp01(a2)
        a = math.sqrt(a2)
        b = math.sqrt(max(1e-16, 1.0 - a2))
        return a, b

    n = getattr(scheduler, "num_inference_steps", None) or (idx + 1)
    a2 = (idx + 1) / float(n)
    a2 = _clamp01(a2)
    a = math.sqrt(a2)
    b = math.sqrt(max(1e-12, 1.0 - a2))
    return a, b


def apply_sea_with_scheduler(
    x: torch.Tensor,
    scheduler,
    idx: int,
    power_exp: float = 2.0,
    dims=None,
    mode: str = "flow",
    norm_mode: str = "mean",
    *,
    real: bool = False,
) -> torch.Tensor:
    """
    Convenience wrapper: compute (a_t, b_t) from the scheduler and
    apply the corresponding Wiener filter.
    """
    a, b = ab_from_scheduler(scheduler, idx, mode=mode)
    return apply_sea_from_ab(
        x,
        a,
        b,
        power_exp=power_exp,
        dims=dims,
        norm_mode=norm_mode,
        real=real,
    )


def rel_l1(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-16) -> float:
    num = (a - b).abs().mean()
    den = b.abs().mean() + eps
    return float((num / den).detach().cpu())
