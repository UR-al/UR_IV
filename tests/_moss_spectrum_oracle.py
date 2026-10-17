"""The MOSS Spectrum forecast written out for the tests (no pack code, torch passed in).

``explicit_forecast`` is the estimator of sorryhyun/ComfyUI-Spectrum-KSampler's
``_vendor/networks/spectrum_forecast.py`` at ff25e70 spelled out in float64 — ``ChebyshevForecaster`` (tau = 2 t / total
- 1, the Chebyshev columns T0..Tm by recurrence, ridge (X'X + lam I) coef = X'H) blended with ``SpectrumPredictor``'s
Newton forward difference H[-1] + k (H[-1] - H[-2]), k = (t* - t[-1]) / (t[-1] - t[-2]) — so the pack's linear form
(one weight per observation, ``moss_spectrum.Forecaster``) is checked against the fit it stands for.
"""
from __future__ import annotations


def explicit_forecast(torch, ts, hs, t_star, m, lam, w, total):
    t = torch.tensor([float(value) for value in ts], dtype=torch.float64)

    def design(values):
        tau = (2.0 * values / total - 1.0).reshape(-1, 1)
        cols = [torch.ones_like(tau), tau]
        for _ in range(2, m + 1):
            cols.append(2 * tau * cols[-1] - cols[-2])
        return torch.cat(cols[: m + 1], 1)

    x = design(t)
    h = torch.stack([item.double().flatten() for item in hs])
    coef = torch.linalg.solve(x.T @ x + lam * torch.eye(m + 1, dtype=torch.float64), x.T @ h)
    cheb = (design(torch.tensor([float(t_star)], dtype=torch.float64)) @ coef)[0]
    if w >= 1.0 or len(ts) < 2:
        return cheb.reshape(hs[0].shape)
    k = (float(t_star) - float(ts[-1])) / max(float(ts[-1]) - float(ts[-2]), 1e-8)   # SpectrumPredictor's clamp
    taylor = h[-1] + k * (h[-1] - h[-2])
    return ((1 - w) * taylor + w * cheb).reshape(hs[0].shape)
