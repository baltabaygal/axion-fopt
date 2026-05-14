# percolation.py
import math
import numpy as np

# Kernel defaults (match your current choices)
Icrit_default     = 1.0
NTAU_PERC_default = 4000
TMAX_FAC_default  = 200.0
LOGG_CLIP_default = 250.0

def cumtrapz_same(y, x):
    dx = np.diff(x)
    out = np.zeros_like(y, dtype=np.float64)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx, dtype=np.float64)
    return out

def t_perc_RD(
    H_star, beta_over_H, vw,
    Icrit=Icrit_default,
    NTAU_PERC=NTAU_PERC_default,
    TMAX_FAC=TMAX_FAC_default,
    LOGG_CLIP=LOGG_CLIP_default,
):
    """
    Percolation time in *cosmic time* units of M_phi^{-1}.

    Inputs:
      H_star      = H_*/M_phi (dimensionless)
      beta_over_H = beta/H_*
      vw          = wall velocity

    Internally uses conformal time grid (called eta), then converts to cosmic t:
      a(eta) = H * eta
      t(eta) = 0.5 * H * eta^2

    Returns:
      t_p (cosmic time, M_phi^{-1}) as float, or np.inf if not reached.
    """
    H  = float(H_star)
    bH = float(beta_over_H)
    vw = float(vw)
    beta = bH * H

    Gamma0 = H**4
    t_PT   = 1.0 / (2.0 * H)        # t_* in their normalization
    tmax   = TMAX_FAC / H

    # conformal time max from t = 0.5*H*eta^2
    eta_max = math.sqrt(2.0 * tmax / H)

    eta = np.linspace(1e-12 / max(H, 1e-300), eta_max, NTAU_PERC, dtype=np.float64)
    a   = H * eta
    t   = 0.5 * H * eta**2

    # log Gamma = log(H^4) + beta*(t - t_PT)
    logG  = math.log(max(Gamma0, 1e-300)) + beta * (t - t_PT)
    logG  = np.minimum(logG, LOGG_CLIP)
    Gamma = np.exp(logG).astype(np.float64)

    # w = Gamma * a^4
    w  = Gamma * (a**4)
    M0 = cumtrapz_same(w, eta)
    M1 = cumtrapz_same(w * eta, eta)
    M2 = cumtrapz_same(w * eta**2, eta)
    M3 = cumtrapz_same(w * eta**3, eta)

    # K(eta) = ∫ dη_n w(η_n) (η-η_n)^3  via moment identity
    K = (eta**3) * M0 - 3.0 * (eta**2) * M1 + 3.0 * eta * M2 - M3
    K = np.maximum(K, 0.0)

    I = (4.0 * np.pi / 3.0) * (vw**3) * K
    I = np.maximum.accumulate(I)

    if (not np.isfinite(I[-1])) or (I[-1] < Icrit):
        return np.inf

    i = int(np.searchsorted(I, Icrit))
    i = max(0, min(i, len(t) - 1))
    t_p = float(t[i])   # already cosmic time in M_phi^{-1}
    return t_p


class PercolationCache:
    """
    Tiny cache for t_p(H_star, beta/H, vw).
    """
    def __init__(self):
        self._cache = {}

    def get(self, H_star, beta_over_H, vw, **kwargs):
        key = (float(H_star), float(beta_over_H), float(vw),
               kwargs.get("Icrit", Icrit_default),
               kwargs.get("NTAU_PERC", NTAU_PERC_default),
               kwargs.get("TMAX_FAC", TMAX_FAC_default),
               kwargs.get("LOGG_CLIP", LOGG_CLIP_default))
        if key in self._cache:
            return self._cache[key]
        tp = t_perc_RD(H_star, beta_over_H, vw, **kwargs)
        self._cache[key] = tp
        return tp
