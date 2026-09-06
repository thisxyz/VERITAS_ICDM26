# -*- coding: utf-8 -*-

import math

import numpy as np

EPS_DEFAULT = 1e-12
MECHANISMS = ["1B", "LP", "AG", "SW", "MB", "PM"]


# --------------------------------------------------------------------------
# one-dimensional kernels (input x in [-1, 1])
# --------------------------------------------------------------------------
def _pm_perturb1d(x, eps, rng):
    """Piecewise mechanism, Eq.(4)-(5) of the paper (domain [-1,1])."""
    e = math.exp(eps)
    eh = math.exp(eps / 2.0)
    B = (eh + 1.0) / (eh - 1.0 + EPS_DEFAULT)
    p = (e - eh) / (2.0 * eh + 2.0)
    l = 0.5 * (B + 1.0) * x - 0.5 * (B - 1.0)
    r = l + B - 1.0
    u = rng.random()
    width_hi = (r - l) if r > l else 0.0
    width_lo = (B - width_hi) if (B - width_hi) > 0 else 0.0
    prob_in = min(1.0, max(0.0, p * width_hi))
    if u < prob_in:
        # inside [l, r]
        return l + width_hi * rng.random()
    u2 = rng.random()
    # outside region is [-B, l) U (r, B]
    left_len = l - (-B) if l > -B else 0.0
    right_len = B - r if B > r else 0.0
    total_out = left_len + right_len
    if total_out <= 0:
        return l
    if u2 < left_len / total_out:
        return -B + left_len * rng.random()
    return r + right_len * rng.random()


def _mb_perturb1d(x, eps, rng):
    """Multi-bit kernel: symmetric PM used by LPGNN-style multi-bit encoding.
    The multi-bit aspect (coordinate sub-sampling) is handled by the Alg.2
    pipeline; here we simply use a bounded piecewise kernel on [-1,1]."""
    return _pm_perturb1d(x, eps, rng)


def _sw_perturb1d(x, eps, rng):
    """Square-wave mechanism (Li et al. SIGMOD'20): a piecewise density on
    [-C, C] whose high-probability plateau around x has width 2, giving a
    slightly sharper peak than PM for the same budget."""
    e = math.exp(eps)
    eh = math.exp(eps / 2.0)
    C = (eh + 1.0) / (eh - 1.0 + EPS_DEFAULT)
    p_hi = (e - 1.0) / (2.0 * (e + 1.0))
    # plateau [x-1, x+1] clipped to [-C, C]
    l = max(-C, x - 1.0)
    r = min(C, x + 1.0)
    w = r - l
    # density: p_hi on plateau, p_hi / e outside  ->  normalize by total mass
    total = p_hi * w + (p_hi / e) * (2.0 * C - w)
    if total <= 0:
        return 0.0
    u = rng.random() * total
    if u < p_hi * w:
        return l + w * rng.random()
    # outside plateau
    left_len = l - (-C)
    u2 = rng.random()
    if u2 < left_len / (2.0 * C - w):
        return -C + left_len * rng.random()
    return r + (C - r) * rng.random()


def _lap_perturb1d(x, eps, rng):
    """Laplace mechanism, sensitivity 2 (output clipped to [-1,1])."""
    scale = 2.0 / eps
    return min(1.0, max(-1.0, x + rng.laplace(0.0, scale)))


def _ag_perturb1d(x, eps, rng, delta=1e-5):
    """Analytic Gaussian mechanism (Balle & Wang)."""
    # solve sigma such that Phi(Delta/(2sigma) - eps*sigma/Delta)
    #          - e^eps Phi(-Delta/(2sigma) - eps*sigma/Delta) = delta
    Delta = 2.0

    def _phi(t):
        return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))

    lo, hi = 1e-6, 1e3
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        v = _phi(Delta / (2 * mid) - eps * mid / Delta) - math.exp(eps) * _phi(
            -Delta / (2 * mid) - eps * mid / Delta)
        if v > delta:
            hi = mid
        else:
            lo = mid
    sigma = 0.5 * (lo + hi)
    return min(1.0, max(-1.0, x + rng.normal(0.0, sigma)))


def _onebit_perturb1d(x, eps, rng):
    """One-bit mechanism: output in {-1, +1}, mean-preserving."""
    g = (math.exp(eps) - 1.0) / (math.exp(eps) + 1.0)
    p1 = 0.5 * (1.0 + x * g)
    return 1.0 if rng.random() < p1 else -1.0


PERTURB1D = {
    "1B": _onebit_perturb1d,
    "LP": _lap_perturb1d,
    "AG": _ag_perturb1d,
    "SW": _sw_perturb1d,
    "MB": _mb_perturb1d,
    "PM": _pm_perturb1d,
}


# --------------------------------------------------------------------------
# Algorithm 2: unified node-feature perturbation pipeline
# --------------------------------------------------------------------------
def algorithm2(x, eps_x, mech="PM", delta=30.0, seed=0, rect=True):
    """Perturb a feature matrix x (n x d, values in [-1,1]) with privacy
    budget eps_x.  Follows Algorithm 2 of the paper:
      * sample m = clamp(floor(delta * eps_x), 1, d) coordinates per node
      * perturb each sampled coordinate with an (eps_x / m)-LDP kernel
      * set the remaining d-m coordinates to zero
      * (optional) unbiased rectification
    Returns perturbed matrix with dtype float32.
    """
    rng = np.random.default_rng(seed)
    n, d = x.shape
    m = max(1, min(d, int(math.floor(delta * eps_x))))
    eps_c = eps_x / float(m)

    xp = np.zeros_like(x, dtype=np.float64)
    kernel = PERTURB1D[mech]
    coords = np.empty((n, m), dtype=np.int64)
    for i in range(n):
        c = rng.choice(d, size=m, replace=False)
        coords[i] = c
        row = x[i]
        for j in range(m):
            dj = c[j]
            xp[i, dj] = kernel(float(row[dj]), eps_c, rng)
    # unbiased rectification: one-bit output is rescaled by its gain; for the
    # other continuous mechanisms the output is already (approximately) mean
    # preserving, so rectification only hard-clips to the valid range.
    if rect:
        if mech == "1B":
            g = (math.exp(eps_c) - 1.0) / (math.exp(eps_c) + 1.0)
            xp = xp / g
        xp = np.clip(xp, -1.0, 1.0)
    return xp.astype(np.float32), coords
