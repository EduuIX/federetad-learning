import numpy as np


# ------------------------- Energy Distance (ED) ------------------------- #

def _avg_l2(X, Y):
    """Average pairwise L2 distance between rows of X and rows of Y."""
    return np.mean(np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1))

def energy_distance(sample_A, sample_B):
    """
    Energy distance between two samples (arrays of shape [k, d]).
    Returns a single scalar: 2E||X - Y|| - E||X - X'|| - E||Y - Y'||
    """
    A = np.vstack(sample_A) if isinstance(sample_A, list) else np.asarray(sample_A)
    B = np.vstack(sample_B) if isinstance(sample_B, list) else np.asarray(sample_B)
    return 2 * _avg_l2(A, B) - _avg_l2(A, A) - _avg_l2(B, B)


class DFVerifier:
    """
    Data-free verifier using Energy Distance (ED).
    - baseline_updates: array-like, shape (k0, d), collected during warm-up.
    - thresh: confirmation threshold on the ED statistic.
    """
    def __init__(self, baseline_updates, thresh):
        self.baseline = np.asarray(baseline_updates)
        self.thresh = float(thresh)

    def confirm(self, current_updates):
        cur = np.asarray(current_updates)
        stat = energy_distance(cur, self.baseline)
        return (stat > self.thresh), float(stat)


# ------------------------- KL Divergence (diagonal Gaussian) ------------------------- #

def _fit_diag_gaussian(X, eps: float = 1e-6):
    """
    Fit a diagonal Gaussian N(mu, diag(var)) to rows of X (shape [k, d]).
    Returns (mu, var). Adds eps to var for numerical stability.
    """
    X = np.asarray(X)
    mu = X.mean(axis=0)
    var = X.var(axis=0) + eps
    return mu, var

def _kl_diag_gaussians(mu0, var0, mu1, var1):
    """
    KL( N0 || N1 ) for diagonal covariances.
    0.5 * sum[ log(var1/var0) + (var0 + (mu0-mu1)^2)/var1 - 1 ]
    """
    term = np.log(var1 / var0) + (var0 + (mu0 - mu1) ** 2) / var1 - 1.0
    return 0.5 * np.sum(term)

def _sym_kl_diag_gaussians(mu0, var0, mu1, var1):
    """Symmetrized KL: KL(N0||N1) + KL(N1||N0)."""
    return _kl_diag_gaussians(mu0, var0, mu1, var1) + _kl_diag_gaussians(mu1, var1, mu0, var0)

class KLVerifier:
    """
    Data-free verifier using (symmetrized) KL divergence between
    diagonal-Gaussian fits of baseline vs current update sketches.

    - baseline_updates/current_updates: arrays of shape (k, d)
    - thresh: confirmation threshold on the (sym) KL statistic
    - symmetric: if True, use KL(p||q) + KL(q||p); else use KL(current||baseline)
    """
    def __init__(self, baseline_updates, thresh, eps: float = 1e-6, symmetric: bool = True):
        self.baseline = np.asarray(baseline_updates)
        self.thresh = float(thresh)
        self.eps = float(eps)
        self.symmetric = bool(symmetric)

        # Fit baseline distribution once
        self.mu_b, self.var_b = _fit_diag_gaussian(self.baseline, eps=self.eps)

    def confirm(self, current_updates):
        cur = np.asarray(current_updates)
        mu_c, var_c = _fit_diag_gaussian(cur, eps=self.eps)
        if self.symmetric:
            stat = _sym_kl_diag_gaussians(mu_c, var_c, self.mu_b, self.var_b)
        else:
            stat = _kl_diag_gaussians(mu_c, var_c, self.mu_b, self.var_b)
        return (stat > self.thresh), float(stat)