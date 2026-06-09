from __future__ import annotations
import numpy as np
import logging

log = logging.getLogger(__name__)


def update_gpr(existing_X: np.ndarray, existing_y_kie: np.ndarray,
               new_delta_dg_values: np.ndarray,
               new_classical_barriers: np.ndarray,
               new_labels=None):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    existing_X = np.asarray(existing_X)
    existing_y_kie = np.asarray(existing_y_kie)
    new_delta_dg = np.asarray(new_delta_dg_values).reshape(-1, 1)
    new_classical = np.asarray(new_classical_barriers).reshape(-1, 1)

    if new_labels is not None:
        new_y = np.asarray(new_labels)
    else:
        new_y = new_delta_dg.ravel()

    if existing_X.ndim == 1:
        existing_X = existing_X.reshape(-1, 1)

    # Augment features with delta_delta_G
    new_X = np.hstack([new_classical, new_delta_dg])
    if existing_X.shape[1] == new_X.shape[1]:
        X_combined = np.vstack([existing_X, new_X])
        y_combined = np.concatenate([existing_y_kie, new_y])
    else:
        X_combined = new_X
        y_combined = new_y

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

    if len(X_combined) < 2:
        log.warning("Too few data points for GPR training")
        return gpr, float("nan"), float("nan")

    gpr.fit(X_combined, y_combined)

    # LOO-R² for KIE predictions
    loo_r2_kie = _loo_r2(gpr, X_combined, y_combined)
    # LOO-R² for delta_dg (second column)
    if X_combined.shape[1] > 1:
        loo_r2_delta_dg = _loo_r2(gpr, X_combined[:, 1:2], y_combined)
    else:
        loo_r2_delta_dg = float("nan")

    return gpr, loo_r2_kie, loo_r2_delta_dg


def _loo_r2(model, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    n = len(y)
    if n < 3:
        return float("nan")

    residuals = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = X[mask]
        y_train = y[mask]
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        gpr_i = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        try:
            gpr_i.fit(X_train, y_train)
            pred = gpr_i.predict(X[i:i+1])[0]
        except Exception:
            pred = y_train.mean()
        residuals.append(y[i] - pred)

    residuals = np.array(residuals)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    if ss_tot < 1e-14:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)
