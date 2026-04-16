"""
gp_regression.py
----------------
Sparse Gaussian Process Regression over mutation space.

Learns residual corrections in ln(KIE) space after the physics + GNN pipeline,
using a physics-informed product kernel:

  k(x_i, x_j) = σ² × k_residue(x_i, x_j)
                   × k_mechanism(x_i, x_j)
                   × k_magnitude(x_i, x_j)

where:
  k_residue  — squared exponential over (axis_distance, ENM participation):
               similarity captures "same neighbourhood on the D-A axis"
  k_mechanism — categorical similarity over mutation mechanism type:
               dynamic mutations near the D-A axis form a coherent family
  k_magnitude — squared exponential over (rigidity_change, vol_change,
               hbond_disruption, dynamic_importance):
               similarity captures "same kind of perturbation to the network"

Sparse approximation (DTC — Deterministic Training Conditional):
  Given N training points X and M << N inducing points Z:
    K_MM  = k(Z, Z)   — M×M kernel at inducing points
    K_NM  = k(X, Z)   — N×M cross-kernel
    Λ     = K_MM + σ_n^{-2} K_MN K_NM    — M×M (cheap to invert)
    α     = σ_n^{-2} Λ^{-1} K_MN y       — M vector of dual variables
    μ(x*) = K_{*M} α                       — O(M) per test point
    σ²(x*)= k(x*,x*) − K_{*M} K_MM^{-1} K_{M*}
            + K_{*M} Σ_M K_{M*}            — posterior variance (M×M ops)

Hyperparameter optimisation: 2-stage grid search over (l_r, l_m, σ_s, σ_n)
maximising the DTC log marginal likelihood.  No external optimisers — NumPy only.

Target variable: y = ln(KIE_exp) − ln(KIE_physics+GNN)
Output: posterior mean (gpr_delta) and variance (gpr_variance) per mutation.

References
----------
Snelson E, Ghahramani Z (2006) NIPS 19:1257. — Sparse GP / FITC
Rasmussen CE, Williams CKI (2006) Gaussian Processes for ML. — MIT Press
Johannissen et al. (2007) FEBS J 278:1701. — AADH dynamics reference
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES   = 6   # number of continuous features per mutation
MAX_INDUCING = 20  # maximum inducing points (switch to exact GP when N ≤ M)

# LOO cross-validation (T172 series, 4 mutations) yields LOO-R² = 0.622.
# With n=4 the GP interpolates the training set but cannot generalise reliably.
# Gate: require at least this many calibration mutations before activating GPR.
# Rationale: n=8 (2× current) is the minimum where the DTC marginal likelihood
# can distinguish length scales and noise reliably in a 6D feature space.
MIN_CALIBRATION_GPR = 8

# Mechanism index mapping
MECHANISM_IDS: Dict[str, int] = {
    'static':    0,
    'dynamic':   1,
    'breathing': 2,
    'mixed':     3,
}

# Mechanism categorical similarity matrix (4×4)
# Encodes domain knowledge: dynamic and mixed are closely related;
# static and breathing are partially related (both affect geometry/flexibility);
# static and dynamic are most distinct (different physics).
_MECH_SIM = np.array([
    #  static  dynamic  breathing  mixed
    [1.00,    0.10,    0.50,      0.35],   # static
    [0.10,    1.00,    0.30,      0.65],   # dynamic
    [0.50,    0.30,    1.00,      0.45],   # breathing
    [0.35,    0.65,    0.45,      1.00],   # mixed
])

# Kernel hyperparameter defaults — serve as grid-search starting point
DEFAULT_LENGTH_RESIDUE   = 3.0   # Å, length scale for axis_distance & ENM
DEFAULT_LENGTH_MAGNITUDE = 0.5   # length scale for mutation magnitude features
DEFAULT_SIGMA_SIGNAL     = 1.0   # signal variance σ_s
DEFAULT_SIGMA_NOISE      = 0.30  # noise standard deviation σ_n (in ln(KIE) units)


# ── Feature encoding ──────────────────────────────────────────────────────────

@dataclass
class GPRFeature:
    """
    Physics-derived feature vector for one mutation.

    Two sub-vectors are used separately by the product kernel:
      residue_features  — where the mutation is relative to the D-A axis
      magnitude_features — what kind of perturbation it causes
    """
    axis_distance:      float   # Å from D-A axis (0–8)
    enm_participation:  float   # ENM participation score [0, 1]
    mechanism_id:       int     # 0=static, 1=dynamic, 2=breathing, 3=mixed
    rigidity_change:    float   # AA_RIGIDITY[new] − AA_RIGIDITY[orig]
    vol_change_norm:    float   # (vol_new − vol_orig) / 100 Å³ (normalised)
    hbond_disruption:   float   # [0, 1]
    dynamic_importance: float   # [0, 1]

    @property
    def residue_features(self) -> np.ndarray:
        """2D position feature: (axis_distance, enm_participation)."""
        return np.array([self.axis_distance, self.enm_participation], dtype=float)

    @property
    def magnitude_features(self) -> np.ndarray:
        """4D perturbation feature: (δrigidity, δvol, hbond_disrupt, dyn_importance)."""
        return np.array([
            self.rigidity_change,
            self.vol_change_norm,
            self.hbond_disruption,
            self.dynamic_importance,
        ], dtype=float)

    def to_array(self) -> np.ndarray:
        """Flat 6D feature vector for K-means inducing point selection."""
        return np.concatenate([self.residue_features, self.magnitude_features])


def extract_gpr_feature(score) -> 'GPRFeature':
    """
    Extract a GPRFeature from a MutationScore object.

    Parameters
    ----------
    score : MutationScore
        From tunnel_score.TunnelScorer.score_mutation()

    Returns
    -------
    GPRFeature
    """
    from breathing import AA_RIGIDITY
    from tunnel_score import AA_VOLUME

    r_orig = AA_RIGIDITY.get(score.orig_aa, 0.4)
    r_new  = AA_RIGIDITY.get(score.new_aa,  0.4)
    v_orig = AA_VOLUME.get(score.orig_aa, 130.0)
    v_new  = AA_VOLUME.get(score.new_aa,  130.0)

    mech    = score.dominant_mechanism
    mech_id = MECHANISM_IDS.get(mech, 3)  # default to 'mixed' if unknown

    return GPRFeature(
        axis_distance     = float(score.axis_distance),
        enm_participation = float(score.enm_participation),
        mechanism_id      = mech_id,
        rigidity_change   = float(r_new - r_orig),
        vol_change_norm   = float((v_new - v_orig) / 100.0),
        hbond_disruption  = float(score.hbond_disruption),
        dynamic_importance= float(score.dynamic_importance),
    )


# ── Kernel functions ──────────────────────────────────────────────────────────

def _k_residue(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> float:
    """Squared exponential kernel over residue position features."""
    diff = x1 - x2
    return float(np.exp(-0.5 * np.dot(diff, diff) / (length_scale ** 2)))


def _k_mechanism(m1: int, m2: int) -> float:
    """Categorical similarity kernel based on physics mechanism."""
    m1 = int(np.clip(m1, 0, 3))
    m2 = int(np.clip(m2, 0, 3))
    return float(_MECH_SIM[m1, m2])


def _k_magnitude(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> float:
    """Squared exponential kernel over mutation magnitude features."""
    diff = x1 - x2
    return float(np.exp(-0.5 * np.dot(diff, diff) / (length_scale ** 2)))


def compute_kernel(
    f1: GPRFeature,
    f2: GPRFeature,
    sigma_signal:      float = DEFAULT_SIGMA_SIGNAL,
    length_residue:    float = DEFAULT_LENGTH_RESIDUE,
    length_magnitude:  float = DEFAULT_LENGTH_MAGNITUDE,
) -> float:
    """
    Full physics-informed product kernel:

      k(x_i, x_j) = σ_s² × k_residue × k_mechanism × k_magnitude
    """
    kr = _k_residue(f1.residue_features, f2.residue_features, length_residue)
    km = _k_mechanism(f1.mechanism_id, f2.mechanism_id)
    kg = _k_magnitude(f1.magnitude_features, f2.magnitude_features, length_magnitude)
    return float(sigma_signal ** 2 * kr * km * kg)


def build_covariance_matrix(
    features1:        List[GPRFeature],
    features2:        Optional[List[GPRFeature]] = None,
    sigma_signal:     float = DEFAULT_SIGMA_SIGNAL,
    length_residue:   float = DEFAULT_LENGTH_RESIDUE,
    length_magnitude: float = DEFAULT_LENGTH_MAGNITUDE,
) -> np.ndarray:
    """
    Build N×M covariance matrix K(features1, features2).

    If features2 is None, builds the N×N matrix K(features1, features1)
    (symmetric, use for training kernel).
    """
    n  = len(features1)
    fs = features2 if features2 is not None else features1
    m  = len(fs)
    K  = np.zeros((n, m))
    for i, fi in enumerate(features1):
        for j, fj in enumerate(fs):
            K[i, j] = compute_kernel(fi, fj, sigma_signal, length_residue, length_magnitude)
    return K


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class GPRResult:
    """Prediction from the Sparse GP for one mutation."""
    mean:      float   # posterior mean correction in ln(KIE) space
    variance:  float   # posterior variance (uncertainty estimate)
    std:       float   # posterior standard deviation
    gpr_delta: float   # alias for mean; for compatibility with GNN naming

    def __post_init__(self):
        self.std       = float(np.sqrt(max(self.variance, 0.0)))
        self.gpr_delta = self.mean


@dataclass
class GPRModel:
    """Fitted sparse GP model — stores all state needed for prediction."""
    features_train:    List[GPRFeature]
    y_train:           np.ndarray
    inducing_features: List[GPRFeature]

    sigma_signal:      float
    sigma_noise:       float
    length_residue:    float
    length_magnitude:  float

    # Pre-computed dual variables and inverse matrices
    _alpha:    np.ndarray
    _K_MM_inv: np.ndarray
    _Sigma_M:  np.ndarray

    log_marginal_likelihood: float = 0.0
    train_r2:   float = 0.0
    n_train:    int   = 0
    n_inducing: int   = 0
    is_fitted:  bool  = False


# ── Sparse GP class ───────────────────────────────────────────────────────────

class SparseGP:
    """
    Sparse Gaussian Process Regression with physics-informed product kernel.

    Hyperparameters are optimised by 2-stage grid search (coarse then fine)
    over the DTC log marginal likelihood.  Inducing points are selected by
    K-means on the 6D feature space (exact GP when N ≤ MAX_INDUCING).

    Usage
    -----
    gpr = SparseGP()
    gpr.fit(cal_residuals, features_list)
    result = gpr.predict(feature)   # → GPRResult
    """

    def __init__(
        self,
        sigma_signal:     float = DEFAULT_SIGMA_SIGNAL,
        sigma_noise:      float = DEFAULT_SIGMA_NOISE,
        length_residue:   float = DEFAULT_LENGTH_RESIDUE,
        length_magnitude: float = DEFAULT_LENGTH_MAGNITUDE,
        max_inducing:     int   = MAX_INDUCING,
        optimize_hp:      bool  = True,
    ):
        self.sigma_signal     = sigma_signal
        self.sigma_noise      = sigma_noise
        self.length_residue   = length_residue
        self.length_magnitude = length_magnitude
        self.max_inducing     = max_inducing
        self.optimize_hp      = optimize_hp
        self._model: Optional[GPRModel] = None

    # ── Inducing point selection ──────────────────────────────────────────────

    def _select_inducing(self, features: List[GPRFeature]) -> List[GPRFeature]:
        """
        Select M inducing points by K-means on the 6D feature space.
        Returns all N points unchanged when N ≤ max_inducing (exact GP).
        """
        n = len(features)
        M = min(n, self.max_inducing)
        if M >= n:
            return list(features)

        X = np.array([f.to_array() for f in features])

        # K-means initialisation (NumPy only, deterministic seed)
        rng     = np.random.RandomState(42)
        idx     = rng.choice(n, M, replace=False)
        centers = X[idx].copy()

        for _ in range(100):
            dists  = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = dists.argmin(axis=1)
            new_c  = np.array([
                X[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
                for k in range(M)
            ])
            if np.allclose(centers, new_c, atol=1e-8):
                break
            centers = new_c

        # Represent each cluster by the nearest actual training feature
        inducing, seen = [], set()
        for k in range(M):
            mask = labels == k
            if not mask.any():
                continue
            idxs        = np.where(mask)[0]
            dists_to_c  = np.sum((X[idxs] - centers[k]) ** 2, axis=1)
            best         = idxs[int(np.argmin(dists_to_c))]
            if best not in seen:
                inducing.append(features[best])
                seen.add(best)
        return inducing

    # ── DTC log marginal likelihood ───────────────────────────────────────────

    def _dtc_lml(
        self,
        features: List[GPRFeature],
        inducing: List[GPRFeature],
        y:        np.ndarray,
        ss:       float,
        sn:       float,
        lr:       float,
        lm:       float,
    ) -> float:
        """
        DTC log marginal likelihood for hyperparameter grid search.

        log p(y) ≈ -½ y^T Λ̃^{-1} y  −  ½ log|Λ̃|  −  n/2 log(2π)

        where Λ̃ = σ_n² I + K_NM K_MM^{-1} K_MN  (Nyström approximation).

        Computed via Woodbury identity in M×M space.
        """
        n  = len(features)
        M  = len(inducing)
        sn2 = sn ** 2

        try:
            K_MM  = build_covariance_matrix(inducing, None,     ss, lr, lm)
            K_NM  = build_covariance_matrix(features, inducing, ss, lr, lm)
            K_MN  = K_NM.T

            K_MM_r = K_MM + 1e-6 * np.eye(M)
            L_M    = np.linalg.cholesky(K_MM_r)

            # Λ = K_MM + σ_n^{-2} K_MN K_NM
            Lambda   = K_MM + (1.0 / sn2) * K_MN @ K_NM
            Lambda_r = Lambda + 1e-6 * np.eye(M)
            L_Lam    = np.linalg.cholesky(Lambda_r)

            # log|Λ̃| = log|K_MM^{-1}| + log|Λ| + n·log(σ_n²)
            log_det = (
                2.0 * np.sum(np.log(np.diag(L_Lam)))
                - 2.0 * np.sum(np.log(np.diag(L_M)))
                + n * np.log(sn2)
            )

            # y^T Λ̃^{-1} y via Woodbury:
            # Λ̃^{-1} = σ_n^{-2} I − σ_n^{-4} K_NM Λ^{-1} K_MN
            v      = np.linalg.solve(L_Lam, K_MN @ y)
            yQinvy = (np.dot(y, y) - (1.0 / sn2) * np.dot(v, v)) / sn2

            lml = -0.5 * (yQinvy + log_det + n * np.log(2.0 * np.pi))
            return float(lml)
        except np.linalg.LinAlgError:
            return -1e10

    # ── Hyperparameter optimisation ───────────────────────────────────────────

    def _optimise_hp(
        self,
        features: List[GPRFeature],
        inducing: List[GPRFeature],
        y:        np.ndarray,
        verbose:  bool = False,
    ) -> Tuple[float, float, float, float]:
        """
        Two-stage grid search over (length_residue, length_magnitude,
        sigma_signal, sigma_noise) maximising DTC log marginal likelihood.
        """
        # Stage 1: coarse grid (4×4×4×4 = 256 evaluations)
        lr_c  = [0.5, 1.5, 4.0, 10.0]
        lm_c  = [0.1, 0.3, 0.8, 2.0]
        ss_c  = [0.3, 0.7, 1.5, 3.0]
        sn_c  = [0.05, 0.15, 0.35, 0.70]

        best_lml = -1e10
        best     = (self.length_residue, self.length_magnitude,
                    self.sigma_signal,   self.sigma_noise)

        for lr in lr_c:
            for lm in lm_c:
                for ss in ss_c:
                    for sn in sn_c:
                        lml = self._dtc_lml(features, inducing, y, ss, sn, lr, lm)
                        if lml > best_lml:
                            best_lml = lml
                            best     = (lr, lm, ss, sn)

        lr0, lm0, ss0, sn0 = best

        # Stage 2: fine grid around best (5×5×5×5 = 625 evaluations)
        def refine(v, factors):
            return sorted(set(max(1e-6, v * f) for f in factors))

        fs = [0.4, 0.7, 1.0, 1.4, 2.0]
        lr_f = refine(lr0, fs)
        lm_f = refine(lm0, fs)
        ss_f = refine(ss0, fs)
        sn_f = refine(sn0, fs)

        for lr in lr_f:
            for lm in lm_f:
                for ss in ss_f:
                    for sn in sn_f:
                        lml = self._dtc_lml(features, inducing, y, ss, sn, lr, lm)
                        if lml > best_lml:
                            best_lml = lml
                            best     = (lr, lm, ss, sn)

        if verbose:
            lr_f, lm_f, ss_f, sn_f = best
            print(f"  HP optim: l_r={lr_f:.3f}  l_m={lm_f:.3f}  "
                  f"σ_s={ss_f:.3f}  σ_n={sn_f:.3f}  LML={best_lml:.4f}")

        return best

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        calibration_residuals: List[Tuple],
        features_list:         List[GPRFeature],
        verbose:               bool = True,
    ) -> GPRModel:
        """
        Fit the sparse GP on calibration residuals.

        Parameters
        ----------
        calibration_residuals : list of (key, orig_aa, new_aa, residual)
            Each residual = ln(KIE_exp) − ln(KIE_physics+GNN) for one mutation.
        features_list : list of GPRFeature
            Feature vectors in the same order as calibration_residuals.
        verbose : bool

        Returns
        -------
        GPRModel (also stored as self._model)
        """
        n = len(calibration_residuals)
        y = np.array([r[3] for r in calibration_residuals], dtype=float)

        if verbose:
            print(f"  Fitting sparse GP: n={n} training points")

        inducing = self._select_inducing(features_list)
        M        = len(inducing)

        if verbose:
            print(f"  Inducing points: M={M}")

        # Optimise hyperparameters
        if self.optimize_hp and n >= 2:
            lr, lm, ss, sn = self._optimise_hp(
                features_list, inducing, y, verbose=verbose)
        else:
            lr, lm, ss, sn = (self.length_residue, self.length_magnitude,
                               self.sigma_signal,   self.sigma_noise)

        sn2 = sn ** 2

        # Build kernel matrices
        K_MM  = build_covariance_matrix(inducing,      None,     ss, lr, lm)
        K_NM  = build_covariance_matrix(features_list, inducing, ss, lr, lm)
        K_MN  = K_NM.T
        K_MM_r = K_MM + 1e-6 * np.eye(M)

        # Λ = K_MM + σ_n^{-2} K_MN K_NM
        Lambda   = K_MM + (1.0 / sn2) * K_MN @ K_NM
        Lambda_r = Lambda + 1e-6 * np.eye(M)

        try:
            Sigma_M = np.linalg.inv(Lambda_r)
        except np.linalg.LinAlgError:
            Sigma_M = np.eye(M) * 1e-3

        try:
            K_MM_inv = np.linalg.inv(K_MM_r)
        except np.linalg.LinAlgError:
            K_MM_inv = np.eye(M) * 1e-3

        # Dual variables: α = σ_n^{-2} Σ_M K_MN y
        alpha = (1.0 / sn2) * Sigma_M @ K_MN @ y

        # Log marginal likelihood at fitted hyperparameters
        lml = self._dtc_lml(features_list, inducing, y, ss, sn, lr, lm)

        # Training R² in residual space
        mu_train = K_NM @ alpha
        ss_res   = float(np.sum((y - mu_train) ** 2))
        ss_tot   = float(np.sum((y - y.mean()) ** 2))
        r2       = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        if verbose:
            print(f"  Train R²(residuals) = {r2:.4f}  |  LML = {lml:.4f}")

        model = GPRModel(
            features_train    = features_list,
            y_train           = y,
            inducing_features = inducing,
            sigma_signal      = ss,
            sigma_noise       = sn,
            length_residue    = lr,
            length_magnitude  = lm,
            _alpha            = alpha,
            _K_MM_inv         = K_MM_inv,
            _Sigma_M          = Sigma_M,
            log_marginal_likelihood = lml,
            train_r2          = r2,
            n_train           = n,
            n_inducing        = M,
            is_fitted         = True,
        )
        self._model = model
        return model

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, feature: GPRFeature) -> GPRResult:
        """
        Predict posterior mean and variance for one mutation.

        Returns GPRResult with mean=0, variance=σ_n² if model is not fitted.
        """
        if self._model is None or not self._model.is_fitted:
            return GPRResult(
                mean=0.0,
                variance=float(self.sigma_noise ** 2),
                std=self.sigma_noise,
                gpr_delta=0.0,
            )

        m  = self._model
        ss = m.sigma_signal
        lr = m.length_residue
        lm = m.length_magnitude

        # K_{*M}: covariance between test point and inducing points (M-vector)
        K_sM = np.array([
            compute_kernel(feature, z, ss, lr, lm)
            for z in m.inducing_features
        ])

        # Posterior mean: μ(x*) = K_{*M} α
        mu = float(K_sM @ m._alpha)

        # Posterior variance: σ²(x*) = k(x*,x*) − K_{*M} K_MM^{-1} K_{M*}
        #                              + K_{*M} Σ_M K_{M*}
        k_ss     = compute_kernel(feature, feature, ss, lr, lm)
        var_sub  = float(K_sM @ m._K_MM_inv @ K_sM)
        var_post = float(K_sM @ m._Sigma_M   @ K_sM)
        variance = max(k_ss - var_sub + var_post, 0.0)

        return GPRResult(mean=mu, variance=variance, std=float(np.sqrt(variance)),
                         gpr_delta=mu)

    def is_fitted(self) -> bool:
        return self._model is not None and self._model.is_fitted


# ── Integration helpers ───────────────────────────────────────────────────────

def compute_gpr_residuals_from_scan(
    all_scores,
    kie_data,
) -> List[Tuple]:
    """
    Extract calibration residuals (post-GNN) from scan results.

    Returns list of (key, orig_aa, new_aa, residual) where
    residual = ln(KIE_exp) − ln(KIE_predicted_including_gnn_delta).

    Parameters
    ----------
    all_scores : list of MutationScore
    kie_data   : list of KIEDataPoint
    """
    import math
    score_map = {}
    for sc in all_scores:
        score_map[(sc.chain, sc.residue_number, sc.new_aa)] = sc

    residuals = []
    for dp in kie_data:
        if dp.new_aa == 'WT':
            continue
        key = (dp.chain, dp.residue, dp.new_aa)
        sc  = score_map.get(key)
        if sc is None:
            # Try other chains
            for sc2 in all_scores:
                if sc2.residue_number == dp.residue and sc2.new_aa == dp.new_aa:
                    sc = sc2
                    break
        if sc is None or sc.predicted_kie <= 0:
            continue
        residual = math.log(dp.kie_298k) - math.log(sc.predicted_kie)
        residuals.append(((sc.chain, sc.residue_number), sc.orig_aa, sc.new_aa, residual))
    return residuals


def build_gpr_model(
    all_scores,
    calibration_residuals: List[Tuple],
    verbose: bool = True,
) -> SparseGP:
    """
    Build and fit a Sparse GP model from scan results and calibration residuals.

    Parameters
    ----------
    all_scores : list of MutationScore
        Full scan results (for feature extraction).
    calibration_residuals : list of (key, orig_aa, new_aa, residual)
        Post-GNN residuals (from compute_gpr_residuals_from_scan).
    verbose : bool

    Returns
    -------
    Fitted SparseGP instance.
    """
    score_map = {}
    for sc in all_scores:
        score_map[(sc.chain, sc.residue_number, sc.new_aa)] = sc

    features_list  = []
    valid_residuals = []

    for (key, orig_aa, new_aa, residual) in calibration_residuals:
        chain, resnum = key
        sc = score_map.get((chain, resnum, new_aa))
        if sc is None:
            for sc2 in all_scores:
                if sc2.residue_number == resnum and sc2.new_aa == new_aa:
                    sc = sc2
                    break
        if sc is None:
            continue
        features_list.append(extract_gpr_feature(sc))
        valid_residuals.append((key, orig_aa, new_aa, residual))

    if len(valid_residuals) < 2:
        if verbose:
            print("  GPR: insufficient calibration data — returning unfitted model")
        return SparseGP(optimize_hp=False)

    gpr = SparseGP(optimize_hp=True)
    gpr.fit(valid_residuals, features_list, verbose=verbose)
    return gpr
